"""Sample-id aware embedding sampling utilities."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.gene_processing import safe_decode_array
from .data.training.datasets import TestSamplerDataset

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupedSample:
    group_value: str
    local_indices_to_load: np.ndarray
    reindexing_map: np.ndarray


@dataclass
class GroupedEmbeddingResult:
    embeddings: Dict[str, np.ndarray]
    obs_indices: Dict[str, np.ndarray]


def _read_obs_column_from_h5(path: str, column: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if column not in obs:
            raise KeyError(f"Column '{column}' not found in adata.obs")

        node = obs[column]
        if isinstance(node, h5py.Dataset):
            return safe_decode_array(node[:]).astype(str)

        if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
            categories = safe_decode_array(node["categories"][:]).astype(str)
            codes = node["codes"][:]
            values = np.full(codes.shape, "", dtype=object)
            valid = codes >= 0
            values[valid] = categories[codes[valid]]
            return values.astype(str)

    raise ValueError(f"Cannot read obs column '{column}' from {path}")


def _ordered_unique(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for value in values.astype(str):
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _safe_filename(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "missing"


def resolve_group_output_path(
    base_output_path: Path,
    group_value: str,
    output_suffix: str,
    *,
    group_obs=None,
    patient_col: Optional[str] = None,
) -> Path:
    """Return ``base/group-patient/group.ext`` for split embedding output."""

    base_dir = (
        base_output_path.parent / base_output_path.stem
        if base_output_path.suffix
        else base_output_path
    )
    parts = [base_dir]

    if patient_col and group_obs is not None and patient_col in group_obs:
        patient_values = group_obs[patient_col].dropna().astype(str).unique()
        if len(patient_values) > 0:
            parts.append(Path(_safe_filename(patient_values[0])))

    return Path(*parts) / f"{_safe_filename(group_value)}{output_suffix}"


class GroupedTestSamplerDataset(TestSamplerDataset):
    """Evaluation sampler that never mixes different obs groups in one cell set."""

    def __init__(
        self,
        adata_or_path,
        genelist_path: str,
        *,
        groupby_col: str = "sample_id",
        sample_size: int = 128,
        mode: str = "eval",
        max_samples: Optional[int] = None,
        gene_name_col: Optional[str] = None,
        filter_organism: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.groupby_col = groupby_col
        self.group_values: np.ndarray
        self.group_obs_indices: Dict[str, np.ndarray] = {}
        super().__init__(
            adata_or_path,
            genelist_path,
            sample_size=sample_size,
            mode=mode,
            max_samples=max_samples,
            gene_name_col=gene_name_col,
            filter_organism=filter_organism,
            random_state=random_state,
        )

    def _load_group_values(self) -> np.ndarray:
        if self.is_adata_object:
            adata = self.adata_object
            if self.groupby_col not in adata.obs:
                raise KeyError(f"Column '{self.groupby_col}' not found in adata.obs")
            values = adata.obs[self.groupby_col].astype(str).to_numpy()
        else:
            values = _read_obs_column_from_h5(self.adata_path, self.groupby_col)

        if len(values) != len(self.human_mask):
            raise ValueError(
                f"Column '{self.groupby_col}' has {len(values)} rows but adata has {len(self.human_mask)} cells"
            )
        return values[self.human_mask].astype(str)

    def _generate_samples(self) -> None:
        log.info("Generating grouped samples by obs column '%s'...", self.groupby_col)

        if self.n_human_cells < 1:
            log.warning("No valid human cells found. Dataset will be empty.")
            self.samples = []
            return

        self.group_values = self._load_group_values()
        human_obs_indices = np.where(self.human_mask)[0]
        self.samples: list[GroupedSample] = []
        self.group_obs_indices = {}

        for group_value in _ordered_unique(self.group_values):
            group_local_indices = np.flatnonzero(self.group_values == group_value)
            if self.max_samples is not None:
                group_local_indices = group_local_indices[
                    : self.max_samples * self.sample_size
                ]
            if len(group_local_indices) == 0:
                continue

            self.group_obs_indices[group_value] = human_obs_indices[group_local_indices]
            n_full_samples = len(group_local_indices) // self.sample_size

            for idx in range(n_full_samples):
                start_idx = idx * self.sample_size
                end_idx = start_idx + self.sample_size
                local_indices_to_load = group_local_indices[start_idx:end_idx]
                reindexing_map = np.arange(self.sample_size)
                self.samples.append(
                    GroupedSample(group_value, local_indices_to_load, reindexing_map)
                )

            remaining_count = len(group_local_indices) - n_full_samples * self.sample_size
            if remaining_count > 0:
                remaining_local_indices = group_local_indices[
                    n_full_samples * self.sample_size :
                ]
                need_to_fill = self.sample_size - remaining_count
                upsampled_relative_indices = self.rng.choice(
                    np.arange(remaining_count),
                    size=need_to_fill,
                    replace=True,
                )
                reindexing_map = np.concatenate(
                    [np.arange(remaining_count), upsampled_relative_indices]
                )
                self.samples.append(
                    GroupedSample(group_value, remaining_local_indices, reindexing_map)
                )

        log.info(
            "Generated %s grouped samples across %s groups",
            len(self.samples),
            len(self.group_obs_indices),
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        loaded_data = self.load_expression_data(sample.local_indices_to_load)
        expression_data = loaded_data[sample.reindexing_map]
        features_tensor = torch.from_numpy(expression_data).float()
        metadata = {
            "file_path": self.file_info["path"],
            "sample_idx": idx,
            "sample_size": len(expression_data),
            "n_genes_found": self.file_info["found_genes"],
            "n_unique_cells_loaded": len(sample.local_indices_to_load),
            "groupby_col": self.groupby_col,
            "group_value": sample.group_value,
        }
        return features_tensor, metadata


def _metadata_list(value: Any, batch_size: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted] * batch_size
    return [value] * batch_size


@torch.no_grad()
def extract_grouped_latent_representations(
    model,
    adata_path: str,
    genelist_path: str,
    *,
    groupby_col: str = "sample_id",
    gene_name_col: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = 42,
    filter_organism: bool = True,
) -> GroupedEmbeddingResult:
    """Extract embeddings while keeping each cell set inside one obs group."""

    from tqdm.auto import tqdm

    model.eval()
    dataset = GroupedTestSamplerDataset(
        adata_path,
        genelist_path,
        groupby_col=groupby_col,
        sample_size=model.n_cells,
        mode="eval",
        max_samples=max_samples,
        gene_name_col=gene_name_col,
        filter_organism=filter_organism,
        random_state=random_state,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    iterator = (
        tqdm(dataloader, desc="Extracting grouped embeddings", unit="batch")
        if show_progress
        else dataloader
    )
    grouped_embeddings: Dict[str, list[np.ndarray]] = {}
    device = next(model.parameters()).device

    for features, metadata in iterator:
        features = features.to(device)
        features_log = torch.log1p(features)
        tokens = model._reduce_and_tokenize(features_log)
        x = model._run_attention_layers(tokens)
        batch_embeddings = x.reshape(features.shape[0], features.shape[1], -1)

        group_values = _metadata_list(metadata["group_value"], features.shape[0])
        unique_counts = [
            int(value)
            for value in _metadata_list(
                metadata["n_unique_cells_loaded"], features.shape[0]
            )
        ]
        for batch_idx, group_value in enumerate(group_values):
            n_unique = unique_counts[batch_idx]
            emb = batch_embeddings[batch_idx, :n_unique].cpu().numpy()
            grouped_embeddings.setdefault(str(group_value), []).append(emb)

    embeddings = {
        group_value: np.concatenate(parts, axis=0)
        for group_value, parts in grouped_embeddings.items()
    }
    return GroupedEmbeddingResult(embeddings=embeddings, obs_indices=dataset.group_obs_indices)