"""Datasets for training Stack cell-set response classifiers."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import h5py
import numpy as np
import torch

from ..data.gene_processing import safe_decode_array
from ..sample_id_sampling import GroupedSample, GroupedTestSamplerDataset, _ordered_unique

log = logging.getLogger(__name__)

DEFAULT_TEST_SAMPLE_IDS = ("P22", "XGY_P_P05_P", "BD_immune08")
DEFAULT_LABELS = ("MPR", "pCR", "NMPR")


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


class CellSetClassificationDataset(GroupedTestSamplerDataset):
    """Cell-set dataset labeled by sample-level treatment response."""

    def __init__(
        self,
        adata_or_path,
        genelist_path: str,
        *,
        split: str,
        test_sample_ids: Sequence[str] = DEFAULT_TEST_SAMPLE_IDS,
        label_col: str = "response",
        groupby_col: str = "sample_id",
        labels: Sequence[str] = DEFAULT_LABELS,
        sample_size: int = 128,
        max_samples: Optional[int] = None,
        gene_name_col: Optional[str] = None,
        filter_organism: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.split = split
        self.test_sample_ids = set(test_sample_ids)
        self.label_col = label_col
        self.labels = tuple(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.sample_labels: list[int] = []

        super().__init__(
            adata_or_path,
            genelist_path,
            groupby_col=groupby_col,
            sample_size=sample_size,
            mode="eval",
            max_samples=max_samples,
            gene_name_col=gene_name_col,
            filter_organism=filter_organism,
            random_state=random_state,
        )

    def _load_label_values(self) -> np.ndarray:
        if self.is_adata_object:
            adata = self.adata_object
            if self.label_col not in adata.obs:
                raise KeyError(f"Column '{self.label_col}' not found in adata.obs")
            values = adata.obs[self.label_col].astype(str).to_numpy()
        else:
            values = _read_obs_column_from_h5(self.adata_path, self.label_col)

        if len(values) != len(self.human_mask):
            raise ValueError(
                f"Column '{self.label_col}' has {len(values)} rows but adata has {len(self.human_mask)} cells"
            )
        return values[self.human_mask].astype(str)

    def _should_use_group(self, sample_id: str) -> bool:
        is_test_sample = sample_id in self.test_sample_ids
        return is_test_sample if self.split == "test" else not is_test_sample

    def _label_for_group(self, sample_id: str, label_values: np.ndarray) -> int:
        group_labels = label_values[self.group_values == sample_id]
        unique_labels = sorted({label for label in group_labels if label != ""})
        if len(unique_labels) != 1:
            raise ValueError(
                f"Expected exactly one response label for sample_id {sample_id}, got {unique_labels}"
            )

        label = unique_labels[0]
        if label not in self.label_to_idx:
            raise ValueError(
                f"Unsupported label '{label}' for sample_id {sample_id}; expected {self.labels}"
            )
        return self.label_to_idx[label]

    def _generate_samples(self) -> None:
        log.info(
            "Generating %s classifier samples by obs column '%s'...",
            self.split,
            self.groupby_col,
        )

        if self.n_human_cells < 1:
            self.samples = []
            self.sample_labels = []
            return

        self.group_values = self._load_group_values()
        label_values = self._load_label_values()
        human_obs_indices = np.where(self.human_mask)[0]
        self.samples: list[GroupedSample] = []
        self.sample_labels = []
        self.group_obs_indices = {}

        for sample_id in _ordered_unique(self.group_values):
            if not self._should_use_group(sample_id):
                continue

            group_local_indices = np.flatnonzero(self.group_values == sample_id)
            if self.max_samples is not None:
                group_local_indices = group_local_indices[
                    : self.max_samples * self.sample_size
                ]
            if len(group_local_indices) == 0:
                continue

            label_idx = self._label_for_group(sample_id, label_values)
            self.group_obs_indices[sample_id] = human_obs_indices[group_local_indices]
            self._append_group_samples(sample_id, group_local_indices, label_idx)

        log.info("Generated %s %s classifier samples", len(self.samples), self.split)

    def _append_group_samples(
        self,
        sample_id: str,
        group_local_indices: np.ndarray,
        label_idx: int,
    ) -> None:
        n_full_samples = len(group_local_indices) // self.sample_size
        for idx in range(n_full_samples):
            start_idx = idx * self.sample_size
            end_idx = start_idx + self.sample_size
            local_indices_to_load = group_local_indices[start_idx:end_idx]
            self.samples.append(
                GroupedSample(sample_id, local_indices_to_load, np.arange(self.sample_size))
            )
            self.sample_labels.append(label_idx)

        remaining_count = len(group_local_indices) - n_full_samples * self.sample_size
        if remaining_count <= 0:
            return

        remaining_local_indices = group_local_indices[n_full_samples * self.sample_size :]
        upsampled_relative_indices = self.rng.choice(
            np.arange(remaining_count),
            size=self.sample_size - remaining_count,
            replace=True,
        )
        reindexing_map = np.concatenate(
            [np.arange(remaining_count), upsampled_relative_indices]
        )
        self.samples.append(GroupedSample(sample_id, remaining_local_indices, reindexing_map))
        self.sample_labels.append(label_idx)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        loaded_data = self.load_expression_data(sample.local_indices_to_load)
        expression_data = loaded_data[sample.reindexing_map]
        label_idx = self.sample_labels[idx]
        metadata: Dict[str, Any] = {
            "sample_idx": idx,
            "sample_id": sample.group_value,
            "split": self.split,
            "label": self.labels[label_idx],
            "n_unique_cells_loaded": len(sample.local_indices_to_load),
        }
        return (
            torch.from_numpy(expression_data).float(),
            torch.tensor(label_idx, dtype=torch.long),
            metadata,
        )
