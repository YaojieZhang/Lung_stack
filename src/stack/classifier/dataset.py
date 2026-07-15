"""Datasets for training Stack cell-set response classifiers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch

from ..data.gene_processing import safe_decode_array
from ..sample_id_sampling import GroupedSample, GroupedTestSamplerDataset, _ordered_unique

log = logging.getLogger(__name__)

DEFAULT_TEST_SAMPLE_IDS = ("P22", "XGY_P_P05_P", "BD_immune08")
DEFAULT_LABELS = ("NMPR", "MPR", "pCR")
MISSING_LABEL_VALUES = {"", "NA", "NaN", "nan", "None", "none", "null"}


@dataclass(frozen=True)
class ClassifierFold:
    fold: int
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]


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


def _is_missing_label(value: str) -> bool:
    return str(value).strip() in MISSING_LABEL_VALUES


def read_sample_label_map(
    adata_or_path,
    *,
    label_col: str = "response",
    groupby_col: str = "sample_id",
) -> dict[str, str]:
    """Read one response label per sample from an AnnData object or .h5ad path."""

    if hasattr(adata_or_path, "obs"):
        adata = adata_or_path
        if groupby_col not in adata.obs:
            raise KeyError(f"Column '{groupby_col}' not found in adata.obs")
        if label_col not in adata.obs:
            raise KeyError(f"Column '{label_col}' not found in adata.obs")
        group_values = adata.obs[groupby_col].astype(str).to_numpy()
        label_values = adata.obs[label_col].astype(str).to_numpy()
    else:
        group_values = _read_obs_column_from_h5(str(adata_or_path), groupby_col)
        label_values = _read_obs_column_from_h5(str(adata_or_path), label_col)

    sample_labels: dict[str, str] = {}
    for sample_id, label in zip(group_values.astype(str), label_values.astype(str)):
        if _is_missing_label(label):
            continue
        existing = sample_labels.get(sample_id)
        if existing is None:
            sample_labels[sample_id] = label
        elif existing != label:
            raise ValueError(
                f"Expected one label for sample_id {sample_id}, got {existing!r} and {label!r}"
            )

    return sample_labels


def make_stratified_holdout_folds(
    sample_labels: Mapping[str, str],
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    test_per_class: int = 2,
    n_folds: int = 7,
    val_per_class: int = 1,
    random_state: Optional[int] = 0,
    test_sample_ids: Optional[Sequence[str]] = None,
) -> tuple[ClassifierFold, ...]:
    """Build fixed test sample IDs plus stratified validation folds.

    When a class has fewer remaining samples than ``n_folds * val_per_class``,
    validation samples for that class are cycled to preserve one validation
    sample per class in every fold.
    """

    if test_per_class < 0:
        raise ValueError("test_per_class must be non-negative")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if val_per_class < 1:
        raise ValueError("val_per_class must be at least 1")

    sample_labels = {
        str(sample_id): str(label)
        for sample_id, label in sample_labels.items()
    }
    label_order = tuple(labels)
    label_set = set(label_order)
    ordered_samples = tuple(str(sample_id) for sample_id in sample_labels)
    by_label = {
        label: [sample_id for sample_id in ordered_samples if sample_labels[sample_id] == label]
        for label in label_order
    }

    missing_labels = [label for label, sample_ids in by_label.items() if not sample_ids]
    if missing_labels:
        raise ValueError(f"No samples found for labels: {missing_labels}")

    rng = np.random.default_rng(random_state)
    if test_sample_ids:
        test_ids = tuple(str(sample_id) for sample_id in test_sample_ids)
        unknown = [sample_id for sample_id in test_ids if sample_id not in sample_labels]
        if unknown:
            raise ValueError(f"Unknown test sample_ids: {unknown}")
    else:
        selected: list[str] = []
        for label, sample_ids in by_label.items():
            if len(sample_ids) <= test_per_class:
                raise ValueError(
                    f"Need more than {test_per_class} samples for label {label}, got {len(sample_ids)}"
                )
            shuffled = np.array(sample_ids, dtype=object)
            rng.shuffle(shuffled)
            selected.extend(str(sample_id) for sample_id in shuffled[:test_per_class])
        test_ids = tuple(selected)

    test_id_set = set(test_ids)
    bad_test_labels = [
        sample_id for sample_id in test_ids if sample_labels[sample_id] not in label_set
    ]
    if bad_test_labels:
        raise ValueError(f"Test sample_ids have labels outside {label_order}: {bad_test_labels}")

    val_schedule: dict[str, list[str]] = {}
    for label in label_order:
        candidates = [
            sample_id
            for sample_id in by_label[label]
            if sample_id not in test_id_set
        ]
        if len(candidates) < val_per_class:
            raise ValueError(
                f"Need at least {val_per_class} non-test samples for label {label}, got {len(candidates)}"
            )
        shuffled = np.array(candidates, dtype=object)
        rng.shuffle(shuffled)
        ordered_candidates = [str(sample_id) for sample_id in shuffled]
        required = n_folds * val_per_class
        repeated = [
            ordered_candidates[idx % len(ordered_candidates)]
            for idx in range(required)
        ]
        val_schedule[label] = repeated

    remaining_ids = tuple(
        sample_id for sample_id in ordered_samples if sample_id not in test_id_set
    )
    folds: list[ClassifierFold] = []
    for fold_idx in range(n_folds):
        val_ids: list[str] = []
        start = fold_idx * val_per_class
        end = start + val_per_class
        for label in label_order:
            val_ids.extend(val_schedule[label][start:end])
        val_id_set = set(val_ids)
        train_ids = tuple(
            sample_id for sample_id in remaining_ids if sample_id not in val_id_set
        )
        folds.append(
            ClassifierFold(
                fold=fold_idx,
                train_sample_ids=train_ids,
                val_sample_ids=tuple(val_ids),
                test_sample_ids=test_ids,
            )
        )

    return tuple(folds)


class CellSetClassificationDataset(GroupedTestSamplerDataset):
    """Cell-set dataset labeled by sample-level treatment response."""

    def __init__(
        self,
        adata_or_path,
        genelist_path: str,
        *,
        split: str,
        test_sample_ids: Sequence[str] = DEFAULT_TEST_SAMPLE_IDS,
        val_sample_ids: Sequence[str] = (),
        sample_ids: Optional[Sequence[str]] = None,
        label_col: str = "response",
        groupby_col: str = "sample_id",
        labels: Sequence[str] = DEFAULT_LABELS,
        sample_size: int = 128,
        max_samples: Optional[int] = None,
        gene_name_col: Optional[str] = None,
        filter_organism: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.split = split
        self.test_sample_ids = set(test_sample_ids)
        self.val_sample_ids = set(val_sample_ids)
        self.sample_ids = None if sample_ids is None else set(sample_ids)
        self.label_col = label_col
        self.labels = tuple(labels)
        self._label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.label_to_idx = dict(self._label_to_idx)
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
        if self.sample_ids is not None:
            return sample_id in self.sample_ids

        is_test_sample = sample_id in self.test_sample_ids
        is_val_sample = sample_id in self.val_sample_ids
        if self.split == "test":
            return is_test_sample
        if self.split == "val":
            return is_val_sample
        return not is_test_sample and not is_val_sample

    def _label_for_group(self, sample_id: str, label_values: np.ndarray) -> int:
        group_labels = label_values[self.group_values == sample_id]
        unique_labels = sorted(
            {label for label in group_labels if not _is_missing_label(label)}
        )
        if len(unique_labels) != 1:
            raise ValueError(
                f"Expected exactly one response label for sample_id {sample_id}, got {unique_labels}"
            )

        label = unique_labels[0]
        if label not in self._label_to_idx:
            raise ValueError(
                f"Unsupported label '{label}' for sample_id {sample_id}; expected {self.labels}"
            )
        return self._label_to_idx[label]

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
        present_label_indices = sorted(set(self.sample_labels))
        self.label_to_idx = {
            self.labels[label_idx]: label_idx for label_idx in present_label_indices
        }

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
