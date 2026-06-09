from __future__ import annotations

import numpy as np
import pytest
import sys
import types
from types import SimpleNamespace

h5py_stub = types.ModuleType("h5py")
h5py_stub.File = object
pandas_stub = types.ModuleType("pandas")
anndata_stub = types.ModuleType("anndata")
anndata_stub.AnnData = object
scipy_stub = types.ModuleType("scipy")
sparse_stub = types.ModuleType("scipy.sparse")
sparse_stub.issparse = lambda matrix: False
scipy_stub.sparse = sparse_stub
torch_stub = types.ModuleType("torch")
torch_utils_stub = types.ModuleType("torch.utils")
torch_data_stub = types.ModuleType("torch.utils.data")
torch_distributed_stub = types.ModuleType("torch.distributed")
pytorch_lightning_stub = types.ModuleType("pytorch_lightning")
pl_loggers_stub = types.ModuleType("pytorch_lightning.loggers")


class _Dataset:
    pass


class _DataLoader:
    pass


class _Logger:
    pass


class _LightningDataModule:
    pass


torch_data_stub.Dataset = _Dataset
torch_data_stub.DataLoader = _DataLoader
torch_data_stub.get_worker_info = lambda: None
torch_utils_stub.data = torch_data_stub
torch_stub.utils = torch_utils_stub
torch_stub.from_numpy = lambda array: array
torch_stub.LongTensor = lambda values: np.asarray(values, dtype=np.int64)
torch_stub.stack = lambda values: np.stack(values)
torch_stub.Tensor = np.ndarray
pl_loggers_stub.TensorBoardLogger = _Logger
pl_loggers_stub.WandbLogger = _Logger
pytorch_lightning_stub.LightningDataModule = _LightningDataModule
sys.modules.setdefault("h5py", h5py_stub)
sys.modules.setdefault("anndata", anndata_stub)
sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.sparse", sparse_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.utils", torch_utils_stub)
sys.modules.setdefault("torch.utils.data", torch_data_stub)
sys.modules.setdefault("torch.distributed", torch_distributed_stub)
sys.modules.setdefault("pytorch_lightning", pytorch_lightning_stub)
sys.modules.setdefault("pytorch_lightning.loggers", pl_loggers_stub)
sys.modules.setdefault("stack.finetune.lightning", types.ModuleType("stack.finetune.lightning"))

from stack.cli.launch_finetuning import prepare_lopo_output_paths, validate_model_gene_dimension
from stack.data.finetuning import datasets as datasets_module
from stack.data.finetuning.datasets import DatasetConfig, MultiDatasetSplittableDataset
from stack.finetune.datamodule import FinetuneDataModule, finetune_collate_fn
from stack.finetune.utils import parse_dataset_configs


class FakePairedMetadata:
    def __init__(self):
        identities = []
        conditions = []
        self.paired_group_condition_identity_pool = {}
        self.group_mapping = {
            0: {
                "config_idx": 0,
                "original_id": "P00",
                "dataset_type": "paired",
            }
        }

        def add_pool(condition: str, identity: str, n_cells: int):
            start = len(identities)
            indices = np.arange(start, start + n_cells, dtype=np.int64)
            identities.extend([identity] * n_cells)
            conditions.extend([condition] * n_cells)
            self.paired_group_condition_identity_pool[(0, condition, identity)] = indices

        add_pool("post", "T", 12)
        add_pool("post", "B", 10)
        add_pool("post", "DC", 2)
        add_pool("pre", "T", 6)
        add_pool("pre", "B", 2)
        add_pool("pre", "DC", 1)

        self.cell_identities = np.array(identities, dtype=str)
        self.conditions = np.array(conditions, dtype=str)

    def get_paired_condition_identity_cells(self, group_id, condition, identity):
        return self.paired_group_condition_identity_pool.get(
            (group_id, condition, identity),
            np.array([], dtype=np.int64),
        )

    def get_cell_identities(self, indices):
        return self.cell_identities[indices]

    def get_timepoints(self, indices):
        return self.conditions[indices]


class FakePreflightMetadata:
    def __init__(self, group_specs):
        self.group_mapping = {}
        self.paired_group_condition_identity_pool = {}
        next_index = 0

        for group_id, spec in group_specs.items():
            self.group_mapping[group_id] = {
                "config_idx": 0,
                "original_id": spec["patient"],
                "dataset_type": "paired",
            }
            for identity, counts in spec["types"].items():
                post_count, pre_count = counts
                post_indices = np.arange(next_index, next_index + post_count, dtype=np.int64)
                next_index += post_count
                pre_indices = np.arange(next_index, next_index + pre_count, dtype=np.int64)
                next_index += pre_count
                self.paired_group_condition_identity_pool[(group_id, "post", identity)] = post_indices
                self.paired_group_condition_identity_pool[(group_id, "pre", identity)] = pre_indices

    def get_paired_condition_identity_cells(self, group_id, condition, identity):
        return self.paired_group_condition_identity_pool.get(
            (group_id, condition, identity),
            np.array([], dtype=np.int64),
        )


def make_dataset(method: str, metadata=None):
    ds = object.__new__(MultiDatasetSplittableDataset)
    ds.metadata_cache = metadata or FakePairedMetadata()
    ds.sample_size = 8
    ds.replacement_ratio = 0.5
    ds.paired_sampling_method = method
    ds.rng = np.random.RandomState(0)
    return ds


def make_preflight_dataset(
    group_specs,
    train_groups,
    val_groups,
    test_groups,
    sample_size=8,
    replacement_ratio=0.5,
    split_strategy="lopo",
    method="method1_capacity_strict",
):
    ds = object.__new__(MultiDatasetSplittableDataset)
    ds.metadata_cache = FakePreflightMetadata(group_specs)
    ds.dataset_configs = [make_config()]
    ds.train_groups = train_groups
    ds.val_groups = val_groups
    ds.test_groups = test_groups
    ds.sample_size = sample_size
    ds.replacement_ratio = replacement_ratio
    ds.split_strategy = split_strategy
    ds.paired_sampling_method = method
    ds.rng = np.random.RandomState(0)
    return ds


def make_config():
    return DatasetConfig(
        path="unused",
        type="paired",
        patient_col="Patient",
        timepoint_col="Timepoint",
        cell_type_col="cell_type",
        pre_condition="pre",
        post_condition="post",
    )


def assert_paired_alignment(ds, left_indices, right_indices, n_kept):
    assert len(left_indices) == ds.sample_size
    assert len(right_indices) == ds.sample_size
    assert np.array_equal(left_indices[:n_kept], right_indices[:n_kept])
    assert np.all(ds.metadata_cache.get_timepoints(left_indices) == "post")
    assert np.all(ds.metadata_cache.get_timepoints(right_indices[:n_kept]) == "post")
    assert np.all(ds.metadata_cache.get_timepoints(right_indices[n_kept:]) == "pre")
    assert np.array_equal(
        ds.metadata_cache.get_cell_identities(left_indices[n_kept:]),
        ds.metadata_cache.get_cell_identities(right_indices[n_kept:]),
    )


def test_method1_uses_common_types_without_repeating_pre_cells():
    ds = make_dataset("method1_capacity_strict")

    sample = ds._build_paired_balanced_sample(0, make_config())

    assert sample is not None
    left_indices, right_indices, position_mask, metadata = sample
    n_kept = int((1.0 - ds.replacement_ratio) * ds.sample_size)
    assert_paired_alignment(ds, left_indices, right_indices, n_kept)
    assert len(np.unique(left_indices)) == len(left_indices)
    assert len(np.unique(right_indices[n_kept:])) == len(right_indices[n_kept:])
    assert position_mask.all()
    assert metadata["paired_sampling_method"] == "method1_capacity_strict"


def test_method2_repeats_pre_cells_when_needed():
    metadata = FakePairedMetadata()
    metadata.paired_group_condition_identity_pool[(0, "pre", "T")] = np.array([24], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "B")] = np.array([30], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "DC")] = np.array([32], dtype=np.int64)
    ds = make_dataset("method2_capacity_repeat_pre", metadata)

    sample = ds._build_paired_balanced_sample(0, make_config())

    assert sample is not None
    left_indices, right_indices, _, _ = sample
    n_kept = int((1.0 - ds.replacement_ratio) * ds.sample_size)
    assert_paired_alignment(ds, left_indices, right_indices, n_kept)
    assert len(np.unique(right_indices[n_kept:])) < len(right_indices[n_kept:])


def test_method3_returns_none_when_post_only_plan_exceeds_strict_pre_capacity():
    metadata = FakePairedMetadata()
    metadata.paired_group_condition_identity_pool[(0, "pre", "T")] = np.array([24], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "B")] = np.array([], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "DC")] = np.array([], dtype=np.int64)
    ds = make_dataset("method3_post_only_strict", metadata)

    sample = ds._build_paired_balanced_sample(0, make_config())

    assert sample is None


def test_method4_repeats_pre_cells_for_post_only_plan():
    metadata = FakePairedMetadata()
    metadata.paired_group_condition_identity_pool[(0, "pre", "T")] = np.array([24], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "B")] = np.array([30], dtype=np.int64)
    metadata.paired_group_condition_identity_pool[(0, "pre", "DC")] = np.array([32], dtype=np.int64)
    ds = make_dataset("method4_post_only_repeat_pre", metadata)

    sample = ds._build_paired_balanced_sample(0, make_config())

    assert sample is not None
    left_indices, right_indices, _, _ = sample
    n_kept = int((1.0 - ds.replacement_ratio) * ds.sample_size)
    assert_paired_alignment(ds, left_indices, right_indices, n_kept)
    assert len(np.unique(right_indices[n_kept:])) < len(right_indices[n_kept:])


def test_parse_paired_dataset_config():
    configs = parse_dataset_configs(
        [
            "paired:/data/nsclc:Patient:Timepoint:cell_type:Pre-biopsy:Post-surgery:false:gene_symbols"
        ]
    )

    assert len(configs) == 1
    config = configs[0]
    assert config.type == "paired"
    assert config.path == "/data/nsclc"
    assert config.patient_col == "Patient"
    assert config.timepoint_col == "Timepoint"
    assert config.cell_type_col == "cell_type"
    assert config.pre_condition == "Pre-biopsy"
    assert config.post_condition == "Post-surgery"
    assert config.filter_organism is False
    assert config.gene_name_col == "gene_symbols"


def test_paired_post_prompt_contract_is_explicit():
    ds = make_dataset("method1_capacity_strict")

    sample = ds._build_paired_balanced_sample(0, make_config())

    assert sample is not None
    left_indices, right_indices, _, _ = sample
    n_kept = int((1.0 - ds.replacement_ratio) * ds.sample_size)
    assert np.all(ds.metadata_cache.get_timepoints(left_indices[:n_kept]) == "post")
    assert np.all(ds.metadata_cache.get_timepoints(right_indices[:n_kept]) == "post")
    assert np.all(ds.metadata_cache.get_timepoints(right_indices[n_kept:]) == "pre")


def test_lopo_paired_preflight_accepts_nonempty_splits():
    group_specs = {
        0: {"patient": "P01", "types": {"T": (8, 8)}},
        1: {"patient": "P02", "types": {"T": (8, 8)}},
        2: {"patient": "P03", "types": {"T": (8, 8)}},
    }
    ds = make_preflight_dataset(group_specs, [0], [1], [2])

    ds._validate_paired_preflight()


def test_lopo_paired_preflight_rejects_empty_split():
    group_specs = {
        0: {"patient": "P01", "types": {"T": (8, 8)}},
        1: {"patient": "P02", "types": {"T": (8, 8)}},
    }
    ds = make_preflight_dataset(group_specs, [0], [], [1])

    with pytest.raises(ValueError, match="empty paired split"):
        ds._validate_paired_preflight()


def test_paired_preflight_rejects_sample_size_too_large_for_patient():
    group_specs = {
        0: {"patient": "P09", "types": {"T": (393, 1000)}},
    }
    ds = make_preflight_dataset(
        group_specs,
        [0],
        [],
        [],
        sample_size=512,
        replacement_ratio=0.75,
        split_strategy="random",
    )

    with pytest.raises(ValueError, match="P09.*only 393"):
        ds._validate_paired_preflight()


def test_paired_preflight_rejects_strict_pre_capacity_failure():
    group_specs = {
        0: {"patient": "P01", "types": {"T": (8, 2)}},
    }
    ds = make_preflight_dataset(
        group_specs,
        [0],
        [],
        [],
        sample_size=8,
        replacement_ratio=0.5,
        split_strategy="random",
        method="method1_capacity_strict",
    )

    with pytest.raises(ValueError, match="method1_capacity_strict cannot allocate"):
        ds._validate_paired_preflight()


def test_create_datasets_from_gene_list_passes_lopo_arguments(monkeypatch):
    captured = {}

    def fake_create_train_val_test_datasets(**kwargs):
        captured.update(kwargs)
        return "train", "val", "test"

    monkeypatch.setattr(
        datasets_module,
        "create_train_val_test_datasets",
        fake_create_train_val_test_datasets,
    )

    result = datasets_module.create_datasets_from_gene_list(
        dataset_configs=[make_config()],
        genelist_path="genes.pkl",
        split_strategy="lopo",
        fold_index=2,
        val_fold_offset=3,
    )

    assert result == ("train", "val", "test")
    assert captured["split_strategy"] == "lopo"
    assert captured["fold_index"] == 2
    assert captured["val_fold_offset"] == 3


def test_checkpoint_gene_dimension_validation():
    validate_model_gene_dimension({"n_genes": 15000}, 15000)

    with pytest.raises(ValueError, match="checkpoint=15000, data=5000"):
        validate_model_gene_dimension({"n_genes": 15000}, 5000)

    with pytest.raises(ValueError, match="missing required n_genes"):
        validate_model_gene_dimension({}, 5000)


def test_prepare_lopo_output_paths_appends_fold_and_seed():
    args = SimpleNamespace(
        split_strategy="lopo",
        save_dir="/tmp/ng",
        fold_index=2,
        random_seed=7,
        run_name=None,
    )

    prepare_lopo_output_paths(args)

    assert args.save_dir == "/tmp/ng/fold_2"
    assert args.run_name == "fold_2_seed_7"


def test_prepare_lopo_output_paths_preserves_existing_fold_dir():
    args = SimpleNamespace(
        split_strategy="lopo",
        save_dir="/tmp/ng/fold_2",
        fold_index=2,
        random_seed=7,
        run_name="nsclc",
    )

    prepare_lopo_output_paths(args)

    assert args.save_dir == "/tmp/ng/fold_2"
    assert args.run_name == "nsclc_fold_2_seed_7"


def test_split_info_includes_patient_labels():
    dm = object.__new__(FinetuneDataModule)
    dm.train_dataset = SimpleNamespace(
        train_groups=[0, 1],
        val_groups=[2],
        test_groups=[np.int64(3)],
        metadata_cache=SimpleNamespace(
            group_mapping={
                0: {"original_id": "P01"},
                1: {"original_id": "P05"},
                2: {"original_id": "P06"},
                3: {"original_id": "P09"},
            }
        ),
    )

    split_info = dm.get_split_info()

    assert split_info["train_patients"] == ["P01", "P05"]
    assert split_info["val_patients"] == ["P06"]
    assert split_info["test_patients"] == ["P09"]
    assert split_info["test_groups"] == [3]


def test_finetune_collate_keeps_variable_metadata_as_list():
    first = (
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([True, True]),
        {"paired_query_type_counts": {"Neutrophil": 1}},
    )
    second = (
        np.full((2, 3), 2.0, dtype=np.float32),
        np.full((2, 3), 3.0, dtype=np.float32),
        np.array([1, 2], dtype=np.int64),
        np.array([True, True]),
        {"paired_query_type_counts": {"T": 1}},
    )

    batch = finetune_collate_fn([first, second])

    assert batch[0].shape == (2, 2, 3)
    assert batch[1].shape == (2, 2, 3)
    assert batch[2].shape == (2, 2)
    assert batch[3].shape == (2, 2)
    assert batch[4] == [first[4], second[4]]
