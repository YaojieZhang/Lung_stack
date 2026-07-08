from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest


ad = pytest.importorskip("anndata")


def test_grouped_sampler_never_crosses_sample_id(tmp_path):
    from stack.sample_id_sampling import GroupedTestSamplerDataset

    genelist_path = tmp_path / "genes.pkl"
    with genelist_path.open("wb") as handle:
        pickle.dump(["G1", "G2"], handle)

    adata = ad.AnnData(
        X=np.arange(10, dtype=np.float32).reshape(5, 2),
        obs=pd.DataFrame(
            {"sample_id": ["pre", "pre", "post", "post", "post"]},
            index=[f"cell_{idx}" for idx in range(5)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    dataset = GroupedTestSamplerDataset(
        adata,
        str(genelist_path),
        sample_size=2,
        groupby_col="sample_id",
        filter_organism=False,
        random_state=0,
    )

    sample_groups = []
    unique_counts = []
    for idx in range(len(dataset)):
        features, metadata = dataset[idx]
        sample_groups.append(metadata["group_value"])
        unique_counts.append(metadata["n_unique_cells_loaded"])
        assert features.shape == (2, 2)

    assert sample_groups == ["pre", "post", "post"]
    assert unique_counts == [2, 2, 1]
    assert dataset.group_obs_indices["pre"].tolist() == [0, 1]
    assert dataset.group_obs_indices["post"].tolist() == [2, 3, 4]


def test_group_output_path_uses_patient_folder(tmp_path):
    from stack.sample_id_sampling import resolve_group_output_path

    obs = pd.DataFrame({"Patient": ["P01", "P01"]})

    output_path = resolve_group_output_path(
        tmp_path / "NG.h5ad",
        "BD_P_P01_N",
        ".h5ad",
        group_obs=obs,
        patient_col="Patient",
    )

    assert output_path == tmp_path / "NG" / "P01" / "BD_P_P01_N.h5ad"
