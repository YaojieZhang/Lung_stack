from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover - optional test dependency
    ad = None


class ClassifierDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        if ad is None:
            self.skipTest("anndata is not installed")

    def test_fixed_test_sample_ids_are_held_out(self) -> None:
        from stack.classifier.dataset import CellSetClassificationDataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            genelist_path = tmp_path / "genes.pkl"
            with genelist_path.open("wb") as handle:
                pickle.dump(["G1", "G2"], handle)

            adata = ad.AnnData(
                X=np.arange(16, dtype=np.float32).reshape(8, 2),
                obs=pd.DataFrame(
                    {
                        "sample_id": [
                            "train_A",
                            "train_A",
                            "XGY_P_P05_P",
                            "XGY_P_P05_P",
                            "BD_immune08",
                            "BD_immune08",
                            "train_B",
                            "train_B",
                        ],
                        "response": [
                            "NMPR",
                            "NMPR",
                            "MPR",
                            "MPR",
                            "NMPR",
                            "NMPR",
                            "MPR",
                            "MPR",
                        ],
                    },
                    index=[f"cell_{idx}" for idx in range(8)],
                ),
                var=pd.DataFrame(index=["G1", "G2"]),
            )

            train_ds = CellSetClassificationDataset(
                adata,
                str(genelist_path),
                split="train",
                sample_size=2,
                filter_organism=False,
                random_state=0,
            )
            test_ds = CellSetClassificationDataset(
                adata,
                str(genelist_path),
                split="test",
                sample_size=2,
                filter_organism=False,
                random_state=0,
            )

            train_groups = [train_ds[idx][2]["sample_id"] for idx in range(len(train_ds))]
            test_groups = [test_ds[idx][2]["sample_id"] for idx in range(len(test_ds))]
            test_labels = [int(test_ds[idx][1].item()) for idx in range(len(test_ds))]

            self.assertEqual(train_groups, ["train_A", "train_B"])
            self.assertEqual(test_groups, ["XGY_P_P05_P", "BD_immune08"])
            self.assertEqual(test_labels, [1, 0])
            self.assertEqual(train_ds.label_to_idx, {"NMPR": 0, "MPR": 1})


if __name__ == "__main__":
    unittest.main()
