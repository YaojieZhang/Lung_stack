from __future__ import annotations

import unittest


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional test dependency
    torch = None


class ClassifierModelTest(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")

    def test_wrapper_mean_pools_cell_embeddings_before_linear_head(self) -> None:
        from stack.classifier.model import StackCellSetClassifier

        class FakeStack(torch.nn.Module):
            n_hidden = 2
            token_dim = 3

            def _reduce_and_tokenize(self, features):
                return features.reshape(features.shape[0], features.shape[1], 2, 3)

            def _run_attention_layers(self, tokens):
                return tokens

        model = StackCellSetClassifier(FakeStack(), n_classes=2)
        features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

        output = model(features)

        expected_cell_embeddings = torch.log1p(features)
        expected_pooled = expected_cell_embeddings.mean(dim=1)
        self.assertEqual(tuple(output["logits"].shape), (1, 2))
        self.assertTrue(torch.allclose(output["cell_set_embedding"], expected_pooled))


if __name__ == "__main__":
    unittest.main()
