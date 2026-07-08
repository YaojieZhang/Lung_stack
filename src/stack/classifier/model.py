"""Model wrapper for end-to-end Stack cell-set classification."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class StackCellSetClassifier(nn.Module):
    """Fine-tune Stack encoder with a linear cell-set response head."""

    def __init__(
        self,
        stack_model: nn.Module,
        *,
        embedding_dim: Optional[int] = None,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.stack_model = stack_model

        if embedding_dim is None:
            embedding_dim = int(stack_model.n_hidden * stack_model.token_dim)

        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features_log = torch.log1p(features)
        tokens = self.stack_model._reduce_and_tokenize(features_log)
        x = self.stack_model._run_attention_layers(tokens)
        cell_embeddings = x.reshape(features.shape[0], features.shape[1], -1)
        cell_set_embedding = cell_embeddings.mean(dim=1)
        logits = self.classifier(cell_set_embedding)
        return {
            "logits": logits,
            "cell_embeddings": cell_embeddings,
            "cell_set_embedding": cell_set_embedding,
        }
