"""Model wrapper for end-to-end Stack cell-set classification."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


HEAD_TYPES = (
    "linear",
    "mlp",
    "layernorm_mlp",
    "layernorm_linear",
    "dropout_linear",
    "attention_pooling",
)


class AttentionPoolingHead(nn.Module):
    """Attention pooling over cells followed by sample-level classification."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        middle_dim = max(1, hidden_dim // 2)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, middle_dim),
            nn.Tanh(),
            nn.Linear(middle_dim, 1),
        )
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, cell_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_score = self.attn(cell_embeddings)
        attn_weight = attn_score.softmax(dim=1)
        sample_embedding = (attn_weight * cell_embeddings).sum(dim=1)
        logits = self.cls(sample_embedding)
        return logits, attn_weight


def build_classifier_head(
    head_type: str,
    hidden_dim: int,
    num_classes: int,
    *,
    middle_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> nn.Module:
    if head_type not in HEAD_TYPES:
        raise ValueError(f"Unknown classifier head_type '{head_type}'. Expected one of {HEAD_TYPES}")

    if middle_dim is None:
        middle_dim = max(1, hidden_dim // 2)

    if head_type == "linear":
        return nn.Linear(hidden_dim, num_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_dim, middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, num_classes),
        )
    if head_type == "layernorm_mlp":
        return nn.Sequential(
            nn.Linear(hidden_dim, middle_dim),
            nn.LayerNorm(middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, num_classes),
        )
    if head_type == "layernorm_linear":
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
    if head_type == "dropout_linear":
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    return AttentionPoolingHead(hidden_dim, num_classes)


class StackCellSetClassifier(nn.Module):
    """Stack encoder with a trainable cell-set response classification head."""

    def __init__(
        self,
        stack_model: nn.Module,
        *,
        embedding_dim: Optional[int] = None,
        n_classes: int = 3,
        head_type: str = "linear",
        middle_dim: Optional[int] = None,
        dropout: float = 0.0,
        freeze_stack: bool = True,
    ) -> None:
        super().__init__()
        self.stack_model = stack_model
        self.freeze_stack = freeze_stack
        self.head_type = head_type

        if embedding_dim is None:
            embedding_dim = int(stack_model.n_hidden * stack_model.token_dim)

        self.embedding_dim = embedding_dim
        self.classifier = build_classifier_head(
            head_type,
            embedding_dim,
            n_classes,
            middle_dim=middle_dim,
            dropout=dropout,
        )

        if self.freeze_stack:
            for parameter in self.stack_model.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_stack:
            self.stack_model.eval()
        return self

    def _encode_cells(self, features: torch.Tensor) -> torch.Tensor:
        features_log = torch.log1p(features)
        tokens = self.stack_model._reduce_and_tokenize(features_log)
        x = self.stack_model._run_attention_layers(tokens)
        return x.reshape(features.shape[0], features.shape[1], -1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.freeze_stack:
            with torch.no_grad():
                cell_embeddings = self._encode_cells(features)
        else:
            cell_embeddings = self._encode_cells(features)

        attention_weights = None
        if self.head_type == "attention_pooling":
            logits, attention_weights = self.classifier(cell_embeddings)
            cell_set_embedding = (attention_weights * cell_embeddings).sum(dim=1)
        else:
            cell_set_embedding = cell_embeddings.mean(dim=1)
            logits = self.classifier(cell_set_embedding)

        output = {
            "logits": logits,
            "cell_embeddings": cell_embeddings,
            "cell_set_embedding": cell_set_embedding,
        }
        if attention_weights is not None:
            output["attention_weights"] = attention_weights
        return output
