from __future__ import annotations
import torch
import torch.nn as nn


class SDSSKGClassifier(nn.Module):
    """MLP classifier used in the SDSS experiments.

    The model accepts a tabular/text-embedding feature vector and an optional KG
    feature vector.  When ``use_kg`` is False, KG is ignored.  This compact
    implementation is included so reviewers can inspect and rerun the prediction
    backbone without relying on hidden code.
    """

    def __init__(
        self,
        in_dim: int,
        kg_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        use_kg: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.use_kg = bool(use_kg and kg_dim and kg_dim > 0)
        total_dim = int(in_dim) + (int(kg_dim) if self.use_kg else 0)
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, int(num_classes)),
        )

    def forward(self, x, kg=None):
        if self.use_kg and kg is not None:
            x = torch.cat([x, kg], dim=1)
        return self.net(x)
