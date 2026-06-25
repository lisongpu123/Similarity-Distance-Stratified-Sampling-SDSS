from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SDSSKGClassifier(nn.Module):
    """MLP classifier used in the SDSS experiments.

    Parameters
    ----------
    in_dim / x_dim:
        Dimension of the base clinical/text feature vector.  ``x_dim`` is
        accepted as a backward-compatible alias because the historical runner
        scripts used this name.
    kg_dim:
        Dimension of the optional KG/DCKG feature vector.
    use_kg:
        Whether to concatenate the KG/DCKG feature vector before prediction.

    The model is intentionally compact so reviewers can inspect the prediction
    backbone without relying on hidden code.
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        kg_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        use_kg: bool = False,
        num_classes: int = 2,
        x_dim: Optional[int] = None,
    ):
        super().__init__()

        if in_dim is None:
            in_dim = x_dim
        if in_dim is None:
            raise ValueError("Either in_dim or x_dim must be provided.")

        self.in_dim = int(in_dim)
        self.kg_dim = int(kg_dim or 0)
        self.use_kg = bool(use_kg and self.kg_dim > 0)

        total_dim = self.in_dim + (self.kg_dim if self.use_kg else 0)
        hidden_dim = int(hidden_dim)
        bottleneck_dim = hidden_dim // 2 if hidden_dim >= 4 else hidden_dim

        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(bottleneck_dim, int(num_classes)),
        )

    def forward(self, x, kg=None):
        if self.use_kg:
            if kg is None:
                raise ValueError("KG features are required when use_kg=True.")
            x = torch.cat([x, kg], dim=1)
        return self.net(x)
