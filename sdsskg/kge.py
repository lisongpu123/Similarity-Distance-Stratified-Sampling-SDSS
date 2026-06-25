from __future__ import annotations
import numpy as np


class KGEEncoder:
    """Lightweight fallback KG encoder.

    The published experiments set ``use_kg=False`` for the SDSS final tables.
    This class is provided only to keep the original runner import-compatible.
    """

    def __init__(self, dim: int = 64, *args, **kwargs):
        self.dim = int(dim)

    def fit(self, entity_lists):
        return self

    def transform(self, entity_lists):
        return np.zeros((len(entity_lists), self.dim), dtype=np.float32)

    def fit_transform(self, entity_lists):
        self.fit(entity_lists)
        return self.transform(entity_lists)
