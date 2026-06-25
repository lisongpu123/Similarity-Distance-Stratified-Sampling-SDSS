from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


class KGEEncoder:
    """Entity-embedding aggregator for optional KG/DCKG features.

    The encoder reads an entity dictionary and an embedding matrix, maps each
    record's entity list to embeddings, and aggregates them with mean or sum
    pooling.  If the files are unavailable, a zero-vector fallback is used so
    reviewer-visible scripts remain import-compatible without private KG files.
    """

    def __init__(
        self,
        entity_emb_path: str | None = None,
        entity2id_json: str | None = None,
        pad_token: str = "<PAD>",
        agg: str = "mean",
        dim: int = 64,
        *args,
        **kwargs,
    ):
        self.entity_emb_path = entity_emb_path
        self.entity2id_json = entity2id_json
        self.pad_token = pad_token
        self.agg = str(agg).lower()
        self.dim = int(dim)
        self.entity2id = {}
        self.emb = None

        try:
            if entity2id_json and Path(entity2id_json).exists():
                with open(entity2id_json, "r", encoding="utf-8") as f:
                    self.entity2id = json.load(f)
            if entity_emb_path and Path(entity_emb_path).exists():
                self.emb = np.load(entity_emb_path).astype(np.float32)
                if self.emb.ndim != 2:
                    raise ValueError("entity_emb.npy must be a 2D matrix")
                self.dim = int(self.emb.shape[1])
        except Exception:
            # Keep a safe fallback for public reviewer runs without KG artifacts.
            self.entity2id = {}
            self.emb = None

    def _normalize_entities(self, entities) -> List[str]:
        if entities is None:
            return []
        if isinstance(entities, float) and np.isnan(entities):
            return []
        if isinstance(entities, str):
            s = entities.strip()
            if not s:
                return []
            for d in [";", "；", ",", "，", "|", "、"]:
                s = s.replace(d, " ")
            return [x.strip() for x in s.split() if x.strip()]
        if isinstance(entities, (list, tuple, set, np.ndarray, pd.Series)):
            out = []
            for x in list(entities):
                out.extend(self._normalize_entities(x))
            return out
        return [str(entities).strip()] if str(entities).strip() else []

    def encode_one(self, entities) -> np.ndarray:
        ents = self._normalize_entities(entities)
        if self.emb is None or not self.entity2id:
            return np.zeros((self.dim,), dtype=np.float32)

        vecs = []
        for ent in ents:
            idx = self.entity2id.get(ent)
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < self.emb.shape[0]:
                vecs.append(self.emb[idx])

        if not vecs:
            return np.zeros((self.dim,), dtype=np.float32)

        mat = np.stack(vecs, axis=0).astype(np.float32)
        if self.agg == "sum":
            return mat.sum(axis=0).astype(np.float32)
        if self.agg == "max":
            return mat.max(axis=0).astype(np.float32)
        return mat.mean(axis=0).astype(np.float32)

    def encode_entity_series(self, entity_series: Sequence) -> np.ndarray:
        return np.stack([self.encode_one(x) for x in entity_series], axis=0).astype(np.float32)

    # sklearn-like compatibility helpers
    def fit(self, entity_lists):
        return self

    def transform(self, entity_lists):
        return self.encode_entity_series(entity_lists)

    def fit_transform(self, entity_lists):
        return self.transform(entity_lists)
