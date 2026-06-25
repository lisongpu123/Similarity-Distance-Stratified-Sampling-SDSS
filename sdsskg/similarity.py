# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np

if not hasattr(np, "object"):
    np.object = object  # type: ignore
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer


def _clean_path(s: str) -> str:
    return str(s).strip().strip('"').strip("'")


def _resolve_local_model_path(model_path: str) -> str:
    p = _clean_path(model_path)
    p_abs = os.path.abspath(os.path.expanduser(p))
    if os.path.isdir(p_abs):
        return p_abs
    p_rel = os.path.normpath(p)
    if os.path.isdir(p_rel):
        return os.path.abspath(p_rel)
    return p


class HFTransformersEncoder:
    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32, max_length: int = 256):
        self.model_path = _resolve_local_model_path(model_path)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        local_only = os.path.isdir(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=local_only, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=local_only)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state
            attn = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().cpu().numpy())

        if len(embs) == 0:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack(embs).astype(np.float32)


class SimilarityBackend:
    """
    Two modes:
    1) HF local model available -> fixed-dim dense embeddings
    2) fallback TF-IDF -> must use fit_transform(train) and transform(test)
    """
    def __init__(
        self,
        model_path: str = "multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self.encoder: Optional[HFTransformersEncoder] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.mode = "tfidf"

        try:
            resolved = _resolve_local_model_path(model_path)
            if os.path.isdir(resolved):
                self.encoder = HFTransformersEncoder(
                    model_path=resolved,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                self.mode = "hf"
        except Exception:
            self.encoder = None
            self.mode = "tfidf"

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        texts = ["" if t is None else str(t) for t in texts]

        if self.mode == "hf" and self.encoder is not None:
            return self.encoder.encode(texts)

        self.vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            max_features=4096,
        )
        x = self.vectorizer.fit_transform(texts)
        arr = x.toarray().astype(np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def transform(self, texts: List[str]) -> np.ndarray:
        texts = ["" if t is None else str(t) for t in texts]

        if self.mode == "hf" and self.encoder is not None:
            return self.encoder.encode(texts)

        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer has not been fitted. Call fit_transform(train_texts) first.")
        x = self.vectorizer.transform(texts)
        arr = x.toarray().astype(np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Backward-compatible shortcut:
        - if HF mode: direct encode
        - if TF-IDF mode and vectorizer not fitted: fit_transform
        - if TF-IDF mode and vectorizer fitted: transform
        """
        if self.mode == "hf":
            return self.transform(texts)
        if self.vectorizer is None:
            return self.fit_transform(texts)
        return self.transform(texts)

    def fit_transform_center(self, texts: List[str]) -> np.ndarray:
        x = self.fit_transform(texts)
        if x.shape[0] == 0:
            return np.zeros((1,), dtype=np.float32)
        c = x.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return c.astype(np.float32)

    def distances_to_center(self, texts: List[str], center: np.ndarray, metric: str = "cosine") -> np.ndarray:
        x = self.transform(texts)
        c = center.astype(np.float32).reshape(1, -1)

        metric = (metric or "cosine").lower()
        if metric == "euclidean":
            return np.sqrt(((x - c) ** 2).sum(axis=1)).astype(np.float32)
        if metric == "manhattan":
            return np.abs(x - c).sum(axis=1).astype(np.float32)

        sims = (x @ c.T).reshape(-1)
        return (1.0 - sims).astype(np.float32)