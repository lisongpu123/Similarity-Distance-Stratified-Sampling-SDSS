# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdsskg.similarity import SimilarityBackend
from sdsskg.sampling import SDSS32Sampler, SDSSConfig

# Toy example only; it does not reproduce manuscript results.
df = pd.read_csv(Path(__file__).with_name("toy_records.csv"))
pos = df[df["label"] == 1].copy()
neg = df[df["label"] == 0].copy()

backend = SimilarityBackend(model_path="__no_local_model__")  # uses TF-IDF fallback
pos_emb = backend.fit_transform(pos["text"].tolist())
center = pos_emb.mean(axis=0)
center = center / ((center ** 2).sum() ** 0.5 + 1e-12)
neg["distance"] = backend.distances_to_center(neg["text"].tolist(), center, metric="cosine")

sampler = SDSS32Sampler(
    config=SDSSConfig(n_bins=4, seed=42, min_bin_size=1),
    seed=42,
    hard_ratio=0.60,
    mid_ratio=0.20,
    easy_ratio=0.20,
)
selected = sampler.sample(pos, neg, k=len(pos), dist_col="distance")
print(selected[["text", "label", "distance"]].sort_values("distance"))
