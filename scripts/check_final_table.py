# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

expected = {
    "DN": ("0.8491 ± 0.0341", "0.8607 ± 0.0136"),
    "DF": ("0.9815 ± 0.0124", "0.9723 ± 0.0199"),
    "DPN": ("0.9918 ± 0.0075", "0.9878 ± 0.0088"),
    "DKA": ("0.9360 ± 0.0171", "0.9191 ± 0.0120"),
}
path = Path("results/final_sdss32_provenance/final_sdss32_table.csv")
if not path.exists():
    raise SystemExit(f"Missing {path}; run scripts/build_final_sdss32_from_archived_grid.py first.")
df = pd.read_csv(path)
for _, r in df.iterrows():
    ds = r["dataset"]
    got = (r["test1_f1_text"], r["test2_f1_text"])
    if expected[ds] != got:
        raise SystemExit(f"Mismatch for {ds}: got {got}, expected {expected[ds]}")
print("Final SDSS table check passed.")
