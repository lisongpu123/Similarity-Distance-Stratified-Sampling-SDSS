# -*- coding: utf-8 -*-
"""Reselect/reconstruct final SDSS3.2 profiles from archived profile-grid outputs.

This module is intentionally conservative: it does not retrain models and does
not modify scores.  It reads archived ``*_profile_summary_*.csv`` files,
extracts the pre-fixed final task-specific profiles, and writes manuscript-ready
mean±SD tables.  It supports recursive folders such as
``outputs/sdss32_final_profile_selection/dn/20260519_002234``.

Usage:
    python -m sdsskg.reselect_sdss32_final_profile_v3 --root results/archived_profile_grid
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

FINAL = {
    "dn":  {"dataset":"DN",  "strata":10, "hard_ratio":0.55, "mid_ratio":0.25, "easy_ratio":0.20},
    "df":  {"dataset":"DF",  "strata":10, "hard_ratio":0.60, "mid_ratio":0.20, "easy_ratio":0.20},
    "dpn": {"dataset":"DPN", "strata":8,  "hard_ratio":0.60, "mid_ratio":0.20, "easy_ratio":0.20},
    "dka": {"dataset":"DKA", "strata":8,  "hard_ratio":0.60, "mid_ratio":0.20, "easy_ratio":0.20},
}
ORDER = {"DN":0,"DF":1,"DPN":2,"DKA":3}


def pick_file(root: Path, key: str, kind: str) -> Path:
    files = list(root.rglob(f"{key}_sdss32_final_profile_{kind}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {kind} file found for {key} under {root}")
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def close(a,b,eps=1e-9):
    return abs(float(a)-float(b)) <= eps


def extract_row(summary: pd.DataFrame, target: dict) -> pd.Series:
    mask = (
        (summary["strata"].astype(int) == int(target["strata"])) &
        (summary["hard_ratio"].astype(float).map(lambda x: close(x, target["hard_ratio"]))) &
        (summary["mid_ratio"].astype(float).map(lambda x: close(x, target["mid_ratio"]))) &
        (summary["easy_ratio"].astype(float).map(lambda x: close(x, target["easy_ratio"])))
    )
    hit = summary.loc[mask]
    if hit.empty:
        raise ValueError(f"Target profile not found: {target}")
    return hit.iloc[0]


def fmt(row, col_mean, col_std):
    return f"{float(row[col_mean]):.4f} ± {float(row[col_std]):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/archived_profile_grid")
    ap.add_argument("--out", default="results/final_sdss32_provenance")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows=[]
    for key,target in FINAL.items():
        sf = pick_file(root, key, "summary")
        summary = pd.read_csv(sf)
        r = extract_row(summary, target)
        rows.append({
            "dataset": target["dataset"],
            "strata": target["strata"],
            "hard_ratio": target["hard_ratio"],
            "mid_ratio": target["mid_ratio"],
            "easy_ratio": target["easy_ratio"],
            "test1_f1_text": fmt(r,"test1_f1_report_only_mean","test1_f1_report_only_std"),
            "test2_f1_text": fmt(r,"test2_f1_report_only_mean","test2_f1_report_only_std"),
            "source_summary_file": str(sf),
        })
    final = pd.DataFrame(rows)
    final["_order"] = final["dataset"].map(ORDER)
    final = final.sort_values("_order").drop(columns="_order")
    final.to_csv(out/"final_sdss32_table.csv", index=False, encoding="utf-8-sig")
    final[["dataset","test1_f1_text"]].to_csv(out/"final_sdss32_test1_f1.csv", index=False, encoding="utf-8-sig")
    final[["dataset","test2_f1_text"]].to_csv(out/"final_sdss32_test2_f1.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out/"final_sdss32_provenance.xlsx", engine="openpyxl") as w:
        final.to_excel(w, index=False, sheet_name="final_table")
    print("\n======== Final SDSS table from archived profile grid ========")
    print(final[["dataset","strata","hard_ratio","mid_ratio","easy_ratio","test1_f1_text","test2_f1_text"]].to_string(index=False))
    print(f"\nSaved to: {out}")

if __name__ == "__main__":
    main()
