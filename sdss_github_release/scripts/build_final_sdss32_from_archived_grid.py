# -*- coding: utf-8 -*-
"""
build_final_sdss32_from_archived_grid.py

Purpose
-------
Reconstruct the paper-ready final SDSS table from archived profile-grid results.

This script DOES NOT retrain models and DOES NOT invent scores.
It reads the archived *_sdss32_final_profile_summary_*.csv files produced by the
profile-grid experiments, extracts the pre-fixed final profile for each dataset,
and writes a transparent provenance table for manuscript/GitHub release.

Expected input files in the current project root or a user-specified --root folder:
    dn_sdss32_final_profile_summary_20260519_002234.csv
    df_sdss32_final_profile_summary_20260519_041636.csv
    dpn_sdss32_final_profile_summary_20260519_121738.csv
    dka_sdss32_final_profile_summary_20260519_131725.csv

Optional seed-level input files:
    *_sdss32_final_profile_detail_*.csv

Final profiles used in the current manuscript:
    DN  : S=10, hard/mid/easy = 0.55/0.25/0.20
    DF  : S=10, hard/mid/easy = 0.60/0.20/0.20
    DPN : S=8,  hard/mid/easy = 0.60/0.20/0.20
    DKA : S=8,  hard/mid/easy = 0.60/0.20/0.20

Run in project root:
    python build_final_sdss32_from_archived_grid.py

If archived csv files are in another folder:
    python build_final_sdss32_from_archived_grid.py --root "H:\\iScience新代码\\sdsskg_full_v7_two_external_no_disease"

Outputs:
    final_sdss32_provenance/final_sdss32_table.csv
    final_sdss32_provenance/final_sdss32_test1_f1.csv
    final_sdss32_provenance/final_sdss32_test2_f1.csv
    final_sdss32_provenance/final_sdss32_source_rows.csv
    final_sdss32_provenance/final_sdss32_seed_detail.csv  # if detail files exist
    final_sdss32_provenance/final_sdss32_provenance.xlsx
    final_sdss32_provenance/README_final_sdss32_provenance.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


FINAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "DN": {
        "dataset_key": "dn",
        "strata": 10,
        "hard_ratio": 0.55,
        "mid_ratio": 0.25,
        "easy_ratio": 0.20,
        "selection_basis": "Final SDSS profile fixed from archived grid: 10 strata with moderate hard-aware ratio.",
    },
    "DF": {
        "dataset_key": "df",
        "strata": 10,
        "hard_ratio": 0.60,
        "mid_ratio": 0.20,
        "easy_ratio": 0.20,
        "selection_basis": "Final SDSS profile fixed from archived grid: 10 strata with stronger hard-aware ratio.",
    },
    "DPN": {
        "dataset_key": "dpn",
        "strata": 8,
        "hard_ratio": 0.60,
        "mid_ratio": 0.20,
        "easy_ratio": 0.20,
        "selection_basis": "Final SDSS profile fixed from archived grid: compact 8-strata profile with stronger hard-aware ratio.",
    },
    "DKA": {
        "dataset_key": "dka",
        "strata": 8,
        "hard_ratio": 0.60,
        "mid_ratio": 0.20,
        "easy_ratio": 0.20,
        "selection_basis": "Final SDSS profile fixed from archived grid: compact 8-strata profile with stronger hard-aware ratio.",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Folder containing archived profile summary/detail CSV files.")
    ap.add_argument("--out", type=str, default="final_sdss32_provenance", help="Output folder.")
    ap.add_argument("--strict", action="store_true", help="Fail if duplicate close matches are found.")
    return ap.parse_args()


def find_latest(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No file found under {root} matching pattern: {pattern}")
    return files[0]


def close_float(a: Any, b: float, tol: float = 1e-8) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def profile_id_from_values(strata: int, hard: float, mid: float, easy: float) -> str:
    # The archived files often use strings such as s10_h0p55_m0p25_e0p20.
    return f"s{strata}_h{hard:.2f}_m{mid:.2f}_e{easy:.2f}".replace(".", "p")


def normalize_profile_id(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace(".", "p")
    s = re.sub(r"0p(\d)0(?=\D|$)", r"0p\1", s)
    return s


def match_profile(df: pd.DataFrame, dataset: str, spec: Dict[str, Any], strict: bool = False) -> pd.Series:
    needed_cols = ["strata", "hard_ratio", "mid_ratio", "easy_ratio"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: summary file missing columns: {missing}")

    mask = (
        df["strata"].apply(lambda x: close_float(x, spec["strata"]))
        & df["hard_ratio"].apply(lambda x: close_float(x, spec["hard_ratio"]))
        & df["mid_ratio"].apply(lambda x: close_float(x, spec["mid_ratio"]))
        & df["easy_ratio"].apply(lambda x: close_float(x, spec["easy_ratio"]))
    )

    matched = df.loc[mask].copy()
    if matched.empty and "profile_id" in df.columns:
        target_pid = normalize_profile_id(profile_id_from_values(
            spec["strata"], spec["hard_ratio"], spec["mid_ratio"], spec["easy_ratio"]
        ))
        matched = df.loc[df["profile_id"].apply(normalize_profile_id) == target_pid].copy()

    if matched.empty:
        raise ValueError(
            f"{dataset}: final profile not found in summary. "
            f"Expected S={spec['strata']}, h/m/e={spec['hard_ratio']}/{spec['mid_ratio']}/{spec['easy_ratio']}"
        )

    if len(matched) > 1:
        if strict:
            raise ValueError(f"{dataset}: multiple rows matched the final profile.")
        matched = matched.iloc[[0]].copy()

    return matched.iloc[0]


def get_col(row: pd.Series, *names: str, default=np.nan) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return default


def text_mean_std(mean: Any, std: Any) -> str:
    return f"{float(mean):.4f} ± {float(std):.4f}"



def display_path(path: Path) -> str:
    """Return a repository-relative path when possible for portable provenance tables."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def collect_final_rows(root: Path, strict: bool = False) -> pd.DataFrame:
    rows = []

    for dataset, spec in FINAL_PROFILES.items():
        key = spec["dataset_key"]
        summary_file = find_latest(root, f"{key}_sdss32_final_profile_summary_*.csv")
        summary = pd.read_csv(summary_file)
        row = match_profile(summary, dataset, spec, strict=strict)

        test1_f1_mean = get_col(row, "test1_f1_report_only_mean", "test1_f1_mean")
        test1_f1_std = get_col(row, "test1_f1_report_only_std", "test1_f1_std")
        test2_f1_mean = get_col(row, "test2_f1_report_only_mean", "test2_f1_mean")
        test2_f1_std = get_col(row, "test2_f1_report_only_std", "test2_f1_std")

        test1_auc_mean = get_col(row, "test1_auc_report_only_mean", "test1_auc_mean")
        test1_auc_std = get_col(row, "test1_auc_report_only_std", "test1_auc_std")
        test2_auc_mean = get_col(row, "test2_auc_report_only_mean", "test2_auc_mean")
        test2_auc_std = get_col(row, "test2_auc_report_only_std", "test2_auc_std")

        profile_id = get_col(row, "profile_id", default=profile_id_from_values(
            spec["strata"], spec["hard_ratio"], spec["mid_ratio"], spec["easy_ratio"]
        ))

        rows.append({
            "dataset": dataset,
            "method": "SDSS",
            "profile_id": profile_id,
            "strata": int(spec["strata"]),
            "hard_ratio": float(spec["hard_ratio"]),
            "mid_ratio": float(spec["mid_ratio"]),
            "easy_ratio": float(spec["easy_ratio"]),
            "test1_f1_mean": float(test1_f1_mean),
            "test1_f1_std": float(test1_f1_std),
            "test1_f1_text": text_mean_std(test1_f1_mean, test1_f1_std),
            "test2_f1_mean": float(test2_f1_mean),
            "test2_f1_std": float(test2_f1_std),
            "test2_f1_text": text_mean_std(test2_f1_mean, test2_f1_std),
            "test1_auc_mean": float(test1_auc_mean) if pd.notna(test1_auc_mean) else np.nan,
            "test1_auc_std": float(test1_auc_std) if pd.notna(test1_auc_std) else np.nan,
            "test2_auc_mean": float(test2_auc_mean) if pd.notna(test2_auc_mean) else np.nan,
            "test2_auc_std": float(test2_auc_std) if pd.notna(test2_auc_std) else np.nan,
            "selection_basis": spec["selection_basis"],
            "source_summary_file": display_path(summary_file),
        })

    final = pd.DataFrame(rows)
    order = {"DN": 0, "DF": 1, "DPN": 2, "DKA": 3}
    final["_order"] = final["dataset"].map(order)
    final = final.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return final


def collect_seed_detail(root: Path, final: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, final_row in final.iterrows():
        dataset = final_row["dataset"]
        key = dataset.lower()
        detail_files = sorted(root.glob(f"{key}_sdss32_final_profile_detail_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Optional fallback for older internal archives; public releases may omit selected-detail files.
        if not detail_files:
            detail_files = sorted(root.glob(f"{key}_sdss32_final_selected_detail_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not detail_files:
            continue

        detail_file = detail_files[0]
        detail = pd.read_csv(detail_file)

        mask = (
            detail["strata"].apply(lambda x: close_float(x, final_row["strata"]))
            & detail["hard_ratio"].apply(lambda x: close_float(x, final_row["hard_ratio"]))
            & detail["mid_ratio"].apply(lambda x: close_float(x, final_row["mid_ratio"]))
            & detail["easy_ratio"].apply(lambda x: close_float(x, final_row["easy_ratio"]))
        )
        sub = detail.loc[mask].copy()
        if sub.empty and "profile_id" in detail.columns:
            target_pid = normalize_profile_id(final_row["profile_id"])
            sub = detail.loc[detail["profile_id"].apply(normalize_profile_id) == target_pid].copy()

        if sub.empty:
            continue

        sub.insert(0, "dataset_label", dataset)
        sub.insert(1, "source_detail_file", display_path(detail_file))
        rows.append(sub)

    if rows:
        return pd.concat(rows, axis=0, ignore_index=True)
    return pd.DataFrame()


def write_outputs(final: pd.DataFrame, seed_detail: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    test1 = final[["dataset", "test1_f1_text"]].rename(columns={"test1_f1_text": "SDSS"})
    test2 = final[["dataset", "test2_f1_text"]].rename(columns={"test2_f1_text": "SDSS"})

    final.to_csv(out_dir / "final_sdss32_table.csv", index=False, encoding="utf-8-sig")
    test1.to_csv(out_dir / "final_sdss32_test1_f1.csv", index=False, encoding="utf-8-sig")
    test2.to_csv(out_dir / "final_sdss32_test2_f1.csv", index=False, encoding="utf-8-sig")
    final[["dataset", "profile_id", "strata", "hard_ratio", "mid_ratio", "easy_ratio", "source_summary_file", "selection_basis"]].to_csv(
        out_dir / "final_sdss32_source_rows.csv", index=False, encoding="utf-8-sig"
    )

    if not seed_detail.empty:
        seed_detail.to_csv(out_dir / "final_sdss32_seed_detail.csv", index=False, encoding="utf-8-sig")

    xlsx_path = out_dir / "final_sdss32_provenance.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        final.to_excel(writer, index=False, sheet_name="final_table")
        test1.to_excel(writer, index=False, sheet_name="Test1_F1")
        test2.to_excel(writer, index=False, sheet_name="Test2_F1")
        if not seed_detail.empty:
            seed_detail.to_excel(writer, index=False, sheet_name="seed_detail")

    readme = f"""# Final SDSS result provenance

This folder was generated by `build_final_sdss32_from_archived_grid.py`.

The reported final SDSS results are reconstructed from archived profile-grid
summary CSV files. The script does not retrain models and does not manually edit
F1 values. It extracts the pre-fixed final profile for each dataset:

| Dataset | Strata | hard/mid/easy | Test1 F1 | Test2 F1 |
|---|---:|---:|---:|---:|
"""
    for _, r in final.iterrows():
        readme += (
            f"| {r['dataset']} | {int(r['strata'])} | "
            f"{r['hard_ratio']:.2f}/{r['mid_ratio']:.2f}/{r['easy_ratio']:.2f} | "
            f"{r['test1_f1_text']} | {r['test2_f1_text']} |\n"
        )
    readme += """

Recommended GitHub release contents:

- `build_final_sdss32_from_archived_grid.py`
- the archived `*_sdss32_final_profile_summary_*.csv` files
- the archived `*_sdss32_final_profile_detail_*.csv` files, if available
- this generated `final_sdss32_provenance.xlsx`

Important note: this is a result-provenance reconstruction script. It is intended
to reproduce the manuscript tables from the archived grid experiments. It should
not be presented as a fresh re-training script. Seed-level detail is included
when the corresponding archived detail files are available; the summary CSV files
are the authoritative source for the manuscript mean ± SD table.
"""
    (out_dir / "README_final_sdss32_provenance.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()

    final = collect_final_rows(root, strict=args.strict)
    seed_detail = collect_seed_detail(root, final)
    write_outputs(final, seed_detail, out_dir)

    print("\n========== Final SDSS table from archived profile grid ==========")
    print(final[["dataset", "strata", "hard_ratio", "mid_ratio", "easy_ratio", "test1_f1_text", "test2_f1_text"]].to_string(index=False))
    print(f"\nSaved to: {out_dir}")
    print(f"Excel: {out_dir / 'final_sdss32_provenance.xlsx'}")


if __name__ == "__main__":
    main()
