# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

FINAL_PROFILES = {
    "DN":  {"strata": 10, "hard_ratio": 0.55, "mid_ratio": 0.25, "easy_ratio": 0.20},
    "DF":  {"strata": 10, "hard_ratio": 0.60, "mid_ratio": 0.20, "easy_ratio": 0.20},
    "DPN": {"strata": 8,  "hard_ratio": 0.60, "mid_ratio": 0.20, "easy_ratio": 0.20},
    "DKA": {"strata": 8,  "hard_ratio": 0.60, "mid_ratio": 0.20, "easy_ratio": 0.20},
}
ORDER = {"DN": 0, "DF": 1, "DPN": 2, "DKA": 3}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="outputs directory")
    ap.add_argument(
        "--prefer_latest_per_seed",
        action="store_true",
        help="If multiple matched runs exist for the same dataset/seed, keep the latest one."
    )
    return ap.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_dataset(run_dir: Path, cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    parts = [str(run_dir).lower()]
    if cfg:
        parts.append(str(cfg.get("run", {}).get("run_name", "")).lower())
        data = cfg.get("data", {})
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, str):
                    parts.append(v.lower())
    text = " ".join(parts)
    if re.search(r"(^|[_\\/.-])dpn([_\\/.-]|$)", text):
        return "DPN"
    if re.search(r"(^|[_\\/.-])dka([_\\/.-]|$)", text):
        return "DKA"
    if re.search(r"(^|[_\\/.-])df([_\\/.-]|$)", text):
        return "DF"
    if re.search(r"(^|[_\\/.-])dn([_\\/.-]|$)", text):
        return "DN"
    return None


def infer_seed(run_dir: Path, cfg: Optional[Dict[str, Any]]) -> Optional[int]:
    if cfg:
        try:
            return int(cfg.get("train", {}).get("seed"))
        except Exception:
            pass
    m = re.search(r"seed(\d+)", str(run_dir).lower())
    return int(m.group(1)) if m else None


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def close(a, b, eps=1e-6):
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return False


def profile_matches(dataset: str, cfg: Optional[Dict[str, Any]]) -> bool:
    if not cfg:
        return False
    sampler = cfg.get("sampler", {})
    target = FINAL_PROFILES[dataset]
    return (
        str(sampler.get("name", "")).lower() == "sdss32"
        and int(sampler.get("strata", -999)) == int(target["strata"])
        and close(sampler.get("hard_ratio"), target["hard_ratio"])
        and close(sampler.get("mid_ratio"), target["mid_ratio"])
        and close(sampler.get("easy_ratio"), target["easy_ratio"])
    )


def collect(outputs: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for test1_path in outputs.rglob("metrics_test1.json"):
        run_dir = test1_path.parent
        test2_path = run_dir / "metrics_test2.json"
        val_path = run_dir / "metrics_val.json"
        cfg_path = run_dir / "config_used.json"
        if not test2_path.exists() or not cfg_path.exists():
            continue

        cfg = load_json(cfg_path)
        dataset = infer_dataset(run_dir, cfg)
        if dataset not in FINAL_PROFILES:
            continue
        if not profile_matches(dataset, cfg):
            continue

        seed = infer_seed(run_dir, cfg)
        if seed is None:
            continue

        test1 = load_json(test1_path) or {}
        test2 = load_json(test2_path) or {}
        val = load_json(val_path) if val_path.exists() else {}
        sampler = cfg.get("sampler", {}) if cfg else {}
        run_name = cfg.get("run", {}).get("run_name", "") if cfg else ""

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "mtime": run_dir.stat().st_mtime,
            "sampler_name": sampler.get("name"),
            "strata": sampler.get("strata"),
            "hard_ratio": sampler.get("hard_ratio"),
            "mid_ratio": sampler.get("mid_ratio"),
            "easy_ratio": sampler.get("easy_ratio"),
            "val_f1": val.get("f1") if isinstance(val, dict) else None,
            "val_auc": val.get("auc") if isinstance(val, dict) else None,
            "val_pr_auc": val.get("pr_auc") if isinstance(val, dict) else None,
            "test1_f1": test1.get("f1"),
            "test1_auc": test1.get("auc"),
            "test1_pr_auc": test1.get("pr_auc"),
            "test1_precision": test1.get("precision"),
            "test1_recall": test1.get("recall"),
            "test1_specificity": test1.get("specificity"),
            "test2_f1": test2.get("f1"),
            "test2_auc": test2.get("auc"),
            "test2_pr_auc": test2.get("pr_auc"),
            "test2_precision": test2.get("precision"),
            "test2_recall": test2.get("recall"),
            "test2_specificity": test2.get("specificity"),
        })
    return pd.DataFrame(rows)


def dedupe_latest_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["dataset", "seed", "mtime"])
    return df.groupby(["dataset", "seed"], as_index=False).tail(1).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in df.groupby("dataset"):
        target = FINAL_PROFILES[dataset]
        row = {
            "dataset": dataset,
            "strata": target["strata"],
            "hard_ratio": target["hard_ratio"],
            "mid_ratio": target["mid_ratio"],
            "easy_ratio": target["easy_ratio"],
            "n": len(g),
            "seeds": ",".join(str(int(x)) for x in sorted(g["seed"].unique())),
        }
        metrics = [
            "val_f1", "val_auc", "val_pr_auc",
            "test1_f1", "test1_auc", "test1_pr_auc",
            "test2_f1", "test2_auc", "test2_pr_auc",
        ]
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        row["test1_f1_text"] = f"{row['test1_f1_mean']:.4f} ± {row['test1_f1_std']:.4f}"
        row["test2_f1_text"] = f"{row['test2_f1_mean']:.4f} ± {row['test2_f1_std']:.4f}"
        row["test1_auc_text"] = f"{row['test1_auc_mean']:.4f} ± {row['test1_auc_std']:.4f}"
        row["test2_auc_text"] = f"{row['test2_auc_mean']:.4f} ± {row['test2_auc_std']:.4f}"
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["dataset"].map(ORDER)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def main():
    args = parse_args()
    outputs = Path(args.outputs)
    if not outputs.exists():
        raise FileNotFoundError(f"outputs not found: {outputs}")

    detail_all = collect(outputs)
    if detail_all.empty:
        print("[ERROR] 没有找到匹配最终 profile 的结果。")
        print("请确认输出目录中有 config_used.json、metrics_test1.json、metrics_test2.json。")
        return

    detail_all["_order"] = detail_all["dataset"].map(ORDER)
    detail_all = detail_all.sort_values(["_order", "seed", "mtime"]).drop(columns="_order").reset_index(drop=True)

    detail = dedupe_latest_per_seed(detail_all) if args.prefer_latest_per_seed else detail_all
    detail["_order"] = detail["dataset"].map(ORDER)
    detail = detail.sort_values(["_order", "seed", "mtime"]).drop(columns="_order").reset_index(drop=True)

    summary = summarize(detail)

    detail_all.to_csv("final_profile_rerun_all_matches.csv", index=False, encoding="utf-8-sig")
    detail.to_csv("final_profile_rerun_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("final_profile_rerun_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter("final_profile_rerun_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        detail.to_excel(writer, index=False, sheet_name="detail_used")
        detail_all.to_excel(writer, index=False, sheet_name="all_matches")

    print("\n========== Final-profile rerun summary ==========")
    for _, r in summary.iterrows():
        print(f"{r['dataset']}: strata={r['strata']}, hard/mid/easy={r['hard_ratio']}/{r['mid_ratio']}/{r['easy_ratio']}, n={int(r['n'])}, seeds={r['seeds']}")
        print(f"  Test1 F1 = {r['test1_f1_text']}")
        print(f"  Test2 F1 = {r['test2_f1_text']}")
        print(f"  Test1 AUC = {r['test1_auc_text']}")
        print(f"  Test2 AUC = {r['test2_auc_text']}")

    print("\n用于论文表格的 F1：")
    print(summary[["dataset", "test1_f1_text", "test2_f1_text"]].to_string(index=False))

    print("\nSaved:")
    print("  final_profile_rerun_all_matches.csv")
    print("  final_profile_rerun_detail.csv")
    print("  final_profile_rerun_summary.csv")
    print("  final_profile_rerun_summary.xlsx")


if __name__ == "__main__":
    main()
