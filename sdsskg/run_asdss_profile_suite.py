# -*- coding: utf-8 -*-
"""
A-SDSS: Adaptive Similarity-Distance Stratified Sampling profile selection.

核心逻辑：
1. 不为每个数据集手工指定 hard_ratio；
2. 预设一组从 balanced 到 hard-aware 的候选 difficulty profiles；
3. 只使用训练集内部验证集选择 profile；
4. 外部 Test1/Test2 只用于最终一次性评估和报告；
5. 与 run_experiment_clinical_asdss.py 配合时，负样本排序为：临床类别优先 + 相似度距离辅助。

运行示例：
python -m sdsskg.run_asdss_profile_suite --base_config ./sdsskg/df_two_external_wokg.yaml --mode all
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import yaml


PROFILES = {
    "P1_balanced": {
        "name": "sdss32",
        "strata": 4,
        "neg_per_pos": 1.0,
        "hard_ratio": 0.40,
        "mid_ratio": 0.40,
        "easy_ratio": 0.20,
    },
    "P2_moderate_hard": {
        "name": "sdss32",
        "strata": 4,
        "neg_per_pos": 1.0,
        "hard_ratio": 0.50,
        "mid_ratio": 0.35,
        "easy_ratio": 0.15,
    },
    "P3_hard_aware": {
        "name": "sdss32",
        "strata": 4,
        "neg_per_pos": 1.0,
        "hard_ratio": 0.60,
        "mid_ratio": 0.30,
        "easy_ratio": 0.10,
    },
    "P4_hard_intensive": {
        "name": "sdss32",
        "strata": 4,
        "neg_per_pos": 1.0,
        "hard_ratio": 0.65,
        "mid_ratio": 0.25,
        "easy_ratio": 0.10,
    },
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True)
    ap.add_argument("--mode", type=str, default="all", choices=["all", "select_only"])
    ap.add_argument("--seeds", type=str, default="42,52,62,72,82")
    ap.add_argument("--penalty", type=float, default=0.5)
    ap.add_argument("--metric", type=str, default="f1", choices=["f1", "auc", "pr_auc"])
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--out_root", type=str, default="outputs/asdss_profile_selection")
    return ap.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def save_yaml(cfg: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_subdir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Output run folder not found: {path}")
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No timestamp subfolder found in: {path}")
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def infer_dataset_name(base_config: Path) -> str:
    return base_config.stem.split("_")[0].lower()


def make_run_name(dataset: str, profile_name: str, seed: int, tag: str) -> str:
    return f"{dataset}_asdss_{profile_name.replace('.', '_')}_seed{seed}_{tag}"


def run_one_config(py: str, cfg_path: Path):
    cmd = [py, "-m", "sdsskg.run_experiment", "--config", str(cfg_path)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_metrics(output_dir: Path, run_name: str) -> Dict[str, Any]:
    run_root = output_dir / run_name
    ts_dir = latest_subdir(run_root)
    return {
        "saved_dir": str(ts_dir),
        "val": read_json(ts_dir / "metrics_val.json"),
        "test1": read_json(ts_dir / "metrics_test1.json"),
        "test2": read_json(ts_dir / "metrics_test2.json"),
    }


def mean_std(vals: List[float]):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    s = pd.Series(vals, dtype=float)
    return float(s.mean()), float(s.std(ddof=1))


def main():
    args = parse_args()
    base_config = Path(args.base_config)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    metric = args.metric
    penalty = float(args.penalty)

    base_cfg = load_yaml(base_config)
    dataset = infer_dataset_name(base_config)
    tag = time.strftime("%Y%m%d_%H%M%S")

    out_root = Path(args.out_root) / dataset / tag
    tmp_cfg_dir = out_root / "tmp_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    output_dir = Path(base_cfg.get("run", {}).get("output_dir", "outputs"))
    details = []

    for profile_name, sampler_cfg in PROFILES.items():
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["sampler"] = copy.deepcopy(sampler_cfg)

            # If the yaml already specified the clinical category column, keep it.
            # Otherwise run_experiment will auto-detect it.
            old_sampler = base_cfg.get("sampler", {}) or {}
            for key in ["clinical_category_col", "negative_category_col", "sample_category_col"]:
                if key in old_sampler:
                    cfg["sampler"][key] = old_sampler[key]

            cfg.setdefault("train", {})
            cfg["train"]["seed"] = int(seed)

            # No-KG experiment. For KG-version experiments, change this to True and ensure KG files exist.
            cfg.setdefault("model", {})
            cfg["model"]["use_kg"] = False
            cfg["model"].setdefault("kg_dim_fallback", 64)

            cfg.setdefault("run", {})
            run_name = make_run_name(dataset, profile_name, seed, tag)
            cfg["run"]["run_name"] = run_name

            tmp_cfg_path = tmp_cfg_dir / f"{run_name}.yaml"
            save_yaml(cfg, tmp_cfg_path)
            run_one_config(args.python, tmp_cfg_path)

            ms = collect_metrics(output_dir, run_name)
            row = {
                "dataset": dataset,
                "profile": profile_name,
                "seed": seed,
                "hard_ratio": sampler_cfg["hard_ratio"],
                "mid_ratio": sampler_cfg["mid_ratio"],
                "easy_ratio": sampler_cfg["easy_ratio"],
                "strata": sampler_cfg.get("strata", ""),
                "neg_per_pos": sampler_cfg.get("neg_per_pos", ""),
                "val_f1": ms["val"].get("f1"),
                "val_auc": ms["val"].get("auc"),
                "val_pr_auc": ms["val"].get("pr_auc"),
                "test1_f1": ms["test1"].get("f1"),
                "test1_auc": ms["test1"].get("auc"),
                "test1_pr_auc": ms["test1"].get("pr_auc"),
                "test2_f1": ms["test2"].get("f1"),
                "test2_auc": ms["test2"].get("auc"),
                "test2_pr_auc": ms["test2"].get("pr_auc"),
                "threshold": ms["val"].get("threshold"),
                "saved_dir": ms["saved_dir"],
            }
            details.append(row)
            pd.DataFrame(details).to_csv(
                out_root / f"{dataset}_asdss_selection_detail_{tag}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    detail_df = pd.DataFrame(details)
    summary_rows = []
    for profile_name, g in detail_df.groupby("profile"):
        val_mean, val_std = mean_std(g[f"val_{metric}"].astype(float).tolist())
        score = val_mean - penalty * val_std
        test1_f1_mean, test1_f1_std = mean_std(g["test1_f1"].astype(float).tolist())
        test2_f1_mean, test2_f1_std = mean_std(g["test2_f1"].astype(float).tolist())
        test1_auc_mean, test1_auc_std = mean_std(g["test1_auc"].astype(float).tolist())
        test2_auc_mean, test2_auc_std = mean_std(g["test2_auc"].astype(float).tolist())
        first = g.iloc[0]
        summary_rows.append({
            "dataset": dataset,
            "profile": profile_name,
            "hard_ratio": first["hard_ratio"],
            "mid_ratio": first["mid_ratio"],
            "easy_ratio": first["easy_ratio"],
            f"val_{metric}_mean": val_mean,
            f"val_{metric}_std": val_std,
            "selection_score": score,
            "test1_f1_mean": test1_f1_mean,
            "test1_f1_std": test1_f1_std,
            "test2_f1_mean": test2_f1_mean,
            "test2_f1_std": test2_f1_std,
            "test1_auc_mean": test1_auc_mean,
            "test1_auc_std": test1_auc_std,
            "test2_auc_mean": test2_auc_mean,
            "test2_auc_std": test2_auc_std,
            "n_seeds": len(g),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    selected = summary_df.iloc[0].to_dict()
    selected_profile = selected["profile"]
    selected_detail = detail_df[detail_df["profile"] == selected_profile].copy()

    detail_path = out_root / f"{dataset}_asdss_selection_detail_{tag}.csv"
    summary_path = out_root / f"{dataset}_asdss_selection_summary_{tag}.csv"
    selected_hparams_path = out_root / f"{dataset}_asdss_selected_hparams_{tag}.csv"
    selected_test_path = out_root / f"{dataset}_asdss_selected_test_summary_{tag}.csv"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([selected]).to_csv(selected_hparams_path, index=False, encoding="utf-8-sig")
    selected_detail.to_csv(selected_test_path, index=False, encoding="utf-8-sig")

    print("\n========== A-SDSS SELECTION FINISHED ==========")
    print(f"Dataset: {dataset}")
    print(f"Metric: val_{metric}")
    print(f"Penalty: {penalty}")
    print(f"Selected profile: {selected_profile}")
    print(f"hard/mid/easy = {selected['hard_ratio']} / {selected['mid_ratio']} / {selected['easy_ratio']}")
    print(f"Selection score = {selected['selection_score']:.6f}")
    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved selected hparams: {selected_hparams_path}")
    print(f"Saved selected test summary: {selected_test_path}")


if __name__ == "__main__":
    main()
