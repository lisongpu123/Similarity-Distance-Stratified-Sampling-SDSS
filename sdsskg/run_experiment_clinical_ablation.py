from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sdsskg.utils import set_global_seed, device_from_str, ensure_dir, now_compact, save_json
from sdsskg.data_io import read_excel, compute_missing_report
from sdsskg.similarity import SimilarityBackend
from sdsskg.sampling import (
    RandomSampler, AllEasySampler, TraditionalHNSampler,
    SDSSSampler, SDSS31Sampler, SDSS32Sampler, SDSSConfig
)
from sdsskg.kge import KGEEncoder
from sdsskg.model import SDSSKGClassifier
from sdsskg.train_eval import make_loader, train_one, evaluate, predict_proba, find_best_threshold
from sdsskg.config import load_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def _robust_parse_numeric(x):
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s == "":
        return np.nan
    if s in {"无", "不详", "未知", "NA", "N/A", "nan", "NaN", "-", "—", "--", "/", "\\"}:
        return np.nan

    s = s.replace("，", ",").replace("：", ":").replace("（", "(").replace("）", ")").replace("％", "%")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan
    vals = []
    for n in nums:
        try:
            vals.append(float(n))
        except Exception:
            continue
    if not vals:
        return np.nan
    if ("-" in s or "～" in s or "~" in s) and len(vals) >= 2:
        return float(sum(vals) / len(vals))
    return float(vals[0])


def map_label(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    pos = s.isin(["正样本", "阳性", "1", "True", "true", "POS", "pos", "positive"])
    neg = s.isin(["负样本", "阴性", "0", "False", "false", "NEG", "neg", "negative"])
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[pos] = 1
    out[neg] = 0
    s2 = s.replace({"1.0": "1", "0.0": "0"})
    is01 = s2.isin(["0", "1"])
    out[is01] = s2[is01].astype(int)
    if out.isna().any():
        bad = series[out.isna()].head(5).tolist()
        raise ValueError(f"Unrecognized label values found (showing up to 5): {bad}")
    return out.astype(int).to_numpy()


def build_entity_list(df: pd.DataFrame) -> pd.Series:
    entity_cols = [c for c in df.columns if "实体" in str(c)]
    if not entity_cols:
        return pd.Series([[]] * len(df), index=df.index)

    def split_entities(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        s = str(v).strip()
        if not s:
            return []
        for d in [";", "；", ",", "，", "|", "、"]:
            s = s.replace(d, " ")
        return [x.strip() for x in s.split() if x.strip()]

    out = []
    for _, row in df[entity_cols].iterrows():
        ents = []
        for c in entity_cols:
            ents.extend(split_entities(row[c]))
        out.append(ents)
    return pd.Series(out, index=df.index)


def _infer_and_exclude_cols(df: pd.DataFrame, label_col: str, cfg: dict) -> List[str]:
    """
    Strict anti-leakage feature exclusion.

    Label, disease/diagnosis, grouping, misdiagnosis/category, source and
    clinical-category columns must never enter model features. These columns may
    be used only as metadata for training-set sampling.
    """
    patterns = [
        "ID", "编号", "住院号", "门诊号", "病案号", "患者ID", "姓名", "电话", "手机号",
        "疾病", "诊断", "标签", "label", "备注", "核实", "误诊", "结局", "分组",
        "临床类别", "样本类别", "样本类型", "负例类型", "负样本类型",
        "难负例", "简单负例", "普通负例", "难易类别", "难度类别",
        "疾病列表", "疾病类别", "病例类别", "采样类别", "source", "来源"
    ]

    out = [label_col]

    for c in cfg.get("data", {}).get("id_like_cols", []) or []:
        if c in df.columns:
            out.append(c)

    # Allow manual exclusion from yaml. This is useful when a new metadata column
    # appears in the Excel files.
    for c in cfg.get("data", {}).get("exclude_feature_cols", []) or []:
        if c in df.columns:
            out.append(c)

    for c in df.columns:
        sc = str(c).strip()
        scl = sc.lower()

        if sc == label_col or "标签" in sc or "label" in scl:
            out.append(c)
            continue

        if "疾病" in sc or "诊断" in sc or "diagnosis" in scl or "disease" in scl:
            out.append(c)
            continue

        if any(p.lower() in scl for p in patterns):
            out.append(c)
            continue

    return list(dict.fromkeys(out))


def _choose_text_cols(df: pd.DataFrame, cfg: dict) -> List[str]:
    text_cols = cfg.get("data", {}).get("text_feature_cols")
    if isinstance(text_cols, list) and len(text_cols) > 0:
        banned = {"标签", "疾病", "疾病（正）"}
        return [c for c in text_cols if c in df.columns and c not in banned]

    candidates = ["自查症状", "检查症状", "现病史", "主诉", "症状", "病情描述"]
    found = [c for c in candidates if c in df.columns]
    return found


def _combine_text(df: pd.DataFrame, text_cols: Sequence[str]) -> pd.Series:
    if not text_cols:
        return pd.Series([""] * len(df), index=df.index)

    def _row_text(row):
        parts = []
        for c in text_cols:
            v = row.get(c, "")
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            sv = str(v).strip()
            if not sv or sv in {"无", "nan", "NaN", "None"}:
                continue
            parts.append(sv)
        return " [SEP] ".join(parts)

    return df.apply(_row_text, axis=1)


def _build_numeric_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    data = {}
    for c in feature_cols:
        if c in df.columns:
            data[c] = df[c].map(_robust_parse_numeric).astype(float)
        else:
            data[c] = np.nan
    return pd.DataFrame(data, columns=feature_cols)


def _fit_imputer(df: pd.DataFrame, strategy: str = "median") -> dict:
    fills = {}
    strategy = (strategy or "median").lower()
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            fills[c] = 0.0
        elif strategy in {"mean", "avg"}:
            fills[c] = float(s.mean())
        elif strategy in {"mode", "most_frequent"}:
            fills[c] = float(s.mode().iloc[0])
        else:
            fills[c] = float(s.median())
    return fills


def _apply_imputer(df: pd.DataFrame, fills: dict) -> pd.DataFrame:
    out = df.copy()
    for c, v in fills.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v)
    return out


def _zscore_fit(df: pd.DataFrame) -> tuple[dict, dict]:
    means, stds = {}, {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        means[c] = float(s.mean())
        std = float(s.std())
        stds[c] = std if std > 1e-12 else 1.0
    return means, stds


def _zscore_apply(df: pd.DataFrame, means: dict, stds: dict) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = (pd.to_numeric(out[c], errors="coerce") - means[c]) / stds[c]
    return out


def _ensure_finite_np(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def _cosine_distance_to_center(emb: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    pos = emb[pos_mask]
    if len(pos) == 0:
        return np.ones((emb.shape[0],), dtype=np.float32)
    center = pos.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_norm @ center.reshape(-1, 1)
    return (1.0 - sims.reshape(-1)).astype(np.float32)



def _detect_clinical_category_col(df: pd.DataFrame, cfg: dict) -> str | None:
    """
    Detect the clinical negative-category column.

    This column is only used for ordering the training negative pool during
    sampling. It is excluded from model features by _infer_and_exclude_cols.
    """
    sampler_cfg = cfg.get("sampler", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    for key in ["clinical_category_col", "negative_category_col", "sample_category_col"]:
        v = sampler_cfg.get(key) or data_cfg.get(key)
        if v and v in df.columns:
            return v

    candidates = [
        "临床类别", "样本类别", "样本类型", "负例类型", "负样本类型",
        "难易类别", "难度类别", "采样类别", "疾病列表类别",
        "疾病类别", "病例类别", "来源", "source"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        sc = str(c).strip()
        if any(k in sc for k in ["临床", "难负例", "简单负例", "负例类型", "样本类型", "疾病列表"]):
            return c

    return None


def _clinical_rank_from_value(v) -> int:
    """
    Clinical-priority difficulty rank. Smaller value means harder negative.

    This rank is used only for sampling within the training negative pool. It is
    never used as a predictive model feature.
    """
    if v is None:
        return 3
    try:
        if pd.isna(v):
            return 3
    except Exception:
        pass

    s = str(v).strip().lower()

    if any(k in s for k in ["误诊", "misdiagnosis", "misdiagnosed"]):
        return 0
    if any(k in s for k in ["难负例", "困难负例", "hard", "difficult", "confusing", "混淆"]):
        return 1
    if any(k in s for k in ["中等", "中难", "相关", "middle", "mid", "moderate"]):
        return 2
    if any(k in s for k in ["简单", "普通", "easy", "common", "一般"]):
        return 3
    return 3


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _attach_clinical_difficulty_score(
    neg_pool: pd.DataFrame,
    cfg: dict,
    dist_col: str = "distance",
) -> tuple[pd.DataFrame, str, dict]:
    """
    Build a clinical-first and distance-second difficulty score.

    Principle:
    1) Clinical category determines the coarse difficulty order:
       misdiagnosis > hard negative > middle negative > common/easy negative.
    2) Similarity distance only orders samples within the clinical negative pool.
    3) The score is used only for training negative sampling, not as model input.
    4) Test labels, external-test performance, and disease/diagnosis columns are
       never used to select the sampling profile.
    """
    neg_pool = neg_pool.copy()
    clinical_col = _detect_clinical_category_col(neg_pool, cfg)

    if clinical_col is None:
        neg_pool["clinical_rank"] = 0
        neg_pool["distance_rank_norm"] = _minmax01(neg_pool[dist_col].to_numpy(dtype=np.float32))
        neg_pool["difficulty_score"] = neg_pool["distance_rank_norm"]
        info = {
            "mode": "distance_only",
            "clinical_category_col": None,
            "note": "No clinical category column detected. Fallback to distance-only SDSS.",
            "leakage_control": "Disease/diagnosis/category columns are excluded from model features."
        }
        return neg_pool, "difficulty_score", info

    neg_pool["clinical_rank"] = neg_pool[clinical_col].map(_clinical_rank_from_value).astype(int)
    neg_pool["distance_rank_norm"] = _minmax01(neg_pool[dist_col].to_numpy(dtype=np.float32))

    # Clinical rank is primary; distance is secondary. The 0.49 multiplier keeps
    # distance from overriding the clinical category order.
    neg_pool["difficulty_score"] = (
        neg_pool["clinical_rank"].astype(float)
        + 0.49 * neg_pool["distance_rank_norm"].astype(float)
    )

    rank_counts = neg_pool["clinical_rank"].value_counts().sort_index().to_dict()
    category_counts = neg_pool[clinical_col].astype(str).value_counts().head(30).to_dict()

    info = {
        "mode": "clinical_then_distance",
        "clinical_category_col": clinical_col,
        "clinical_rank_counts": {str(k): int(v) for k, v in rank_counts.items()},
        "clinical_category_counts_top30": {str(k): int(v) for k, v in category_counts.items()},
        "difficulty_score": "clinical_rank + 0.49 * normalized_distance",
        "leakage_control": "Clinical category is used only for training negative sampling and is excluded from model features."
    }
    return neg_pool, "difficulty_score", info

def _pick_sampler(cfg: dict):
    sampler_name = str(cfg.get("sampler", {}).get("name", "sdss")).lower()
    seed = int(cfg["train"].get("seed", 42))
    neg_per_pos = float(cfg.get("sampler", {}).get("neg_per_pos", 1))

    if sampler_name == "random":
        return RandomSampler(seed=seed), neg_per_pos
    if sampler_name == "all_easy":
        return AllEasySampler(seed=seed), neg_per_pos
    if sampler_name in {"all_hard", "traditional_hn", "hn"}:
        return TraditionalHNSampler(seed=seed), neg_per_pos

    n_bins = int(cfg.get("sampler", {}).get("strata", cfg.get("sampler", {}).get("n_bins", 10)))
    min_bin_size = int(cfg.get("sampler", {}).get("min_bin_size", 5))

    if sampler_name in {"sdss31", "sdss_31", "sdss3.1"}:
        hard_ratio = float(cfg.get("sampler", {}).get("hard_ratio", 0.4))
        mid_ratio = float(cfg.get("sampler", {}).get("mid_ratio", 0.4))
        easy_ratio = float(cfg.get("sampler", {}).get("easy_ratio", 0.2))
        sampler = SDSS31Sampler(
            config=SDSSConfig(n_bins=n_bins, seed=seed, min_bin_size=min_bin_size),
            seed=seed,
            hard_ratio=hard_ratio,
            mid_ratio=mid_ratio,
            easy_ratio=easy_ratio,
        )
        return sampler, neg_per_pos

    if sampler_name in {"sdss32", "sdss_32", "sdss3.2"}:
        hard_ratio = float(cfg.get("sampler", {}).get("hard_ratio", 0.4))
        mid_ratio = float(cfg.get("sampler", {}).get("mid_ratio", 0.4))
        easy_ratio = float(cfg.get("sampler", {}).get("easy_ratio", 0.2))
        sampler = SDSS32Sampler(
            config=SDSSConfig(n_bins=n_bins, seed=seed, min_bin_size=min_bin_size),
            seed=seed,
            hard_ratio=hard_ratio,
            mid_ratio=mid_ratio,
            easy_ratio=easy_ratio,
        )
        return sampler, neg_per_pos

    sampler = SDSSSampler(
        config=SDSSConfig(n_bins=n_bins, seed=seed, min_bin_size=min_bin_size),
        seed=seed
    )
    return sampler, neg_per_pos


def _binary_metrics(y_true, y_prob, thr):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, confusion_matrix, brier_score_loss
    )
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(thr)).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": brier,
        "threshold": float(thr),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def _load_test(cfg, prefix: str) -> pd.DataFrame:
    if cfg["data"].get(f"{prefix}_path"):
        return read_excel(cfg["data"][f"{prefix}_path"], cfg["data"].get(f"{prefix}_sheet")).copy()

    pos_key = f"{prefix}_pos_path"
    neg_key = f"{prefix}_neg_path"
    if cfg["data"].get(pos_key) and cfg["data"].get(neg_key):
        pos = read_excel(cfg["data"][pos_key], cfg["data"].get(f"{prefix}_pos_sheet")).copy()
        neg = read_excel(cfg["data"][neg_key], cfg["data"].get(f"{prefix}_neg_sheet")).copy()
        label_col = cfg["data"].get("label_col_raw", "标签")
        pos[label_col] = "正样本"
        neg[label_col] = "负样本"
        return pd.concat([pos, neg], axis=0).reset_index(drop=True)

    raise ValueError(f"Missing config for {prefix}. Provide {prefix}_path OR {prefix}_pos_path + {prefix}_neg_path.")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["run_name"] / now_compact()
    ensure_dir(out_dir)

    seed = int(cfg["train"].get("seed", 42))
    set_global_seed(seed, deterministic=True)
    device = device_from_str(cfg["train"].get("device", "cpu"))

    label_col = cfg["data"].get("label_col_raw", "标签")

    train_pos = read_excel(cfg["data"]["train_pos_path"], cfg["data"].get("train_pos_sheet")).copy()
    train_neg = read_excel(cfg["data"]["train_neg_path"], cfg["data"].get("train_neg_sheet")).copy()
    train_pos[label_col] = "正样本"
    train_neg[label_col] = "负样本"
    train_df = pd.concat([train_pos, train_neg], axis=0).reset_index(drop=True)

    test1_df = _load_test(cfg, "test1")
    test2_df = _load_test(cfg, "test2")

    save_json({
        "train_shape": list(train_df.shape),
        "test1_shape": list(test1_df.shape),
        "test2_shape": list(test2_df.shape),
    }, out_dir / "shapes.json")

    compute_missing_report(train_df).to_csv(out_dir / "missing_train.csv", index=False, encoding="utf-8-sig")
    compute_missing_report(test1_df).to_csv(out_dir / "missing_test1.csv", index=False, encoding="utf-8-sig")
    compute_missing_report(test2_df).to_csv(out_dir / "missing_test2.csv", index=False, encoding="utf-8-sig")

    y_all = map_label(train_df[label_col])
    y_test1 = map_label(test1_df[label_col])
    y_test2 = map_label(test2_df[label_col])

    exclude_cols = _infer_and_exclude_cols(train_df, label_col, cfg)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    X_test1_num_df = _build_numeric_X(test1_df, feature_cols)
    X_test2_num_df = _build_numeric_X(test2_df, feature_cols)

    text_cols = _choose_text_cols(train_df, cfg)
    train_text = _combine_text(train_df, text_cols)
    test1_text = _combine_text(test1_df, text_cols)
    test2_text = _combine_text(test2_df, text_cols)

    sim_cfg = cfg.get("similarity", {})
    sim = SimilarityBackend(
        model_path=sim_cfg.get("model_path", sim_cfg.get("local_model_path", "multilingual-MiniLM-L12-v2")),
        device=str(device),
        batch_size=int(sim_cfg.get("batch_size", 32)),
        max_length=int(sim_cfg.get("max_length", 256)),
    )

    text_all_emb = sim.fit_transform(train_text.tolist()).astype(np.float32)
    text_test1_emb = sim.transform(test1_text.tolist()).astype(np.float32)
    text_test2_emb = sim.transform(test2_text.tolist()).astype(np.float32)

    pos_mask = (y_all == 1)
    neg_mask = (y_all == 0)

    pos_df = train_df.loc[pos_mask].reset_index(drop=True)
    neg_df = train_df.loc[neg_mask].reset_index(drop=True)

    all_dist = _cosine_distance_to_center(text_all_emb, pos_mask)
    neg_dist = all_dist[neg_mask]

    neg_pool = neg_df.copy()
    neg_pool["distance"] = neg_dist

    # Clinical-category first + similarity-distance second.
    # The resulting difficulty_score is used only for training negative sampling.
    neg_pool, sampler_dist_col, clinical_sampling_info = _attach_clinical_difficulty_score(
        neg_pool=neg_pool,
        cfg=cfg,
        dist_col="distance",
    )
    save_json(clinical_sampling_info, out_dir / "clinical_sampling_info.json")

    cols_to_save = []
    clinical_col = clinical_sampling_info.get("clinical_category_col")
    if clinical_col and clinical_col in neg_pool.columns:
        cols_to_save.append(clinical_col)
    for c in ["distance", "clinical_rank", "distance_rank_norm", "difficulty_score"]:
        if c in neg_pool.columns:
            cols_to_save.append(c)
    if cols_to_save:
        neg_pool[cols_to_save].to_csv(
            out_dir / "negative_pool_difficulty_ranking.csv",
            index=False,
            encoding="utf-8-sig"
        )

    sampler, neg_per_pos = _pick_sampler(cfg)
    n_need = max(1, int(np.ceil(len(pos_df) * neg_per_pos)))
    try:
        sampled_neg = sampler.sample(
            positives=pos_df,
            pool=neg_pool,
            k=n_need,
            dist_col=sampler_dist_col
        ).reset_index(drop=True)
    except TypeError:
        sampled_neg = sampler.sample(pos_df, neg_pool).reset_index(drop=True)
        if len(sampled_neg) > n_need:
            sampled_neg = sampled_neg.iloc[:n_need].reset_index(drop=True)

    bal_df = pd.concat([
        pos_df,
        sampled_neg.drop(
            columns=["distance", "clinical_rank", "distance_rank_norm", "difficulty_score"],
            errors="ignore"
        )
    ], axis=0).reset_index(drop=True)
    y_bal = map_label(bal_df[label_col])

    X_bal_num_df = _build_numeric_X(bal_df, feature_cols)
    bal_text = _combine_text(bal_df, text_cols)
    bal_text_emb = sim.transform(bal_text.tolist()).astype(np.float32)

    ents_bal = build_entity_list(bal_df)
    ents_test1 = build_entity_list(test1_df)
    ents_test2 = build_entity_list(test2_df)

    idx = np.arange(len(bal_df))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(cfg["train"].get("val_ratio", 0.2)),
        random_state=seed,
        stratify=y_bal
    )

    X_tr_num_df = X_bal_num_df.iloc[tr_idx].reset_index(drop=True)
    X_va_num_df = X_bal_num_df.iloc[va_idx].reset_index(drop=True)

    imputer = _fit_imputer(X_tr_num_df, strategy=cfg.get("impute", {}).get("strategy", "median"))
    X_tr_num_df = _apply_imputer(X_tr_num_df, imputer)
    X_va_num_df = _apply_imputer(X_va_num_df, imputer)
    X_test1_num_df = _apply_imputer(X_test1_num_df, imputer)
    X_test2_num_df = _apply_imputer(X_test2_num_df, imputer)

    means, stds = _zscore_fit(X_tr_num_df)
    X_tr_num_df = _zscore_apply(X_tr_num_df, means, stds)
    X_va_num_df = _zscore_apply(X_va_num_df, means, stds)
    X_test1_num_df = _zscore_apply(X_test1_num_df, means, stds)
    X_test2_num_df = _zscore_apply(X_test2_num_df, means, stds)

    X_tr = np.concatenate([X_tr_num_df.to_numpy(dtype=np.float32), bal_text_emb[tr_idx].astype(np.float32)], axis=1)
    X_va = np.concatenate([X_va_num_df.to_numpy(dtype=np.float32), bal_text_emb[va_idx].astype(np.float32)], axis=1)
    X_test1 = np.concatenate([X_test1_num_df.to_numpy(dtype=np.float32), text_test1_emb.astype(np.float32)], axis=1)
    X_test2 = np.concatenate([X_test2_num_df.to_numpy(dtype=np.float32), text_test2_emb.astype(np.float32)], axis=1)

    X_tr = _ensure_finite_np(X_tr)
    X_va = _ensure_finite_np(X_va)
    X_test1 = _ensure_finite_np(X_test1)
    X_test2 = _ensure_finite_np(X_test2)

    use_kg = bool(cfg.get("model", {}).get("use_kg", True))
    if use_kg:
        kge_cfg = cfg["kge"]
        kge = KGEEncoder(
            entity_emb_path=kge_cfg["entity_emb_path"],
            entity2id_json=kge_cfg["entity2id_json"],
            pad_token=kge_cfg.get("pad_token", "<PAD>"),
            agg=kge_cfg.get("agg", "mean"),
        )
        KG_tr = _ensure_finite_np(kge.encode_entity_series(ents_bal.iloc[tr_idx].reset_index(drop=True)))
        KG_va = _ensure_finite_np(kge.encode_entity_series(ents_bal.iloc[va_idx].reset_index(drop=True)))
        KG_test1 = _ensure_finite_np(kge.encode_entity_series(ents_test1))
        KG_test2 = _ensure_finite_np(kge.encode_entity_series(ents_test2))
    else:
        kg_dim_fallback = int(cfg.get("model", {}).get("kg_dim_fallback", 64))
        KG_tr = np.zeros((len(tr_idx), kg_dim_fallback), dtype=np.float32)
        KG_va = np.zeros((len(va_idx), kg_dim_fallback), dtype=np.float32)
        KG_test1 = np.zeros((len(test1_df), kg_dim_fallback), dtype=np.float32)
        KG_test2 = np.zeros((len(test2_df), kg_dim_fallback), dtype=np.float32)

    y_tr = y_bal[tr_idx]
    y_va = y_bal[va_idx]

    model = SDSSKGClassifier(
        x_dim=X_tr.shape[1],
        kg_dim=KG_tr.shape[1],
        hidden_dim=int(cfg["model"].get("hidden_dim", 128)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
        use_kg=use_kg,
        num_classes=2,
    ).to(device)

    train_loader = make_loader(X_tr, KG_tr, y_tr, batch_size=int(cfg["train"].get("batch_size", 16)), shuffle=True)
    val_loader = make_loader(X_va, KG_va, y_va, batch_size=int(cfg["train"].get("batch_size", 16)), shuffle=False)

    best_state = None
    best_score = -1e18
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 8))
    epochs = int(cfg["train"].get("epochs", 30))
    history = []

    for epoch in range(1, epochs + 1):
        train_stat = train_one(
            model=model,
            loader=train_loader,
            device=device,
            lr=float(cfg["train"].get("lr", 1e-3)),
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
            grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
        )
        val_stat = evaluate(model, val_loader, device=device, threshold=0.5)
        score = val_stat.get("auc", 0.0)
        history.append({"epoch": epoch, "train": train_stat, "val": val_stat})

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob = predict_proba(model, X_va, KG_va, device=device)
    thr = find_best_threshold(y_va, val_prob, mode=str(cfg.get("eval", {}).get("threshold_strategy", "f1")).lower())
    test1_prob = predict_proba(model, X_test1, KG_test1, device=device)
    test2_prob = predict_proba(model, X_test2, KG_test2, device=device)

    val_metrics = _binary_metrics(y_va, val_prob, thr)
    test1_metrics = _binary_metrics(y_test1, test1_prob, thr)
    test2_metrics = _binary_metrics(y_test2, test2_prob, thr)

    save_json(cfg, out_dir / "config_used.json")
    save_json({"use_kg": use_kg, "kg_dim_used": int(KG_tr.shape[1])}, out_dir / "kg_usage.json")
    save_json({"threshold": float(thr)}, out_dir / "best_threshold.json")
    save_json(history, out_dir / "train_history.json")
    save_json(val_metrics, out_dir / "metrics_val.json")
    save_json(test1_metrics, out_dir / "metrics_test1.json")
    save_json(test2_metrics, out_dir / "metrics_test2.json")

    pd.DataFrame({"excluded_feature_cols": exclude_cols}).to_csv(out_dir / "excluded_feature_cols.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_cols_numeric": feature_cols}).to_csv(out_dir / "feature_cols_numeric.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"text_feature_cols": text_cols}).to_csv(out_dir / "text_feature_cols.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": y_test1, "y_prob": test1_prob, "y_pred": (test1_prob >= thr).astype(int)}).to_csv(out_dir / "predictions_test1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": y_test2, "y_prob": test2_prob, "y_pred": (test2_prob >= thr).astype(int)}).to_csv(out_dir / "predictions_test2.csv", index=False, encoding="utf-8-sig")

    print("Validation:", val_metrics)
    print("Test1:", test1_metrics)
    print("Test2:", test2_metrics)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
