# -*- coding: utf-8 -*-
"""
Plot clinical-category distance distribution for SDSS3.2.

功能：
1. 读取四个病种的 *_negative_distance_detail.csv
2. 按临床类别绘制 cosine distance 箱线图 + 散点
3. 输出高清 PNG 和 PDF

使用前确认你已经运行过：
python -m sdsskg.compute_clinical_distance_distribution --base_config ...

输入目录示例：
outputs/distance_distribution/

输出：
Fig_similarity_distance_distribution.png
Fig_similarity_distance_distribution.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 修改这里：你的结果目录
# =========================
ROOT = Path(r"outputs/distance_distribution")

# 如果你的 outputs 不在当前目录，可以改成绝对路径，例如：
# ROOT = Path(r"H:\iScience新代码\sdsskg_full_v7_two_external_no_disease\outputs\distance_distribution")


# =========================
# 2. 数据集显示名称
# =========================
DATASETS = {
    "dn": "Diabetic Nephropathy",
    "df": "Diabetic Foot",
    "dpn": "Diabetic Peripheral Neuropathy",
    "dka": "Diabetic Ketoacidosis",
}

# 论文中建议使用的类别顺序
CATEGORY_ORDER = [
    "误诊难负例",
    "难负例",
    "中等难负例",
    "普通负例",
    "简单负例",
]

CATEGORY_LABELS_EN = {
    "误诊难负例": "Misdiagnosed\nnegatives",
    "难负例": "Hard\nnegatives",
    "中等难负例": "Middle\nnegatives",
    "普通负例": "Common\nnegatives",
    "简单负例": "Easy\nnegatives",
}


def find_latest_detail_file(root: Path, dataset: str) -> Path:
    """
    自动寻找每个病种最新的 negative distance detail 文件。
    """
    ds_dir = root / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Cannot find dataset directory: {ds_dir}")

    files = list(ds_dir.rglob("*_negative_distance_detail.csv"))
    if not files:
        raise FileNotFoundError(f"No *_negative_distance_detail.csv found under: {ds_dir}")

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_dataset_distance(root: Path, dataset: str) -> pd.DataFrame:
    """
    读取单个病种的 distance 明细。
    """
    file_path = find_latest_detail_file(root, dataset)
    print(f"[LOAD] {dataset}: {file_path}")

    df = pd.read_csv(file_path)

    # 兼容不同列名
    if "临床类别_规范" not in df.columns:
        raise ValueError(f"Column '临床类别_规范' not found in {file_path}")

    if "distance" not in df.columns:
        raise ValueError(f"Column 'distance' not found in {file_path}")

    df = df.copy()
    df["dataset"] = dataset
    df["dataset_name"] = DATASETS.get(dataset, dataset.upper())
    df["category"] = df["临床类别_规范"].astype(str)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["distance"])

    return df


def plot_distance_distribution():
    all_rows = []
    for ds in DATASETS:
        all_rows.append(load_dataset_distance(ROOT, ds))

    data = pd.concat(all_rows, axis=0).reset_index(drop=True)

    # =========================
    # 3. 绘图
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    rng = np.random.default_rng(42)

    for ax, ds in zip(axes, DATASETS.keys()):
        sub = data[data["dataset"] == ds].copy()
        title = DATASETS[ds]

        # 当前数据集中实际存在的类别
        cats = [c for c in CATEGORY_ORDER if c in sub["category"].unique()]

        box_data = []
        x_positions = []
        labels = []

        for i, cat in enumerate(cats, start=1):
            vals = sub.loc[sub["category"] == cat, "distance"].dropna().values
            if len(vals) == 0:
                continue
            box_data.append(vals)
            x_positions.append(i)
            labels.append(CATEGORY_LABELS_EN.get(cat, cat))

        # 箱线图
        bp = ax.boxplot(
            box_data,
            positions=x_positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            medianprops={"linewidth": 1.5},
            meanprops={"marker": "D", "markersize": 4},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.35},
        )

        # 给箱体填充浅色
        for patch in bp["boxes"]:
            patch.set_alpha(0.35)

        # 叠加散点，增强可读性
        for i, cat in enumerate(cats, start=1):
            vals = sub.loc[sub["category"] == cat, "distance"].dropna().values

            # 样本太多时抽样显示，避免图太密
            if len(vals) > 180:
                vals_show = rng.choice(vals, size=180, replace=False)
            else:
                vals_show = vals

            jitter = rng.normal(loc=0, scale=0.045, size=len(vals_show))
            ax.scatter(
                np.full(len(vals_show), i) + jitter,
                vals_show,
                s=12,
                alpha=0.35,
                edgecolors="none",
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Cosine distance to positive centroid", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(bottom=0)

        # 标注样本量
        for i, cat in enumerate(cats, start=1):
            n = int((sub["category"] == cat).sum())
            vals = sub.loc[sub["category"] == cat, "distance"].dropna().values
            if len(vals) > 0:
                y = np.nanmax(vals)
                ax.text(i, y + 0.015, f"n={n}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Similarity distance distribution across clinically defined negative categories",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = Path("Fig_similarity_distance_distribution.png")
    out_pdf = Path("Fig_similarity_distance_distribution.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print(f"[SAVED] {out_png.resolve()}")
    print(f"[SAVED] {out_pdf.resolve()}")


if __name__ == "__main__":
    plot_distance_distribution()