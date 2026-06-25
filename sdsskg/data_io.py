from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return pd.read_excel(path)


def compute_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "missing_n": df.isna().sum().values,
        "missing_rate": df.isna().mean().values,
    })
