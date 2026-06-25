from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def device_from_str(s: str = "auto"):
    try:
        import torch
        if s == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(s)
    except Exception:
        return "cpu"


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_compact():
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
