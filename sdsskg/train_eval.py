from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def make_loader(
    X: np.ndarray,
    KG: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader from numpy arrays.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    KG_t = torch.tensor(KG, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_t, KG_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(
    model: nn.Module,
    loader: DataLoader,
    device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
) -> dict:
    """
    Train one epoch.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_samples = 0

    for xb, kgb, yb in loader:
        xb = xb.to(device)
        kgb = kgb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb, kgb)
        loss = criterion(logits, yb)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = yb.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs

    return {"loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    KG: np.ndarray,
    device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Predict positive-class probability.
    """
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    KG_t = torch.tensor(KG, dtype=torch.float32)

    ds = TensorDataset(X_t, KG_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    for xb, kgb in dl:
        xb = xb.to(device)
        kgb = kgb.to(device)

        logits = model(xb, kgb)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.detach().cpu().numpy())

    if len(probs) == 0:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(probs).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model on a loader.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_y = []
    all_prob = []
    total_loss = 0.0
    n_samples = 0

    for xb, kgb, yb in loader:
        xb = xb.to(device)
        kgb = kgb.to(device)
        yb = yb.to(device)

        logits = model(xb, kgb)
        loss = criterion(logits, yb)

        prob = torch.softmax(logits, dim=1)[:, 1]

        bs = yb.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs

        all_y.append(yb.detach().cpu().numpy())
        all_prob.append(prob.detach().cpu().numpy())

    if n_samples == 0:
        return {
            "loss": np.nan,
            "auc": np.nan,
            "pr_auc": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        }

    y_true = np.concatenate(all_y).astype(int)
    y_prob = np.concatenate(all_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan

    return {
        "loss": total_loss / max(n_samples, 1),
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mode: str = "f1",
) -> float:
    """
    Search best threshold on validation set.
    mode: f1 | youden
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_thr = 0.5
    best_score = -1e18

    thresholds = np.linspace(0.05, 0.95, 181)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        if mode.lower() == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr