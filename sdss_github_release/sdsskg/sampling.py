from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


class BaseSampler:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        raise NotImplementedError


class RandomSampler(BaseSampler):
    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        if k is None:
            k = len(positives)
        if len(pool) <= k:
            return pool.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        idx = self.rng.choice(len(pool), size=int(k), replace=False)
        return pool.iloc[idx].reset_index(drop=True)


class AllEasySampler(BaseSampler):
    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        if k is None:
            k = len(positives)
        df = pool.sort_values(dist_col, ascending=False).reset_index(drop=True)
        return df.iloc[: min(int(k), len(df))].reset_index(drop=True)


class TraditionalHNSampler(BaseSampler):
    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        if k is None:
            k = len(positives)
        df = pool.sort_values(dist_col, ascending=True).reset_index(drop=True)
        return df.iloc[: min(int(k), len(df))].reset_index(drop=True)


@dataclass
class SDSSConfig:
    n_bins: int = 10
    seed: int = 42
    min_bin_size: int = 5


class SDSSSampler(BaseSampler):
    def __init__(self, config: SDSSConfig, seed: int = 42):
        super().__init__(seed=seed)
        self.config = config

    def _make_bins(self, pool: pd.DataFrame, dist_col: str) -> List[pd.DataFrame]:
        df = pool.sort_values(dist_col, ascending=True).reset_index(drop=True)
        n = len(df)
        n_bins = max(1, int(self.config.n_bins))
        if n == 0:
            return []
        if n_bins * max(1, self.config.min_bin_size) > n:
            n_bins = max(1, n // max(1, self.config.min_bin_size))
            n_bins = max(1, n_bins)
        indices = np.array_split(np.arange(n), n_bins)
        return [df.iloc[idx].reset_index(drop=True) for idx in indices if len(idx) > 0]

    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        if k is None:
            k = len(positives)
        k = int(k)
        bins = self._make_bins(pool, dist_col)
        if not bins:
            return pool.iloc[:0].copy()
        taken = []
        remaining = k
        nonempty = [b.copy() for b in bins]
        while remaining > 0 and any(len(b) > 0 for b in nonempty):
            for i, b in enumerate(nonempty):
                if remaining <= 0:
                    break
                if len(b) == 0:
                    continue
                row_idx = self.rng.integers(0, len(b))
                taken.append(b.iloc[[row_idx]])
                nonempty[i] = b.drop(index=row_idx).reset_index(drop=True)
                remaining -= 1
        if not taken:
            return pool.iloc[:0].copy()
        return pd.concat(taken, axis=0).reset_index(drop=True)


class _BiasMixin:
    def _group_bins(self, bins: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        m = len(bins)
        if m == 1:
            return bins, [], []
        if m == 2:
            return bins[:1], bins[1:], []
        hard_end = max(1, int(np.ceil(m * 0.34)))
        easy_start = max(hard_end + 1, int(np.floor(m * 0.67)))
        hard_bins = bins[:hard_end]
        mid_bins = bins[hard_end:easy_start]
        easy_bins = bins[easy_start:]
        return hard_bins, mid_bins, easy_bins

    def _make_bins(self, pool: pd.DataFrame, dist_col: str) -> List[pd.DataFrame]:
        df = pool.sort_values(dist_col, ascending=True).reset_index(drop=True)
        n = len(df)
        n_bins = max(1, int(self.config.n_bins))
        if n == 0:
            return []
        if n_bins * max(1, self.config.min_bin_size) > n:
            n_bins = max(1, n // max(1, self.config.min_bin_size))
            n_bins = max(1, n_bins)
        indices = np.array_split(np.arange(n), n_bins)
        return [df.iloc[idx].reset_index(drop=True) for idx in indices if len(idx) > 0]

    def _take_from_group(self, group_bins: List[pd.DataFrame], need: int) -> List[pd.DataFrame]:
        if need <= 0 or not group_bins:
            return []
        chosen = []
        bins = [b.copy() for b in group_bins]
        remaining = int(need)
        for i, b in enumerate(bins):
            if remaining <= 0:
                break
            if len(b) == 0:
                continue
            row_idx = self.rng.integers(0, len(b))
            chosen.append(b.iloc[[row_idx]])
            bins[i] = b.drop(index=row_idx).reset_index(drop=True)
            remaining -= 1
        while remaining > 0 and any(len(b) > 0 for b in bins):
            nonempty_ids = [i for i, b in enumerate(bins) if len(b) > 0]
            gid = int(self.rng.choice(nonempty_ids))
            b = bins[gid]
            row_idx = self.rng.integers(0, len(b))
            chosen.append(b.iloc[[row_idx]])
            bins[gid] = b.drop(index=row_idx).reset_index(drop=True)
            remaining -= 1
        return chosen

    def _biased_sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int], dist_col: str) -> pd.DataFrame:
        if k is None:
            k = len(positives)
        k = int(k)
        bins = self._make_bins(pool, dist_col)
        if not bins:
            return pool.iloc[:0].copy()

        hard_bins, mid_bins, easy_bins = self._group_bins(bins)
        k_hard = int(round(k * self.hard_ratio))
        k_mid = int(round(k * self.mid_ratio))
        k_easy = int(k - k_hard - k_mid)

        chosen = []
        chosen.extend(self._take_from_group(hard_bins, k_hard))
        chosen.extend(self._take_from_group(mid_bins, k_mid))
        chosen.extend(self._take_from_group(easy_bins, k_easy))

        if len(chosen) < k:
            already = pd.concat(chosen, axis=0).reset_index(drop=True) if chosen else pool.iloc[:0].copy()
            remaining_pool = pool.copy()
            if len(already) > 0:
                key_cols = list(remaining_pool.columns)
                rem_tmp = remaining_pool.astype(str).agg("||".join, axis=1)
                sel_tmp = set(already[key_cols].astype(str).agg("||".join, axis=1).tolist())
                remaining_pool = remaining_pool.loc[~rem_tmp.isin(sel_tmp)].copy()
            backfill = remaining_pool.sort_values(dist_col, ascending=True).head(k - len(chosen))
            if len(backfill) > 0:
                chosen.append(backfill)

        if not chosen:
            return pool.iloc[:0].copy()
        out = pd.concat(chosen, axis=0).reset_index(drop=True)
        if len(out) > k:
            out = out.iloc[:k].reset_index(drop=True)
        return out


class SDSS31Sampler(BaseSampler, _BiasMixin):
    def __init__(self, config: SDSSConfig, seed: int = 42, hard_ratio: float = 0.4, mid_ratio: float = 0.4, easy_ratio: float = 0.2):
        BaseSampler.__init__(self, seed=seed)
        self.config = config
        self.hard_ratio = float(hard_ratio)
        self.mid_ratio = float(mid_ratio)
        self.easy_ratio = float(easy_ratio)

    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        return self._biased_sample(positives, pool, k, dist_col)


class SDSS32Sampler(BaseSampler, _BiasMixin):
    def __init__(self, config: SDSSConfig, seed: int = 42, hard_ratio: float = 0.40, mid_ratio: float = 0.40, easy_ratio: float = 0.20):
        BaseSampler.__init__(self, seed=seed)
        self.config = config
        self.hard_ratio = float(hard_ratio)
        self.mid_ratio = float(mid_ratio)
        self.easy_ratio = float(easy_ratio)
        s = self.hard_ratio + self.mid_ratio + self.easy_ratio
        if abs(s - 1.0) > 1e-8:
            raise ValueError("hard_ratio + mid_ratio + easy_ratio must equal 1.0")

    def sample(self, positives: pd.DataFrame, pool: pd.DataFrame, k: Optional[int] = None, dist_col: str = "distance") -> pd.DataFrame:
        return self._biased_sample(positives, pool, k, dist_col)
