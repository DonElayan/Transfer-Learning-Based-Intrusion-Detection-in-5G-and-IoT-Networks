"""Loading prepared parquet, leakage-safe splits and label-budget sampling.

Splits are assigned per *feature-vector hash*: every row with the same canonical vector
lands in the same split, so exact duplicates and label-conflicting twins can never sit on
both sides of a train/test boundary (the leakage the Phase 2.2 audit warned about).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def hash_uniform(vec_hash: np.ndarray, salt: int = 0) -> np.ndarray:
    """splitmix64 finaliser -> float in [0, 1). Deterministic across runs and OSes."""
    with np.errstate(over="ignore"):
        z = vec_hash.astype(np.uint64) ^ np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def assign_split(vec_hash: np.ndarray, file_id: np.ndarray, *, val_frac: float = 0.1, test_frac: float = 0.2,
                 salt: int = 2026, test_files: tuple[int, ...] = ()) -> np.ndarray:
    """Group split by vec_hash; with test_files, those files form the test set instead."""
    u = hash_uniform(vec_hash, salt)
    split = np.full(len(u), TRAIN, dtype=np.int8)
    if test_files:
        is_test = np.isin(file_id, np.asarray(test_files))
        split[is_test] = TEST
        rest = ~is_test
        split[rest & (u < val_frac)] = VAL
    else:
        split[u < test_frac] = TEST
        split[(u >= test_frac) & (u < test_frac + val_frac)] = VAL
    return split


def cap_per_class(labels: np.ndarray, idx: np.ndarray, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Keep at most `cap` rows per class (uniformly at random); all rows when cap is None."""
    if cap is None:
        return np.sort(idx)
    keep = []
    for c in np.unique(labels[idx]):
        members = idx[labels[idx] == c]
        if len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        keep.append(members)
    return np.sort(np.concatenate(keep)) if keep else idx[:0]


def take_rows(path: Path, rows: np.ndarray, columns: list[str] | None = None, batch_size: int = 1_000_000) -> pd.DataFrame:
    """Read only the selected (sorted, global) row numbers, batch by batch."""
    rows = np.sort(rows)
    pf = pq.ParquetFile(path)
    pieces, offset = [], 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        lo, hi = np.searchsorted(rows, [offset, offset + n])
        if hi > lo:
            pieces.append(batch.take(rows[lo:hi] - offset).to_pandas())
        offset += n
    if not pieces:
        return pf.schema_arrow.empty_table().to_pandas()
    return pd.concat(pieces, ignore_index=True)


@dataclass
class DomainData:
    domain: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    split_counts: dict

    def classes(self) -> list[str]:
        return sorted(set(self.train["cls"]) | set(self.test["cls"]))


def load_domain(prepared: Path, domain: str, *, caps: dict[str, int | None], seed: int = 0,
                salt: int = 2026, val_frac: float = 0.1, test_frac: float = 0.2,
                test_files: tuple[int, ...] = ()) -> DomainData:
    path = Path(prepared) / f"{domain}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run `python -m bot_ton_tl.prepare` first")
    meta = pq.read_table(path, columns=["vec_hash", "file_id", "cls"], read_dictionary=["cls"]).to_pandas()
    split = assign_split(meta["vec_hash"].to_numpy(), meta["file_id"].to_numpy(), val_frac=val_frac,
                         test_frac=test_frac, salt=salt, test_files=test_files)
    labels = meta["cls"].to_numpy(dtype=object)
    rng = np.random.default_rng(seed)
    counts, chosen = {}, []
    for code, name in SPLIT_NAMES.items():
        idx = np.flatnonzero(split == code)
        counts[name] = pd.Series(labels[idx]).value_counts().to_dict()
        chosen.append(cap_per_class(labels, idx, caps.get(name), rng))
    rows = np.sort(np.concatenate(chosen))
    frame = take_rows(path, rows)  # one pass over the file for all three splits
    row_split = split[rows]
    frames = {name: frame.loc[row_split == code].reset_index(drop=True) for code, name in SPLIT_NAMES.items()}
    return DomainData(domain, frames["train"], frames["val"], frames["test"], counts)


def sample_budget(labels: np.ndarray, shots: int | None, rng: np.random.Generator,
                  exclude: tuple[str, ...] = ()) -> np.ndarray:
    """`shots` labelled rows per class (all available if fewer); None = every row."""
    idx = np.flatnonzero(~np.isin(labels, np.asarray(exclude, dtype=object))) if exclude else np.arange(len(labels))
    return cap_per_class(labels, idx, shots, rng)


def holdout_split(labels: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-class split of a labelled budget into fit / early-stopping parts.
    Classes with a single example go entirely to the fit part."""
    fit, hold = [], []
    for c in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == c))
        n = len(members)
        k = 0 if n < 2 else min(n - 1, max(1, int(round(n * frac))))
        hold.append(members[:k])
        fit.append(members[k:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(hold))
