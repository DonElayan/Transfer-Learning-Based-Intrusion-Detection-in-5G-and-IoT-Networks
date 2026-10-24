"""Canonical rows -> model features, computed identically for both datasets.

Feature groups (choose with --features):
  volume  log1p of bytes / packets per direction and in total
  shape   bytes per packet, direction ratios, reply flag
  timing  log1p of duration, packet rate and byte rate. These depend on how each tool
          cuts flows (Argus status intervals vs Zeek connection timeouts) and were the
          three most shifted features between BoT-IoT and ToN-IoT, so ablate them.
  proto   one-hot tcp / udp / icmp / other
  ports   coarse port ranges + a few well-known service flags (tcp/udp only).
          Ports can act as dataset shortcuts, so compare with and without them.
  state   coarse connection state (established / no_reply / reset / other); a
          hypothesised mapping between Argus and Zeek taxonomies, off by default.

Raw port numbers, IP addresses, timestamps and ids are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import STATE_COARSE, STATE_LEVELS

DEFAULT_GROUPS = ("volume", "shape", "timing", "proto", "ports")
ALL_GROUPS = ("volume", "shape", "timing", "proto", "ports", "state")
PROTOS = ("tcp", "udp", "icmp", "other")
SERVICE_PORTS = {
    "http": (80, 8080, 8000, 8008, 8888),
    "https": (443, 8443),
    "dns": (53,),
    "ssh": (22,),
    "telnet": (23, 2323),
    "ftp": (20, 21),
    "mqtt": (1883, 8883),
    "smb": (139, 445),
}
RATE_EPS = 1e-3  # seconds; keeps rates finite for zero-duration flows


def _port_range(port: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "na": ~valid,
        "well_known": valid & (port < 1024),
        "registered": valid & (port >= 1024) & (port < 49152),
        "dynamic": valid & (port >= 49152),
    }


def build_features(df: pd.DataFrame, domain: str, groups=DEFAULT_GROUPS) -> tuple[np.ndarray, list[str]]:
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}; choose from {ALL_GROUPS}")
    sb = df["src_bytes"].to_numpy(dtype=np.float64)
    db = df["dst_bytes"].to_numpy(dtype=np.float64)
    sp = df["src_pkts"].to_numpy(dtype=np.float64)
    dp = df["dst_pkts"].to_numpy(dtype=np.float64)
    dur = df["duration"].to_numpy(dtype=np.float64)
    tb, tp = sb + db, sp + dp
    cols: dict[str, np.ndarray] = {}

    if "volume" in groups:
        cols.update({
            "log_src_bytes": np.log1p(sb), "log_dst_bytes": np.log1p(db),
            "log_src_pkts": np.log1p(sp), "log_dst_pkts": np.log1p(dp),
            "log_tot_bytes": np.log1p(tb), "log_tot_pkts": np.log1p(tp),
        })
    if "shape" in groups:
        cols.update({
            "log_src_bytes_per_pkt": np.log1p(sb / np.maximum(sp, 1)),
            "log_dst_bytes_per_pkt": np.log1p(db / np.maximum(dp, 1)),
            "src_byte_share": sb / np.maximum(tb, 1),
            "src_pkt_share": sp / np.maximum(tp, 1),
            "has_reply": (dp > 0).astype(np.float64),
        })
    if "timing" in groups:
        cols.update({
            "log_duration": np.log1p(dur),
            "log_pkt_rate": np.log1p(tp / (dur + RATE_EPS)),
            "log_byte_rate": np.log1p(tb / (dur + RATE_EPS)),
        })
    proto = df["proto"].astype(str).to_numpy()
    if "proto" in groups:
        for p in PROTOS:
            cols[f"proto_{p}"] = (proto == p).astype(np.float64)
    if "ports" in groups:
        l4 = np.isin(proto, ("tcp", "udp"))
        for side in ("src", "dst"):
            port = df[f"{side}_port"].to_numpy(dtype=np.float64)
            valid = l4 & np.isfinite(port)
            for name, mask in _port_range(port, valid).items():
                cols[f"{side}_port_{name}"] = mask.astype(np.float64)
        dport = df["dst_port"].to_numpy(dtype=np.float64)
        dvalid = l4 & np.isfinite(dport)
        for service, ports in SERVICE_PORTS.items():
            cols[f"dst_svc_{service}"] = (dvalid & np.isin(dport, ports)).astype(np.float64)
    if "state" in groups:
        coarse = df["state"].astype(str).str.lower().map(STATE_COARSE[domain]).fillna("other").to_numpy()
        for level in STATE_LEVELS:
            cols[f"state_{level}"] = (coarse == level).astype(np.float64)

    names = list(cols)
    X = np.column_stack([cols[n] for n in names]).astype(np.float32)
    return X, names


def continuous_mask(names: list[str]) -> np.ndarray:
    """True for continuous columns; one-hot / flag columns stay 0/1 (z-scoring an
    indicator per domain would give the same flow different values in each dataset)."""
    return np.array([n.startswith("log_") or n.endswith("_share") for n in names], dtype=bool)


class Standardizer:
    """z-scoring of the continuous columns, fitted on one domain's *training* rows
    (labels are never used, so fitting on the unlabeled target pool is legitimate)."""

    def __init__(self, continuous: np.ndarray) -> None:
        self.continuous = continuous
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        self.mean = np.where(self.continuous, mean, 0.0).astype(np.float32)
        self.std = np.where(self.continuous & (std > 1e-6), std, 1.0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray, clip: float = 10.0) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.clip(Z, -clip, clip).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "continuous": self.continuous.tolist()}
