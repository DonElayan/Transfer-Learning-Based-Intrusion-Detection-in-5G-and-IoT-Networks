#!/usr/bin/env python3
r"""BoT-IoT -> ToN-IoT transfer learning for intrusion detection - single-file version.

HOW TO RUN IN PYCHARM
  1. Install the packages once (PyCharm terminal, inside your project's interpreter):
         python -m pip install numpy pandas pyarrow scikit-learn torch matplotlib
  2. Check the SETTINGS block below. DATA_ROOT already points at your folder with the
     4 BoT-IoT files (UNSW_2018_IoT_Botnet_Full5pc_1..4.csv) and the 23 ToN-IoT files
     (Network_dataset_1..23.csv).
  3. First run: set QUICK_TRIAL = True (uses the first 100,000 rows of each file and short
     training, a few minutes) to confirm everything works. Then set it back to False.
  4. Right-click this file -> Run.

WHAT IT DOES
  prepare     streams the 27 CSVs into one shared flow format (proto, ports, duration,
              bytes, packets), fixes hex ports / -1 sentinels / Zeek '-' values, drops corrupt
              rows such as src_bytes='0.0.0.0', removes exact duplicates, writes two parquet
              files and prepare_report.txt
  diagnose    measures how different the two datasets are (domain classifier AUC, KS per feature)
  experiment  BoT-IoT is the source, ToN-IoT the target with only k labelled flows per class.
              Proposed model tl_match: BoT-IoT pre-trained network + consistency training on
              the unlabelled ToN flows. Compared with target_only (same network from scratch),
              fine-tuning, source_only and a random forest (for information), over label
              budgets and seeds, binary and multiclass. Writes tables, CSVs and plots.

OUTPUTS (inside <DATA_ROOT>\TL_BoT_ToN_pipeline\)
  prepared\prepare_report.txt            data-cleaning counts (read this first)
  results\diagnose\diagnose_report.txt   domain shift
  results\main\summary.txt               main result tables (mean +- std over seeds)
  results\main\learning_curve_*.png      figures
  results\main\results.csv, per_class.csv
  results\main_no_timing\...             one folder per ablation
  pipeline_log.txt                        everything printed to the console

This file bundles the modules of the bot_ton_tl/ package (same code, see bot_ton_tl/README.md
for the method and the reasoning behind every data decision).

COMMAND LINE (optional, overrides the settings)
  python bot_ton_tl_all_in_one.py --self-test     synthetic data, no real files needed
  python bot_ton_tl_all_in_one.py --quick         quick trial on the real files
  python bot_ton_tl_all_in_one.py --stages experiment --ablations no_timing no_ports
"""
from __future__ import annotations

# ================================ SETTINGS (edit here) ================================
DATA_ROOT = r"D:\Abdallah\Pycharm folder\pythonProject\ToN-IoT FULL"  # folder with all 27 CSVs
OUTPUT_DIR = None              # None -> <DATA_ROOT>\TL_BoT_ToN_pipeline
STAGES = ["prepare", "diagnose", "experiment"]
REUSE_PREPARED = True          # skip 'prepare' if it already finished with the same settings
QUICK_TRIAL = False            # True: first 100k rows per file, 1 seed, 2 budgets, short training
ABLATIONS = []                 # optional extra runs after the main one, any of:
                               #   "no_timing"      drop duration / packet-rate / byte-rate features
                               #   "no_ports"       drop port features
                               #   "unseen_classes" never label UNSEEN_CLASSES (zero-day detection)
                               #   "merged_classes" merge dos+ddos

SEEDS = [0, 1, 2]
SHOTS = [10, 50, 100, 500, 1000]       # labelled ToN-IoT flows per class
INCLUDE_FULL = True                    # also train on the whole (capped) ToN pool = upper bound
TASKS = ["binary", "multiclass"]
# tl_match = PROPOSED model (BoT-IoT transfer + consistency training on unlabelled ToN flows)
# match_scratch = tl_match without BoT-IoT (shows what the transfer adds)
# target_only = same network from scratch (the reference); rf = random forest (for information)
# also available: dann_zero, frozen, finetune, dann, dann_full
METHODS = ["source_only", "target_only", "rf", "finetune_full", "tl_match", "match_scratch"]
ARCH = "mlp"                           # "mlp" or "cnn1d" (1-D version of the repo's CNN-B)
FEATURES = ["volume", "shape", "timing", "proto", "ports"]   # add "state" for the coarse-state ablation
DEVICE = "auto"                        # "auto", "cpu" or "cuda"

TON_BYTES = "ip"         # ToN columns paired with BoT sbytes/dbytes: "ip" (src_ip_bytes) or "payload"
BOT_L2_BYTES = 14        # Ethernet header bytes removed per BoT-IoT packet (0 = keep raw Argus bytes)
CHUNK_SIZE = 250_000     # rows per read in 'prepare'; use 100_000 if memory is short

# used by the ablations
UNSEEN_CLASSES = ["ransomware", "mitm", "backdoor"]  # never labelled -> zero-day detection rates
MERGE_CLASSES = ["dos+ddos"]                          # single flows cannot separate DoS from DDoS
# ======================================================================================

import argparse
import copy
import json
import math
import platform
import re
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from torch import nn
except ImportError as _exc:
    sys.exit(f"Missing Python package '{_exc.name}'. Install the requirements with:\n"
             f"    python -m pip install numpy pandas pyarrow scikit-learn torch matplotlib")

# ==================================================================================================
# SCHEMA
#   Dataset schemas and the canonical (shared) flow representation.
#
#   Only fields that exist with comparable semantics in BOTH datasets are kept:
#
#       canonical        BoT-IoT (Argus)   ToN-IoT (Zeek)            notes
#       ---------        ---------------   --------------            -----
#       proto            proto             proto                     tcp/udp/icmp/other
#       src_port         sport             src_port                  hex decoded, -1 -> missing
#       dst_port         dport             dst_port                  hex decoded, -1 -> missing
#       duration         dur               duration                  seconds in both
#       src_bytes        sbytes - 14*spkts src_ip_bytes  (default)   IP-level bytes incl. headers
#       dst_bytes        dbytes - 14*dpkts dst_ip_bytes  (default)   IP-level bytes incl. headers
#       src_pkts         spkts             src_pkts
#       dst_pkts         dpkts             dst_pkts
#       state            state             conn_state                different taxonomies,
#                                                                    only a coarse mapping
#
#   Why src_ip_bytes and not src_bytes: Zeek's ``orig_bytes`` (ToN ``src_bytes``) counts
#   *payload* bytes only and is 0 for SYN scans / floods, while Argus ``sbytes`` counts the
#   bytes of every packet including headers and is never 0 when ``spkts > 0``. Zeek's
#   ``orig_ip_bytes`` (ToN ``src_ip_bytes``) is the header-inclusive counterpart. The
#   prepare report prints the evidence (rows with packets > 0 but bytes == 0) so the choice
#   can be checked on the real files; ``--ton-bytes payload`` switches back.
#
#   Identifiers, addresses, timestamps, BoT-IoT's window-aggregate columns (TnBPSrcIP,
#   N_IN_Conn_P_DstIP, ...), the *_number encodings and all labels are never features.
# ==================================================================================================

BOT = "bot"
TON = "ton"
DOMAINS = (BOT, TON)

CANONICAL_NUMERIC = [
    "src_port", "dst_port", "duration",
    "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
]
# Columns that define a flow's feature vector (used for de-duplication / group splits).
VECTOR_COLUMNS = ["proto"] + CANONICAL_NUMERIC


def normalize_col(name: object) -> str:
    """'src_ip_bytes' -> 'srcipbytes', 'TnP_PSrcIP' -> 'tnppsrcip'."""
    return re.sub(r"[^a-z0-9]+", "", str(name).replace("\ufeff", "").strip().lower())


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file_pattern: str
    file_numbers: tuple[int, ...]
    expected_rows: dict[int, int]
    # canonical name -> normalized raw header name
    columns: dict[str, str]
    # canonical numeric columns where Zeek's "-" (unset) means 0 rather than invalid
    dash_means_zero: tuple[str, ...] = ()
    extra_columns: dict[str, str] = field(default_factory=dict)
    optional: tuple[str, ...] = ()
    # bytes per packet removed from the byte counts (link-layer header, see BOT_SPEC)
    l2_bytes_per_pkt: int = 0

    def file_name(self, number: int) -> str:
        return self.file_pattern.format(number)


BOT_SPEC = DatasetSpec(
    name=BOT,
    file_pattern="UNSW_2018_IoT_Botnet_Full5pc_{}.csv",
    file_numbers=(1, 2, 3, 4),
    expected_rows={1: 1_000_000, 2: 1_000_000, 3: 1_000_000, 4: 668_522},
    columns={
        "proto": "proto",
        "src_port": "sport",
        "dst_port": "dport",
        "duration": "dur",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "src_pkts": "spkts",
        "dst_pkts": "dpkts",
        "state": "state",
        "binary": "attack",
        "cls": "category",
        "subcls": "subcategory",
    },
    optional=("subcls",),
    # Argus byte counts are whole Ethernet frames: in the repository's BoT-IoT.csv the
    # smallest bytes/packet is 42 = 14 (Ethernet) + 20 (IP) + 8 (UDP), and a lone TCP RST
    # is 60 bytes (padded minimum frame). Zeek's *_ip_bytes start at the IP header, so
    # 14 bytes per packet are removed by default (--bot-l2-bytes 0 keeps the raw values).
    l2_bytes_per_pkt=14,
)


def ton_spec(ton_bytes: str = "ip") -> DatasetSpec:
    if ton_bytes not in ("ip", "payload"):
        raise ValueError("ton_bytes must be 'ip' or 'payload'")
    src_b, dst_b = ("srcipbytes", "dstipbytes") if ton_bytes == "ip" else ("srcbytes", "dstbytes")
    other_src, other_dst = ("srcbytes", "dstbytes") if ton_bytes == "ip" else ("srcipbytes", "dstipbytes")
    return DatasetSpec(
        name=TON,
        file_pattern="Network_dataset_{}.csv",
        file_numbers=tuple(range(1, 24)),
        expected_rows={**{i: 1_000_000 for i in range(1, 23)}, 23: 339_021},
        columns={
            "proto": "proto",
            "src_port": "srcport",
            "dst_port": "dstport",
            "duration": "duration",
            "src_bytes": src_b,
            "dst_bytes": dst_b,
            "src_pkts": "srcpkts",
            "dst_pkts": "dstpkts",
            "state": "connstate",
            "binary": "label",
            "cls": "type",
        },
        # Zeek writes "-" for an unset duration / payload byte count (e.g. one packet).
        dash_means_zero=("duration", "src_bytes", "dst_bytes", "src_alt_bytes", "dst_alt_bytes"),
        # The other byte definition is parsed too: an unparseable value there (the
        # known src_bytes='0.0.0.0' rows) marks a corrupt record, and it feeds the
        # byte-semantics evidence in the prepare report.
        extra_columns={"src_alt_bytes": other_src, "dst_alt_bytes": other_dst},
    )


def get_spec(domain: str, ton_bytes: str = "ip", bot_l2_bytes: int = 14) -> DatasetSpec:
    if domain == BOT:
        return replace(BOT_SPEC, l2_bytes_per_pkt=bot_l2_bytes)
    return ton_spec(ton_bytes)


def canonical_proto(raw: str) -> str:
    p = raw.strip().lower()
    if p in ("tcp", "udp"):
        return p
    if p in ("icmp", "ipv6-icmp", "icmp6"):
        return "icmp"
    return "other"


def canonical_class(raw: str) -> str:
    return raw.strip().lower()


# Coarse connection-state abstraction. This is a hypothesis, not an established
# equivalence: it is only used when the 'state' feature group is switched on.
#   Argus: http://qosient.com/argus (ra -s state), Zeek: conn.log conn_state.
STATE_COARSE = {
    BOT: {
        "con": "established", "fin": "established", "acc": "established", "clo": "established",
        "int": "no_reply", "req": "no_reply",
        "rst": "reset", "urp": "reset",  # urp = ICMP port unreachable, i.e. rejected
    },
    TON: {
        "sf": "established", "s1": "established", "s2": "established", "s3": "established",
        "s0": "no_reply", "sh": "no_reply", "shr": "no_reply", "oth": "no_reply",
        "rej": "reset", "rsto": "reset", "rstr": "reset", "rstos0": "reset", "rstrh": "reset",
    },
}
STATE_LEVELS = ("established", "no_reply", "reset", "other")

# Label names shared verbatim by both taxonomies plus one documented hypothesis
# (BoT 'reconnaissance' ~ ToN 'scanning'). Only used for source-only multiclass
# reporting; fine-tuning always learns the full ToN-IoT label set with a new head.
SHARED_CLASS_MAP = {
    BOT: {"normal": "normal", "dos": "dos", "ddos": "ddos", "reconnaissance": "scanning"},
    TON: {"normal": "normal", "dos": "dos", "ddos": "ddos", "scanning": "scanning"},
}


# ==================================================================================================
# PARSING
#   Vectorised, policy-explicit parsing of raw CSV tokens.
#
#   Every parser returns the parsed float64 values *and* boolean masks describing what was
#   seen, so the caller decides (and reports) what happens to blanks, Zeek "-" values,
#   hexadecimal ports, sentinels and invalid tokens such as ToN-IoT's src_bytes='0.0.0.0'.
# ==================================================================================================

HEX_RE = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


@dataclass
class Parsed:
    values: np.ndarray       # float64, NaN where not a usable number
    blank: np.ndarray        # empty / nan-like token
    dash: np.ndarray         # Zeek's "-" (unset)
    hexadecimal: np.ndarray  # token like 0x0050 (decoded into values)
    invalid: np.ndarray      # non-empty token that is not a finite number
    tokens: pd.Series        # stripped raw tokens (for reporting invalid examples)


def parse_numeric(raw: pd.Series, *, allow_hex: bool = False, dash_value: float | None = None) -> Parsed:
    """Parse a column of raw string tokens.

    allow_hex:  decode tokens such as '0x0303' as base-16 integers (BoT-IoT ports).
    dash_value: value assigned to '-' tokens; None leaves them NaN (reported as dash).
    """
    tokens = raw.astype(str).str.strip()
    lowered = tokens.str.lower()
    blank = lowered.isin(BLANK_TOKENS).to_numpy()
    dash = tokens.eq("-").to_numpy()
    if allow_hex:
        hexadecimal = tokens.str.fullmatch(HEX_RE.pattern).fillna(False).to_numpy(dtype=bool)
    else:
        hexadecimal = np.zeros(len(tokens), dtype=bool)

    plain = ~(blank | dash | hexadecimal)
    values = np.full(len(tokens), np.nan, dtype=np.float64)
    if plain.any():
        values[plain] = pd.to_numeric(tokens[plain], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    if hexadecimal.any():
        hex_tokens = tokens[hexadecimal]
        lookup = {t: float(int(t, 16)) for t in hex_tokens.unique()}
        values[hexadecimal] = hex_tokens.map(lookup).to_numpy(dtype=np.float64)
    if dash_value is not None and dash.any():
        values[dash] = dash_value

    nonfinite = ~np.isfinite(values) & ~np.isnan(values)
    values[nonfinite] = np.nan
    invalid = plain & np.isnan(values)
    return Parsed(values, blank, dash, hexadecimal, invalid, tokens)


def parse_port(raw: pd.Series) -> tuple[np.ndarray, dict[str, int], pd.Series]:
    """Ports: hex decoded; -1 / out-of-range / blank / '-' / invalid become NaN.

    Returns (values, counts, invalid_tokens) where counts has keys
    hex, sentinel_minus1, out_of_range, missing, invalid.
    """
    p = parse_numeric(raw, allow_hex=True)
    v = p.values
    minus1 = v == -1
    out_of_range = np.isfinite(v) & ~minus1 & ((v < 0) | (v > 65535) | (v != np.floor(v)))
    v = v.copy()
    v[minus1 | out_of_range] = np.nan
    counts = {
        "hex": int(p.hexadecimal.sum()),
        "sentinel_minus1": int(minus1.sum()),
        "out_of_range": int(out_of_range.sum()),
        "missing": int((p.blank | p.dash).sum()),
        "invalid": int(p.invalid.sum()),
    }
    return v, counts, p.tokens[p.invalid]


def parse_binary_label(raw: pd.Series) -> np.ndarray:
    """'0'/'1' (also '0.0'/'1.0') -> 0/1, anything else -> -1."""
    v = pd.to_numeric(raw.astype(str).str.strip(), errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    out = np.full(len(v), -1, dtype=np.int8)
    out[v == 0] = 0
    out[v == 1] = 1
    return out


# ==================================================================================================
# FEATURES
#   Canonical rows -> model features, computed identically for both datasets.
#
#   Feature groups (choose with --features):
#     volume  log1p of bytes / packets per direction and in total
#     shape   bytes per packet, direction ratios, reply flag
#     timing  log1p of duration, packet rate and byte rate. These depend on how each tool
#             cuts flows (Argus status intervals vs Zeek connection timeouts) and were the
#             three most shifted features between BoT-IoT and ToN-IoT, so ablate them.
#     proto   one-hot tcp / udp / icmp / other
#     ports   coarse port ranges + a few well-known service flags (tcp/udp only).
#             Ports can act as dataset shortcuts, so compare with and without them.
#     state   coarse connection state (established / no_reply / reset / other); a
#             hypothesised mapping between Argus and Zeek taxonomies, off by default.
#
#   Raw port numbers, IP addresses, timestamps and ids are never used.
# ==================================================================================================

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


# ==================================================================================================
# DATA
#   Loading prepared parquet, leakage-safe splits and label-budget sampling.
#
#   Splits are assigned per *feature-vector hash*: every row with the same canonical vector
#   lands in the same split, so exact duplicates and label-conflicting twins can never sit on
#   both sides of a train/test boundary (the leakage the Phase 2.2 audit warned about).
# ==================================================================================================

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


# ==================================================================================================
# MODELS
#   Small encoders suitable for constrained IoT/edge devices, plus the DANN pieces.
# ==================================================================================================

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 128), out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in (*hidden, out_dim):
            # LayerNorm instead of BatchNorm: stable with tiny few-shot batches and it
            # does not carry source-domain running statistics into the target domain.
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DEncoder(nn.Module):
    """1-D counterpart of the repository's CNN-B (conv -> pool -> conv -> pool -> dense)."""

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True),
        )
        flat = 64 * ((in_dim + 3) // 4)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x.unsqueeze(1)))


def make_encoder(arch: str, in_dim: int) -> nn.Module:
    if arch == "mlp":
        return MLPEncoder(in_dim)
    if arch == "cnn1d":
        return CNN1DEncoder(in_dim)
    raise ValueError(f"unknown arch {arch!r} (mlp | cnn1d)")


class Net(nn.Module):
    """encoder + task head. head='linear' or 'mlp' (the repo's CNN-TL used a dense stack)."""

    def __init__(self, encoder: nn.Module, n_classes: int, head: str = "linear"):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        if head == "linear":
            self.head = nn.Linear(d, n_classes)
        else:
            self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(grad_reverse(z, alpha)).squeeze(-1)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ==================================================================================================
# METRICS
#   Evaluation metrics. Macro scores are the headline numbers: accuracy is dominated by
#   the majority attack classes in both datasets and hides failures on normal traffic.
# ==================================================================================================

def binary_metrics(y_true: np.ndarray, p_attack: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p_attack >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_dr": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    normal = y_true == 0
    out["far_fpr"] = float((y_pred[normal] == 1).mean()) if normal.any() else float("nan")
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, p_attack)
        out["pr_auc"] = average_precision_score(y_true, p_attack)
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_rates(cls_true: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Share of each original class handled correctly (detection rate for attack
    classes in the binary task, recall in the multiclass task)."""
    return {str(c): float(correct[cls_true == c].mean()) for c in np.unique(cls_true)}


# ==================================================================================================
# TRAIN
#   Training loops: supervised (source pre-training / target-only), linear-probe-then-
#   fine-tune (LP-FT), domain-adversarial pre-training (DANN, Ganin et al. 2016) and the
#   proposed transfer + consistency training on unlabelled target flows (train_consistency).
#
#   Training is step-based with mini-batches drawn with replacement, so the same loop
#   serves 30-sample few-shot budgets and 500k-row pre-training. No torch DataLoader is
#   used (no worker processes - safe on Windows).
# ==================================================================================================

@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 100
    patience: int = 6          # evaluations without improvement before stopping
    balanced: bool = True      # class-balanced batches (essential for BoT-IoT: 477 normal rows)
    device: str = "cpu"


class BatchSampler:
    def __init__(self, y: np.ndarray, batch_size: int, balanced: bool, rng: np.random.Generator):
        self.y, self.batch_size, self.balanced, self.rng = y, batch_size, balanced, rng
        self.members = [np.flatnonzero(y == c) for c in np.unique(y)]

    def __call__(self) -> np.ndarray:
        n = len(self.y)
        if not self.balanced or len(self.members) < 2:
            return self.rng.integers(0, n, size=min(self.batch_size, max(n, 1)))
        which = self.rng.integers(0, len(self.members), size=self.batch_size)
        idx = np.empty(self.batch_size, dtype=np.int64)
        for k, members in enumerate(self.members):
            mask = which == k
            if mask.any():
                idx[mask] = members[self.rng.integers(0, len(members), size=int(mask.sum()))]
        return idx


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu", batch: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def macro_f1(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str) -> float:
    pred = predict_proba(model, X, device).argmax(1)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def fit(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: TrainConfig, rng: np.random.Generator,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
        param_groups: list[dict] | None = None) -> dict:
    """Supervised training with early stopping on validation macro-F1 (if given)."""
    device = cfg.device
    model.to(device)
    params = param_groups if param_groups is not None else [{"params": [p for p in model.parameters() if p.requires_grad]}]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    sampler = BatchSampler(y, cfg.batch_size, cfg.balanced, rng)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    has_val = X_val is not None and len(X_val) > 0
    best_score, best_state, bad, step = -math.inf, None, 0, 0
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.from_numpy(sampler()).to(device)
        loss = loss_fn(model(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if has_val and (step % cfg.eval_every == 0 or step == cfg.steps):
            score = macro_f1(model, X_val, y_val, device)
            if score > best_score + 1e-4:
                best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"steps": step, "val_macro_f1": best_score if has_val else float("nan")}


def train_supervised(arch: str, X: np.ndarray, y: np.ndarray, n_classes: int, cfg: TrainConfig, seed: int,
                     X_val=None, y_val=None, head: str = "linear",
                     encoder_state: dict | None = None) -> tuple[Net, dict]:
    """encoder_state=None trains from scratch; otherwise the encoder starts from the given
    pre-trained weights and is trained exactly like the from-scratch model."""
    torch.manual_seed(seed)
    model = Net(make_encoder(arch, X.shape[1]), n_classes, head=head)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    info = fit(model, X, y, cfg, np.random.default_rng(seed), X_val, y_val)
    return model, info


def train_consistency(encoder_state: dict | None, arch: str, X: np.ndarray, y: np.ndarray, n_classes: int,
                      X_pool: np.ndarray, cfg: TrainConfig, seed: int, X_val=None, y_val=None,
                      threshold: float = 0.9, corrupt: float = 0.3, unlabeled_weight: float = 1.0,
                      unlabeled_ratio: int = 4) -> tuple[Net, dict]:
    """FixMatch for tabular flows (Sohn et al. 2020; corruption as in SCARF, Bahri et al. 2022).

    Every step: a class-balanced labelled batch (cross-entropy) plus `unlabeled_ratio` x as many
    unlabelled target flows. The model's prediction on the clean flow becomes a pseudo-label
    when its confidence is >= `threshold`; the model is then trained to predict that label for
    a corrupted copy where a `corrupt` share of the features is replaced by the same feature of
    random other pool flows. Starts from the given (BoT-IoT) encoder when encoder_state is set.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    model = Net(make_encoder(arch, X.shape[1]), n_classes).to(device)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    sampler = BatchSampler(y, cfg.batch_size, cfg.balanced, rng)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y.astype(np.int64)).to(device)
    Xp = torch.from_numpy(X_pool).to(device)
    nu = cfg.batch_size * unlabeled_ratio
    has_val = X_val is not None and len(X_val) > 0
    best_score, best_state, bad, step, used = -math.inf, None, 0, 0, 0.0
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.from_numpy(sampler()).to(device)
        loss = ce(model(Xt[idx]), yt[idx])
        u = Xp[torch.from_numpy(rng.integers(0, len(X_pool), size=nu)).to(device)]
        with torch.no_grad():
            model.eval()
            probs = torch.softmax(model(u), dim=1)
            model.train()
        conf, pseudo = probs.max(1)
        mask = conf >= threshold
        if mask.any():
            donors = Xp[torch.from_numpy(rng.integers(0, len(X_pool), size=nu)).to(device)]
            swap = torch.rand_like(u) < corrupt
            strong = torch.where(swap, donors, u)
            loss = loss + unlabeled_weight * ce(model(strong[mask]), pseudo[mask]) * mask.float().mean()
            used += float(mask.float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if has_val and (step % cfg.eval_every == 0 or step == cfg.steps):
            score = macro_f1(model, X_val, y_val, device)
            if score > best_score + 1e-4:
                best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"steps": step, "val_macro_f1": best_score if has_val else float("nan"),
                   "pseudo_share": round(used / max(step, 1), 3)}


def finetune(encoder_state: dict, arch: str, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int,
             cfg_lp: TrainConfig, cfg_ft: TrainConfig | None, X_val=None, y_val=None,
             head: str = "linear", encoder_lr_mult: float = 0.1) -> tuple[Net, dict]:
    """LP-FT (Kumar et al., ICLR 2022): train a new head on the frozen pre-trained encoder,
    then (if cfg_ft) unfreeze everything with a smaller encoder learning rate. cfg_ft=None
    gives the frozen-feature-extractor variant the repository's CNN-TL notebook used."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(make_encoder(arch, X.shape[1]), n_classes, head=head)
    model.encoder.load_state_dict(encoder_state)
    for p in model.encoder.parameters():
        p.requires_grad = False
    info_lp = fit(model, X, y, cfg_lp, rng, X_val, y_val)
    info = {"lp_steps": info_lp["steps"], "val_macro_f1": info_lp["val_macro_f1"]}
    if cfg_ft is not None:
        for p in model.encoder.parameters():
            p.requires_grad = True
        groups = [{"params": model.head.parameters(), "lr": cfg_ft.lr},
                  {"params": model.encoder.parameters(), "lr": cfg_ft.lr * encoder_lr_mult}]
        info_ft = fit(model, X, y, cfg_ft, rng, X_val, y_val, param_groups=groups)
        info.update(ft_steps=info_ft["steps"], val_macro_f1=info_ft["val_macro_f1"])
    return model, info


def train_dann(arch: str, Xs: np.ndarray, ys: np.ndarray, n_classes: int, Xt: np.ndarray, cfg: TrainConfig,
               seed: int, lambda_max: float = 1.0) -> tuple[Net, dict]:
    """Source-supervised training with a gradient-reversed domain discriminator on
    unlabeled target flows. Labels of the target domain are never touched. The final
    model is kept (no target labels exist for model selection in this stage)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    model = Net(make_encoder(arch, Xs.shape[1]), n_classes).to(device)
    disc = DomainDiscriminator(model.encoder.out_dim).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    src_sampler = BatchSampler(ys, cfg.batch_size, cfg.balanced, rng)
    Xs_t = torch.from_numpy(Xs).to(device)
    ys_t = torch.from_numpy(ys.astype(np.int64)).to(device)
    Xt_t = torch.from_numpy(Xt).to(device)
    last_dom_acc = float("nan")
    for step in range(1, cfg.steps + 1):
        model.train()
        disc.train()
        p = step / cfg.steps
        alpha = lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
        si = torch.from_numpy(src_sampler()).to(device)
        ti = torch.from_numpy(rng.integers(0, len(Xt), size=cfg.batch_size)).to(device)
        zs = model.encoder(Xs_t[si])
        zt = model.encoder(Xt_t[ti])
        cls_loss = ce(model.head(zs), ys_t[si])
        dom_labels = torch.cat([torch.zeros(len(si)), torch.ones(len(ti))]).to(device)
        dom_logits = disc(torch.cat([zs, zt]), alpha)
        dom_loss = bce(dom_logits, dom_labels)
        loss = cls_loss + dom_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            last_dom_acc = float(((dom_logits > 0).float() == dom_labels).float().mean())
    # domain accuracy near 0.5 means the encoder no longer separates the datasets
    return model, {"steps": cfg.steps, "final_domain_acc": last_dom_acc}


def scaled(cfg: TrainConfig, **changes) -> TrainConfig:
    return replace(cfg, **changes)


# ==================================================================================================
# PREPARE
#   Stage 1: stream the raw CSVs into one harmonised, de-duplicated parquet per dataset.
#
#       python -m bot_ton_tl.prepare --root "D:\Abdallah\Pycharm folder\pythonProject\ToN-IoT FULL" --out prepared
#
#   Reads only the columns in schema.py (so ToN-IoT Network_dataset_6.csv's extra 'uid'
#   column is irrelevant), applies the parsing policy below, removes exact duplicate
#   (feature-vector, label) rows and writes:
#
#       <out>/bot.parquet, <out>/ton.parquet   canonical rows + vec_hash (for group splits)
#       <out>/prepare_report.json / .txt       every count needed for a methods section
#
#   Row policy (each case is counted in the report):
#     * core numeric (duration, bytes, packets) blank / unparseable / negative -> row dropped
#       (this is what removes the ToN-IoT src_bytes='0.0.0.0' records)
#     * Zeek '-' in duration / byte columns -> 0 (Zeek leaves them unset for one-packet flows)
#     * ports: hexadecimal decoded, -1 / out of range / unparseable -> missing (row kept)
#     * binary label not 0/1, blank class, or binary label contradicting the class
#       (normal <-> 0) -> row dropped
# ==================================================================================================

CORE_NUMERIC = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
MAX_TOKEN_EXAMPLES = 25


class DomainStats:
    """Everything that ends up in the prepare report for one dataset."""

    def __init__(self) -> None:
        self.files: list[dict] = []
        self.rows_read = 0
        self.rows_kept = 0
        self.dropped: Counter = Counter()
        self.invalid_tokens: dict[str, Counter] = defaultdict(Counter)
        self.port_counts: dict[str, Counter] = defaultdict(Counter)
        self.dash_to_zero: Counter = Counter()
        self.class_counts_raw: Counter = Counter()
        self.class_counts_kept: Counter = Counter()
        self.proto_counts: Counter = Counter()
        self.state_counts: Counter = Counter()
        self.bytes_evidence: Counter = Counter()
        self.single_pkt_tcp_src_bytes: Counter = Counter()
        self.dedup: dict = {}

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "rows_read": self.rows_read,
            "rows_after_row_policy": self.rows_kept,
            "dropped_by_reason": dict(self.dropped),
            "invalid_tokens_top": {c: cnt.most_common(MAX_TOKEN_EXAMPLES) for c, cnt in self.invalid_tokens.items()},
            "port_parsing": {c: dict(cnt) for c, cnt in self.port_counts.items()},
            "dash_to_zero": {k: v for k, v in self.dash_to_zero.items() if v},
            "class_counts_raw": dict(self.class_counts_raw.most_common()),
            "class_counts_after_row_policy": dict(self.class_counts_kept.most_common()),
            "proto_counts": dict(self.proto_counts.most_common()),
            "state_counts_top": self.state_counts.most_common(30),
            "bytes_evidence": dict(self.bytes_evidence),
            "single_packet_tcp_src_bytes_top": self.single_pkt_tcp_src_bytes.most_common(10),
            "dedup": self.dedup,
        }


def resolve_columns(path: Path, spec: DatasetSpec) -> dict[str, str]:
    """canonical name -> raw header string as it appears in this file."""
    header = pd.read_csv(path, nrows=0, dtype=str, encoding="utf-8-sig", encoding_errors="replace")
    by_norm: dict[str, str] = {}
    for raw in header.columns:
        by_norm.setdefault(normalize_col(raw), raw)
    wanted = {**spec.columns, **spec.extra_columns}
    missing = sorted(c for c, n in wanted.items() if n not in by_norm and c not in spec.optional)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing} "
                         f"(looked for {[wanted[c] for c in missing]}; header has {sorted(by_norm)})")
    return {c: by_norm[n] for c, n in wanted.items() if n in by_norm}


def harmonize_chunk(raw: pd.DataFrame, spec: DatasetSpec, colmap: dict[str, str], file_number: int,
                    stats: DomainStats) -> pd.DataFrame:
    n = len(raw)
    drop = np.zeros(n, dtype=bool)
    out: dict[str, np.ndarray] = {}

    numeric_cols = CORE_NUMERIC + list(spec.extra_columns)
    for col in numeric_cols:
        zero_dash = col in spec.dash_means_zero
        p = parse_numeric(raw[colmap[col]], dash_value=0.0 if zero_dash else None)
        if zero_dash:
            stats.dash_to_zero[col] += int(p.dash.sum())
        bad_missing = p.blank | (p.dash & (not zero_dash))
        bad_negative = np.isfinite(p.values) & (p.values < 0)
        for reason, mask in (("missing", bad_missing), ("invalid", p.invalid), ("negative", bad_negative)):
            k = int((mask & ~drop).sum())
            if k:
                stats.dropped[f"{reason}_numeric:{col}"] += k
        if p.invalid.any():
            stats.invalid_tokens[col].update(p.tokens[p.invalid].str.slice(0, 60).value_counts().to_dict())
        drop |= bad_missing | p.invalid | bad_negative
        out[col] = p.values

    for col in ("src_port", "dst_port"):
        values, counts, bad_tokens = parse_port(raw[colmap[col]])
        stats.port_counts[col].update(counts)
        if len(bad_tokens):
            stats.invalid_tokens[col].update(bad_tokens.str.slice(0, 60).value_counts().to_dict())
        out[col] = values

    proto_raw = raw[colmap["proto"]].astype(str)
    out["proto"] = proto_raw.map(canonical_proto).to_numpy(dtype=object)
    out["state"] = raw[colmap["state"]].astype(str).str.strip().str.lower().to_numpy(dtype=object)

    cls = raw[colmap["cls"]].astype(str).map(canonical_class)
    stats.class_counts_raw.update(cls.value_counts().to_dict())
    binary = parse_binary_label(raw[colmap["binary"]])
    blank_cls = cls.eq("").to_numpy()
    bad_binary = binary < 0
    expected_binary = (cls.to_numpy(dtype=object) != "normal").astype(np.int8)
    mismatch = ~bad_binary & ~blank_cls & (binary != expected_binary)
    for reason, mask in (("blank_class", blank_cls), ("bad_binary_label", bad_binary),
                         ("binary_contradicts_class", mismatch)):
        k = int((mask & ~drop).sum())
        if k:
            stats.dropped[reason] += k
    drop |= blank_cls | bad_binary | mismatch
    out["cls"] = cls.to_numpy(dtype=object)
    out["binary"] = binary
    if "subcls" in colmap:
        out["subcls"] = raw[colmap["subcls"]].astype(str).map(canonical_class).to_numpy(dtype=object)
    else:
        out["subcls"] = out["cls"]

    frame = pd.DataFrame(out)
    frame["file_id"] = np.int16(file_number)
    frame = frame.loc[~drop].reset_index(drop=True)

    # Evidence for the byte-definition choice (see schema.py), on the raw values.
    has_src = frame["src_pkts"] > 0
    stats.bytes_evidence["rows_with_src_pkts_gt0"] += int(has_src.sum())
    stats.bytes_evidence["of_which_src_bytes_eq0"] += int((has_src & (frame["src_bytes"] == 0)).sum())
    if has_src.any():
        per_pkt = float((frame.loc[has_src, "src_bytes"] / frame.loc[has_src, "src_pkts"]).min())
        stats.bytes_evidence["min_src_bytes_per_pkt"] = min(stats.bytes_evidence.get("min_src_bytes_per_pkt", per_pkt), per_pkt)
    if "src_alt_bytes" in frame:
        stats.bytes_evidence["of_which_src_alt_bytes_eq0"] += int((has_src & (frame["src_alt_bytes"] == 0)).sum())
    single = (frame["proto"] == "tcp") & (frame["src_pkts"] == 1) & (frame["dst_pkts"] == 0)
    stats.single_pkt_tcp_src_bytes.update(frame.loc[single, "src_bytes"].value_counts().head(20).to_dict())
    if spec.l2_bytes_per_pkt:
        for side in ("src", "dst"):
            frame[f"{side}_bytes"] = np.maximum(frame[f"{side}_bytes"] - spec.l2_bytes_per_pkt * frame[f"{side}_pkts"], 0.0)

    stats.class_counts_kept.update(frame["cls"].value_counts().to_dict())
    stats.proto_counts.update(frame["proto"].value_counts().to_dict())
    stats.state_counts.update(frame["state"].value_counts().to_dict())

    frame["vec_hash"] = pd.util.hash_pandas_object(frame[VECTOR_COLUMNS], index=False).to_numpy(dtype=np.uint64)
    frame["row_hash"] = pd.util.hash_pandas_object(frame[VECTOR_COLUMNS + ["cls"]], index=False).to_numpy(dtype=np.uint64)
    for col in CANONICAL_NUMERIC + list(spec.extra_columns):
        frame[col] = frame[col].astype(np.float64)
    return frame


def prepare_domain(domain: str, root: Path, out_dir: Path, *, ton_bytes: str, bot_l2_bytes: int, chunk_size: int,
                   max_rows_per_file: int | None, allow_missing_files: bool, dedup: bool,
                   files: list[Path] | None = None) -> DomainStats:
    spec = get_spec(domain, ton_bytes, bot_l2_bytes)
    stats = DomainStats()
    parts_dir = out_dir / f"_{domain}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    if files:  # explicit file list (another release of the dataset); numbered 1..n
        plan = [(i, Path(f)) for i, f in enumerate(files, start=1)]
        spec = replace(spec, expected_rows={})
    else:
        plan = [(number, root / spec.file_name(number)) for number in spec.file_numbers]

    for number, path in plan:
        if not path.is_file():
            if allow_missing_files:
                stats.files.append({"file": path.name, "status": "MISSING"})
                print(f"  [skip] missing {path.name}", flush=True)
                continue
            raise FileNotFoundError(f"{path} not found (use --allow-missing-files to skip)")
        t0 = time.time()
        colmap = resolve_columns(path, spec)
        part = parts_dir / f"part_{number:03d}.parquet"
        writer = None
        rows_read = rows_kept = 0
        reader = pd.read_csv(path, usecols=list(dict.fromkeys(colmap.values())), dtype=str,
                             keep_default_na=False, na_filter=False, chunksize=chunk_size,
                             nrows=max_rows_per_file, encoding="utf-8-sig", encoding_errors="replace",
                             low_memory=False)
        for raw in reader:
            rows_read += len(raw)
            frame = harmonize_chunk(raw, spec, colmap, number, stats)
            rows_kept += len(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(part, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
            parts.append(part)
        expected = spec.expected_rows.get(number)
        if max_rows_per_file:
            status = f"PARTIAL(first {max_rows_per_file:,} rows)"
        elif expected is None:
            status = "OK(no reference count)"
        else:
            status = "OK" if rows_read == expected else f"ROW_COUNT_DIFFERS(expected {expected:,})"
        stats.files.append({"file": path.name, "rows_read": rows_read, "rows_kept": rows_kept,
                            "expected_rows": expected, "status": status, "seconds": round(time.time() - t0, 1)})
        stats.rows_read += rows_read
        stats.rows_kept += rows_kept
        print(f"  {path.name}: read {rows_read:,} kept {rows_kept:,} [{status}] {time.time() - t0:.0f}s", flush=True)

    if not parts:
        raise RuntimeError(f"no {domain} files were processed under {root}")
    finalize_domain(parts, out_dir / f"{domain}.parquet", stats, dedup=dedup)
    for part in parts:
        part.unlink()
    parts_dir.rmdir()
    return stats


def finalize_domain(parts: list[Path], final_path: Path, stats: DomainStats, *, dedup: bool) -> None:
    """Exact de-duplication on (feature vector, class) across all files of a dataset."""
    row_hashes, vec_hashes, class_codes, sizes = [], [], [], []
    class_names: dict[str, int] = {}
    for part in parts:  # one part at a time keeps peak memory at ~20 bytes per row
        t = pq.read_table(part, columns=["row_hash", "vec_hash", "cls"])
        sizes.append(t.num_rows)
        row_hashes.append(t.column("row_hash").to_numpy())
        vec_hashes.append(t.column("vec_hash").to_numpy())
        enc = pc.dictionary_encode(t.column("cls").combine_chunks())
        lookup = np.array([class_names.setdefault(v, len(class_names)) for v in enc.dictionary.to_pylist()],
                          dtype=np.int16)
        class_codes.append(lookup[enc.indices.to_numpy()] if len(lookup) else np.zeros(0, dtype=np.int16))
        del t
    row_hash = np.concatenate(row_hashes)
    vec_hash = np.concatenate(vec_hashes)
    cls = pd.Categorical.from_codes(np.concatenate(class_codes), categories=list(class_names))
    del row_hashes, vec_hashes, class_codes

    if dedup:
        _, first = np.unique(row_hash, return_index=True)
        keep = np.zeros(len(row_hash), dtype=bool)
        keep[first] = True
    else:
        keep = np.ones(len(row_hash), dtype=bool)
    removed = pd.Series(cls[~keep]).value_counts()
    before = pd.Series(cls).value_counts()
    _, vec_counts = np.unique(vec_hash[keep], return_counts=True)
    conflict = vec_counts > 1
    stats.dedup = {
        "enabled": dedup,
        "rows_before": int(len(keep)),
        "rows_after": int(keep.sum()),
        "duplicates_removed_by_class": {str(k): int(v) for k, v in removed.items() if v},
        "duplicate_fraction_by_class": {str(k): round(float(removed.get(k, 0)) / float(v), 4) for k, v in before.items() if v},
        "distinct_feature_vectors": int(len(vec_counts)),
        "label_conflict_vectors": int(conflict.sum()),
        "rows_in_label_conflict_vectors": int(vec_counts[conflict].sum()),
        "class_counts_final": {str(k): int(v) for k, v in pd.Series(cls[keep]).value_counts().items()},
        "note": "64-bit hashes of the canonical vector; label-conflict rows are kept and "
                "stay in one split because splits are assigned per vec_hash.",
    }
    del row_hash, vec_hash

    writer = None
    offset = 0
    for part, size in zip(parts, sizes):
        table = pq.read_table(part)
        table = table.select([c for c in table.column_names if c != "row_hash"])
        table = table.filter(pa.array(keep[offset:offset + size]))
        offset += size
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema)
        writer.write_table(table)
    writer.close()


def render_report(report: dict) -> str:
    lines = ["BoT-IoT -> ToN-IoT  |  PREPARE REPORT", "=" * 72,
             f"created: {report['created']}", f"root: {report['root']}", f"settings: {report['settings']}", ""]
    for domain in DOMAINS:
        if domain not in report["domains"]:
            continue
        d = report["domains"][domain]
        lines.append(f"[{domain.upper()}] rows read {d['rows_read']:,} | kept after row policy {d['rows_after_row_policy']:,}"
                     f" | after dedup {d['dedup']['rows_after']:,}")
        for f in d["files"]:
            lines.append(f"  {f['file']}: {f.get('rows_read', 0):,} read, {f.get('rows_kept', 0):,} kept  [{f['status']}]")
        lines.append(f"  dropped by reason: {d['dropped_by_reason'] or 'none'}")
        lines.append(f"  zeek '-' set to 0: {d['dash_to_zero'] or 'none'}")
        lines.append(f"  port parsing: {d['port_parsing']}")
        for col, toks in d["invalid_tokens_top"].items():
            lines.append(f"  invalid tokens in {col}: {toks[:8]}")
        lines.append(f"  classes (raw): {d['class_counts_raw']}")
        lines.append(f"  classes (final): {d['dedup']['class_counts_final']}")
        lines.append(f"  duplicate fraction by class: {d['dedup']['duplicate_fraction_by_class']}")
        lines.append(f"  label-conflict vectors: {d['dedup']['label_conflict_vectors']:,} "
                     f"({d['dedup']['rows_in_label_conflict_vectors']:,} rows)")
        lines.append(f"  proto: {d['proto_counts']}")
        lines.append(f"  bytes evidence: {d['bytes_evidence']}")
        lines.append(f"  src_bytes of 1-packet one-way TCP flows (value: count): {d['single_packet_tcp_src_bytes_top']}")
        lines.append("")
    lines += [
        "How to read the bytes evidence:",
        "  of_which_src_bytes_eq0 should be ~0 in BOTH datasets if src_bytes counts header bytes.",
        "  For ToN, of_which_src_alt_bytes_eq0 is the other byte definition (payload when --ton-bytes ip).",
        "  If the ToN payload column is often 0 while BoT sbytes never is, the header-inclusive",
        "  src_ip_bytes/dst_ip_bytes pairing is the semantically consistent one.",
        "  min_src_bytes_per_pkt (raw): 42 in BoT = 14 Ethernet + 20 IP + 8 UDP, i.e. Argus counts the",
        "  link-layer header; Zeek *_ip_bytes start at IP (min 20-28), hence --bot-l2-bytes 14.",
    ]
    return "\n".join(lines) + "\n"


def prepare_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="folder with the 4 BoT-IoT and 23 ToN-IoT CSVs")
    ap.add_argument("--bot-root", type=Path, default=None, help="BoT-IoT folder if different from --root")
    ap.add_argument("--out", type=Path, default=Path("prepared"))
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    ap.add_argument("--ton-bytes", choices=["ip", "payload"], default="ip",
                    help="ToN byte columns paired with BoT sbytes/dbytes (default: header-inclusive src_ip_bytes)")
    ap.add_argument("--bot-l2-bytes", type=int, default=14,
                    help="Ethernet header bytes removed per packet from BoT-IoT byte counts (0 = keep raw)")
    ap.add_argument("--bot-files", type=Path, nargs="+", default=None,
                    help="explicit BoT-IoT CSV list instead of UNSW_2018_IoT_Botnet_Full5pc_1..4.csv")
    ap.add_argument("--ton-files", type=Path, nargs="+", default=None,
                    help="explicit ToN-IoT CSV list instead of Network_dataset_1..23.csv")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--max-rows-per-file", type=int, default=None, help="quick trial on the first N rows of each file")
    ap.add_argument("--allow-missing-files", action="store_true")
    ap.add_argument("--no-dedup", action="store_true", help="keep exact duplicate rows")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(args.root),
              "settings": {"ton_bytes": args.ton_bytes, "bot_l2_bytes": args.bot_l2_bytes, "dedup": not args.no_dedup,
                           "max_rows_per_file": args.max_rows_per_file}, "domains": {}}
    report_path = args.out / "prepare_report.json"
    if report_path.exists():  # keep the other domain's section when re-running one domain
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        report["domains"] = {k: v for k, v in previous.get("domains", {}).items() if k not in args.domains}

    for domain in args.domains:
        root = args.bot_root if (domain == BOT and args.bot_root) else args.root
        print(f"[{domain.upper()}] reading from {root}", flush=True)
        stats = prepare_domain(domain, root, args.out, ton_bytes=args.ton_bytes, bot_l2_bytes=args.bot_l2_bytes,
                               chunk_size=args.chunk_size, max_rows_per_file=args.max_rows_per_file,
                               allow_missing_files=args.allow_missing_files, dedup=not args.no_dedup,
                               files=args.bot_files if domain == BOT else args.ton_files)
        report["domains"][domain] = stats.to_dict()

    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    text = render_report(report)
    (args.out / "prepare_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {args.out.resolve()}")
    return 0


# ==================================================================================================
# DIAGNOSE
#   Stage 2: how far apart are BoT-IoT and ToN-IoT in the shared feature space?
#
#       python -m bot_ton_tl.diagnose --prepared prepared --out results/diagnose
#
#   Reports, on the training splits only:
#     * per-feature mean/std per dataset and the two-sample KS statistic, overall and for the
#       class pairs both taxonomies share (normal, dos, ddos, reconnaissance~scanning)
#     * a domain classifier's ROC-AUC and the proxy A-distance (Ben-David et al. 2010):
#       AUC ~0.5 -> datasets indistinguishable, ~1.0 -> trivially separable. Trees are
#       invariant to any shared monotone rescaling, so this also holds for the scaled inputs.
# ==================================================================================================

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def domain_auc(Xs: np.ndarray, Xt: np.ndarray, seed: int = 0, max_rows: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = min(len(Xs), len(Xt), max_rows)
    if n < 20:
        return float("nan"), float("nan")
    Xs = Xs[rng.choice(len(Xs), n, replace=False)]
    Xt = Xt[rng.choice(len(Xt), n, replace=False)]
    X = np.concatenate([Xs, Xt])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    err = float(((p >= 0.5) != y).mean())
    return float(roc_auc_score(y, p)), 2.0 * (1.0 - 2.0 * min(err, 0.5))


def diagnose_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=Path("prepared"))
    ap.add_argument("--out", type=Path, default=Path("results") / "diagnose")
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--cap", type=int, default=20_000, help="training rows per class per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    caps = {"train": args.cap, "val": 0, "test": 0}
    src = load_domain(args.prepared, BOT, caps=caps, seed=args.seed).train
    tgt = load_domain(args.prepared, TON, caps=caps, seed=args.seed).train
    groups = tuple(args.features)
    Xs, names = build_features(src, BOT, groups)
    Xt, _ = build_features(tgt, TON, groups)
    cs, ct = src["cls"].to_numpy(dtype=object), tgt["cls"].to_numpy(dtype=object)

    pairs = [("all", np.ones(len(cs), bool), np.ones(len(ct), bool))]
    inv_t = {v: k for k, v in SHARED_CLASS_MAP[TON].items()}
    for s_cls, shared in SHARED_CLASS_MAP[BOT].items():
        t_cls = inv_t[shared]
        pairs.append((f"{s_cls}~{t_cls}", cs == s_cls, ct == t_cls))

    rows = []
    for j, name in enumerate(names):
        row = {"feature": name, "bot_mean": float(Xs[:, j].mean()), "ton_mean": float(Xt[:, j].mean()),
               "bot_std": float(Xs[:, j].std()), "ton_std": float(Xt[:, j].std())}
        for label, ms, mt in pairs:
            row[f"ks_{label}"] = ks_statistic(Xs[ms, j], Xt[mt, j])
        rows.append(row)
    shift = pd.DataFrame(rows).sort_values("ks_all", ascending=False)
    shift.to_csv(args.out / "feature_shift.csv", index=False)

    lines = ["BoT-IoT vs ToN-IoT | domain-shift diagnostics (training splits)", "=" * 72,
             f"rows: BoT {len(cs):,} {pd.Series(cs).value_counts().to_dict()}",
             f"      ToN {len(ct):,} {pd.Series(ct).value_counts().to_dict()}", "",
             "Domain classifier (3-fold CV, gradient boosting). AUC 0.5 = indistinguishable.",
             f"{'subset':28s} {'n_bot':>8s} {'n_ton':>8s} {'AUC':>7s} {'proxy A-dist':>13s}"]
    for label, ms, mt in pairs:
        auc, pad = domain_auc(Xs[ms], Xt[mt], args.seed)
        lines.append(f"{label:28s} {int(ms.sum()):8d} {int(mt.sum()):8d} {auc:7.3f} {pad:13.3f}")
    lines += ["", "Most shifted features (KS statistic; 0 = same distribution, 1 = disjoint):",
              shift.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
              "Reading this: a near-1.0 AUC even within matched classes means the encoder must",
              "learn an invariance (DANN) or be re-fitted on target labels (fine-tuning); expect",
              "source_only to fail. Features with KS close to 1 inside matched classes are",
              "candidates to drop in an ablation (--features)."]
    text = "\n".join(lines) + "\n"
    (args.out / "diagnose_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ==================================================================================================
# SYNTHETIC
#   Small synthetic stand-ins for the 27 raw CSVs, with the real headers and quirks.
#
#   Used by the tests and for a dry run of the whole pipeline without the real data:
#
#       python -m bot_ton_tl.synthetic --out synthetic_raw
#
#   Quirks reproduced: BoT-IoT hexadecimal ICMP 'ports' and -1 ARP ports, header-inclusive
#   Argus byte counts, ToN-IoT payload vs IP byte columns, Zeek '-' durations, the
#   src_bytes='0.0.0.0' records in Network_dataset_1/22/23, the extra 'uid' column in
#   Network_dataset_6 and exact duplicate rows. Numbers are invented - only the format is real.
# ==================================================================================================

BOT_HEADER = [
    "pkSeqID", "stime", "flgs", "flgs_number", "proto", "proto_number", "saddr", "sport", "daddr", "dport",
    "pkts", "bytes", "state", "state_number", "ltime", "seq", "dur", "mean", "stddev", "sum", "min", "max",
    "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate", "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP",
    "TnP_PDstIP", "TnP_PerProto", "TnP_Per_Dport", "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP",
    "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP", "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
    "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP", "attack", "category", "subcategory",
]
TON_HEADER = [
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration", "src_bytes", "dst_bytes",
    "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_query",
    "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version",
    "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth",
    "http_method", "http_uri", "http_referrer", "http_version", "http_request_body_len",
    "http_response_body_len", "http_status_code", "http_user_agent", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "label", "type",
]

# archetype: (proto choices, dst ports, src pkts mean, reply prob, dst pkts mean, payload/pkt, duration scale)
ARCHETYPES = {
    "normal": (["tcp", "udp"], [80, 443, 53, 1883, 8080], 8, 0.95, 8, 300, 0.8),
    "dos": (["tcp", "udp"], [80, 8080], 40, 0.3, 2, 40, 0.05),
    "ddos": (["tcp", "udp"], [80, 53], 12, 0.1, 1, 30, 0.02),
    "scan": (["tcp", "tcp", "udp", "icmp"], None, 1.2, 0.4, 1, 0, 0.001),
    "theft": (["tcp"], [5000, 8000, 21], 20, 1.0, 20, 500, 3.0),
    "backdoor": (["tcp"], [4444, 31337], 6, 1.0, 6, 120, 20.0),
    "web": (["tcp"], [80, 8080], 6, 1.0, 5, 600, 0.3),
    "password": (["tcp"], [22, 21, 23], 10, 1.0, 9, 80, 1.5),
    "ransomware": (["tcp"], [445, 139], 50, 1.0, 40, 1200, 5.0),
    "mitm": (["tcp", "udp", "icmp"], [80, 53], 3, 0.6, 2, 100, 0.2),
}
BOT_CLASSES = {"ddos": ("DDoS", "ddos", 0.45), "dos": ("DoS", "dos", 0.38), "reconnaissance": ("Reconnaissance", "scan", 0.11),
               "normal": ("Normal", "normal", 0.05), "theft": ("Theft", "theft", 0.01)}
TON_CLASSES = {"scanning": ("scan", 0.22), "ddos": ("ddos", 0.18), "dos": ("dos", 0.12), "normal": ("normal", 0.12),
               "xss": ("web", 0.09), "password": ("password", 0.08), "injection": ("web", 0.07),
               "backdoor": ("backdoor", 0.06), "ransomware": ("ransomware", 0.04), "mitm": ("mitm", 0.02)}
BOT_SUB = {"ddos": ["TCP", "UDP", "HTTP"], "dos": ["TCP", "UDP", "HTTP"], "reconnaissance": ["Service_Scan", "OS_Fingerprint"],
           "normal": ["Normal"], "theft": ["Keylogging", "Data_Exfiltration"]}


def _flows(archetype: str, n: int, rng: np.random.Generator, dur_scale: float) -> dict[str, np.ndarray]:
    protos, ports, sp_mean, reply_p, dp_mean, payload, dur = ARCHETYPES[archetype]
    proto = rng.choice(protos, n)
    sp = 1 + rng.poisson(sp_mean, n)
    reply = rng.random(n) < reply_p
    dp = np.where(reply, 1 + rng.poisson(dp_mean, n), 0)
    pay_s = np.maximum(0, rng.normal(payload, payload * 0.3 + 1, n)).round()
    pay_d = np.maximum(0, rng.normal(payload * 0.8, payload * 0.3 + 1, n)).round()
    dport = rng.choice(ports, n) if ports else rng.integers(1, 65535, n)
    sport = rng.integers(1024, 65535, n)
    duration = rng.lognormal(np.log(dur * dur_scale + 1e-4), 0.8, n)
    return {"proto": proto, "spkts": sp, "dpkts": dp, "sport": sport, "dport": dport, "dur": duration,
            "s_payload": sp * pay_s, "d_payload": dp * pay_d}


def _class_draw(classes: dict, n: int, rng: np.random.Generator, weight_index: int) -> np.ndarray:
    names = list(classes)
    w = np.array([classes[c][weight_index] for c in names], dtype=float)
    return rng.choice(names, n, p=w / w.sum())


def make_bot_file(n: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    cls = _class_draw(BOT_CLASSES, n, rng, 2)
    cols = {c: np.zeros(n, dtype=object) for c in BOT_HEADER}
    frame = pd.DataFrame(cols)
    for c in np.unique(cls):
        m = cls == c
        f = _flows(BOT_CLASSES[c][1], int(m.sum()), rng, dur_scale=3.0)
        proto = f["proto"].astype(object)
        arp = rng.random(len(proto)) < (0.03 if c == "normal" else 0.0)
        proto[arp] = "arp"
        hdr = 54  # Ethernet + IP + TCP headers: Argus counts every byte of the frame
        frame.loc[m, "proto"] = proto
        frame.loc[m, "spkts"] = f["spkts"]
        frame.loc[m, "dpkts"] = f["dpkts"]
        frame.loc[m, "sbytes"] = (f["s_payload"] + hdr * f["spkts"]).astype(int)
        frame.loc[m, "dbytes"] = (f["d_payload"] + hdr * f["dpkts"]).astype(int)
        frame.loc[m, "dur"] = np.round(f["dur"], 6)
        sport = f["sport"].astype(object)
        dport = f["dport"].astype(object)
        icmp = proto == "icmp"
        sport[icmp] = "0x0008"
        dport[icmp] = [f"0x{v:04x}" for v in rng.integers(0, 65535, int(icmp.sum()))]
        sport[arp], dport[arp] = -1, -1
        frame.loc[m, "sport"] = sport
        frame.loc[m, "dport"] = dport
        frame.loc[m, "state"] = np.where(f["dpkts"] > 0, "CON", np.where(icmp, "ECO", "INT"))
        frame.loc[m, "category"] = BOT_CLASSES[c][0]
        frame.loc[m, "subcategory"] = rng.choice(BOT_SUB[c], int(m.sum()))
        frame.loc[m, "attack"] = 0 if c == "normal" else 1
    frame["pkSeqID"] = np.arange(start_id, start_id + n)
    frame["stime"] = 1528089000 + np.arange(n) * 0.01
    frame["ltime"] = frame["stime"] + frame["dur"].astype(float)
    frame["saddr"] = "192.168.100.147"
    frame["daddr"] = "192.168.100.3"
    frame["pkts"] = frame["spkts"].astype(int) + frame["dpkts"].astype(int)
    frame["bytes"] = frame["sbytes"].astype(int) + frame["dbytes"].astype(int)
    return _with_duplicates(frame, rng)


def make_ton_file(n: int, rng: np.random.Generator, number: int) -> pd.DataFrame:
    cls = _class_draw(TON_CLASSES, n, rng, 1)
    frame = pd.DataFrame({c: np.zeros(n, dtype=object) for c in TON_HEADER})
    for c in np.unique(cls):
        m = cls == c
        f = _flows(TON_CLASSES[c][0], int(m.sum()), rng, dur_scale=1.0)
        hdr = 40  # Zeek *_ip_bytes: IP + TCP headers, no Ethernet header
        frame.loc[m, "proto"] = f["proto"]
        frame.loc[m, "src_pkts"] = f["spkts"]
        frame.loc[m, "dst_pkts"] = f["dpkts"]
        frame.loc[m, "src_bytes"] = f["s_payload"].astype(int)   # Zeek orig_bytes: payload only
        frame.loc[m, "dst_bytes"] = f["d_payload"].astype(int)
        frame.loc[m, "src_ip_bytes"] = (f["s_payload"] + hdr * f["spkts"]).astype(int)
        frame.loc[m, "dst_ip_bytes"] = (f["d_payload"] + hdr * f["dpkts"]).astype(int)
        frame.loc[m, "duration"] = np.round(f["dur"], 6)
        frame.loc[m, "src_port"] = f["sport"]
        frame.loc[m, "dst_port"] = f["dport"]
        frame.loc[m, "conn_state"] = np.where(f["dpkts"] > 0, "SF", "S0")
        frame.loc[m, "type"] = c
        frame.loc[m, "label"] = 0 if c == "normal" else 1
    one_pkt = (frame["src_pkts"].astype(int) == 1) & (frame["dst_pkts"].astype(int) == 0)
    frame.loc[one_pkt & (rng.random(n) < 0.5), "duration"] = "-"
    for col in ("service", "dns_query", "ssl_version", "http_method", "weird_name"):
        frame[col] = "-"
    frame["ts"] = 1554198358 + np.arange(n)
    frame["src_ip"], frame["dst_ip"] = "192.168.1.37", "192.168.1.1"
    if number in (1, 22, 23):  # the known corrupt records
        bad = rng.choice(n, size=3, replace=False)
        frame.loc[bad, "src_bytes"] = "0.0.0.0"
    frame = _with_duplicates(frame, rng)
    if number == 6:
        frame.insert(0, "uid", [f"C{i:08d}" for i in range(len(frame))])
    return frame


def _with_duplicates(frame: pd.DataFrame, rng: np.random.Generator, frac: float = 0.1) -> pd.DataFrame:
    dup = frame.sample(frac=frac, random_state=int(rng.integers(1 << 31)))
    return pd.concat([frame, dup], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1 << 31)))


def write_synthetic(out: Path, bot_rows_per_file: int = 2000, ton_rows_per_file: int = 600, seed: int = 7) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(1, 5):
        make_bot_file(bot_rows_per_file, rng, i * 10_000_000).to_csv(out / f"UNSW_2018_IoT_Botnet_Full5pc_{i}.csv", index=False)
    for i in range(1, 24):
        make_ton_file(ton_rows_per_file, rng, i).to_csv(out / f"Network_dataset_{i}.csv", index=False)
    return out


def synthetic_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("synthetic_raw"))
    ap.add_argument("--bot-rows", type=int, default=2000)
    ap.add_argument("--ton-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    print(f"wrote {write_synthetic(args.out, args.bot_rows, args.ton_rows, args.seed).resolve()}")
    return 0


# ==================================================================================================
# EXPERIMENT
#   Stage 3: baselines vs. transfer learning on ToN-IoT with scarce labels.
#
#       python -m bot_ton_tl.experiment --prepared prepared --out results/run1
#
#   For every seed the BoT-IoT model is pre-trained once (and a DANN model once); then for
#   every label budget k (k labelled ToN-IoT flows per class, drawn from the ToN training
#   pool) and task (binary / multiclass) these methods are trained and scored on the fixed,
#   group-split ToN-IoT test set:
#
#       source_only   BoT-IoT model applied unchanged (binary: P(attack) = 1 - P(normal))
#       dann_zero     DANN model applied unchanged (alignment used unlabeled ToN flows only)
#       target_only   same network trained from scratch on the k-shot budget    (no transfer)
#       rf            random forest on the k-shot budget                         (for information)
#       frozen        pre-trained encoder frozen + new head (the repo's CNN-TL recipe)
#       finetune      pre-trained encoder, LP-FT (head first, then everything at a 10x smaller
#                     encoder learning rate - conservative, keeps the encoder close to BoT-IoT)
#       dann          DANN-pre-trained encoder, LP-FT
#       finetune_full pre-trained encoder trained exactly like target_only (same learning rate,
#                     steps and early stopping): the ONLY difference from target_only is the
#                     initialisation, so finetune_full - target_only is the clean transfer effect
#       dann_full     same, starting from the DANN-pre-trained encoder
#       tl_match      PROPOSED. BoT-IoT-pre-trained network + FixMatch-style consistency training
#                     on the unlabelled ToN pool: confident predictions on clean flows become
#                     pseudo-labels for corrupted copies (a share of features swapped with other
#                     pool flows, as in SCARF). Uses the k-shot labels + unlabelled ToN flows.
#       match_scratch ablation: tl_match without the BoT-IoT initialisation
#
#   Model selection inside a budget uses 20% of that budget only; the large ToN validation
#   split is never used for few-shot runs (it would leak labels beyond the budget).
# ==================================================================================================

ALL_METHODS = ("source_only", "dann_zero", "target_only", "rf", "frozen", "finetune", "dann",
               "finetune_full", "dann_full", "tl_match", "match_scratch")
NEEDS_SOURCE = {"source_only", "frozen", "finetune", "finetune_full", "tl_match"}
NEEDS_DANN = {"dann_zero", "dann", "dann_full"}


def experiment_parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=Path("prepared"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    ap.add_argument("--shots", nargs="+", type=int, default=[10, 50, 100, 500, 1000],
                    help="labelled ToN-IoT flows per class")
    ap.add_argument("--include-full", action="store_true", help="also train on the whole (capped) ToN pool (upper bound)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arch", choices=["mlp", "cnn1d"], default="mlp")
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--norm", choices=["shared", "source", "per_domain"], default="shared",
                    help="shared: one scaler fitted on BoT train + the unlabeled ToN pool (features keep the "
                         "same physical units in both datasets); per_domain: separate scalers (AdaBN-style "
                         "ablation; also absorbs label-shift, so use with care)")
    ap.add_argument("--merge", nargs="*", default=[], metavar="A+B",
                    help="merge classes in both datasets, e.g. dos+ddos (single flows cannot tell DoS from "
                         "DDoS: in BoT-IoT almost all label-conflicting vectors are dos|ddos)")
    ap.add_argument("--unseen-classes", nargs="*", default=[],
                    help="ToN classes never labelled (zero-day study); binary detection reported per class")
    ap.add_argument("--source-cap", type=int, default=50_000, help="BoT-IoT training rows per class")
    ap.add_argument("--target-pool-cap", type=int, default=20_000, help="ToN training-pool rows per class")
    ap.add_argument("--test-cap", type=int, default=50_000, help="test rows per class (both datasets)")
    ap.add_argument("--val-cap", type=int, default=5_000, help="BoT-IoT validation rows per class")
    ap.add_argument("--test-files", nargs="*", type=int, default=[],
                    help="ToN file numbers used as test set (scenario split) instead of the hash split")
    ap.add_argument("--split-salt", type=int, default=2026)
    ap.add_argument("--steps-pretrain", type=int, default=4000)
    ap.add_argument("--steps-dann", type=int, default=4000)
    ap.add_argument("--steps-target", type=int, default=1500)
    ap.add_argument("--steps-lp", type=int, default=500)
    ap.add_argument("--steps-ft", type=int, default=1000)
    ap.add_argument("--steps-match", type=int, default=3000, help="max steps for tl_match / match_scratch")
    ap.add_argument("--match-threshold", type=float, default=0.9, help="confidence needed for a pseudo-label")
    ap.add_argument("--match-corrupt", type=float, default=0.3, help="share of features swapped in the corrupted copy")
    ap.add_argument("--match-ratio", type=int, default=4, help="unlabelled flows per labelled flow in a batch")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args(argv)
    if args.quick:
        args.shots, args.seeds = args.shots[:2], args.seeds[:1]
        args.steps_pretrain = args.steps_dann = 200
        args.steps_target, args.steps_lp, args.steps_ft, args.steps_match = 150, 60, 100, 150
        args.rf_trees = 25
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out is None:
        args.out = Path("results") / time.strftime("run_%Y%m%d_%H%M%S")
    return args


class Encoded:
    """Features + labels of one split, ready for training."""

    def __init__(self, frame: pd.DataFrame, domain: str, groups):
        X, self.names = build_features(frame, domain, groups)
        self.raw = X
        self.X = X
        self.cls = frame["cls"].to_numpy(dtype=object)
        self.binary = frame["binary"].to_numpy(dtype=np.int64)

    def rescale(self, scaler: Standardizer) -> None:
        self.X = scaler.transform(self.raw)


def steps_for(base: int, n: int, batch: int) -> int:
    """Few-shot budgets use the base step count; big training sets get ~3 epochs (max 8000)."""
    return int(max(base, min(8000, 3 * n // max(batch, 1))))


def run(args: argparse.Namespace) -> pd.DataFrame:
    t_start = time.time()
    out: Path = args.out
    (out / "models").mkdir(parents=True, exist_ok=True)
    caps_src = {"train": args.source_cap, "val": args.val_cap, "test": args.test_cap}
    # The ToN validation split is deliberately not loaded: few-shot model selection may only
    # use the labelled budget itself.
    caps_tgt = {"train": args.target_pool_cap, "val": 0, "test": args.test_cap}
    print(f"loading {args.prepared} ...", flush=True)
    src: DomainData = load_domain(args.prepared, BOT, caps=caps_src, salt=args.split_salt)
    tgt: DomainData = load_domain(args.prepared, TON, caps=caps_tgt, salt=args.split_salt,
                                  test_files=tuple(args.test_files))
    groups = tuple(args.features)
    merge = {c: g for g in args.merge for c in g.lower().split("+")}
    for frame in (src.train, src.val, src.test, tgt.train, tgt.val, tgt.test):
        frame["cls"] = frame["cls"].map(lambda c: merge.get(c, c))

    s_tr, s_va, s_te = (Encoded(f, BOT, groups) for f in (src.train, src.val, src.test))
    t_tr, t_te = (Encoded(f, TON, groups) for f in (tgt.train, tgt.test))
    mask = continuous_mask(s_tr.names)
    if args.norm == "shared":
        src_scaler = tgt_scaler = Standardizer(mask).fit(np.concatenate([s_tr.raw, t_tr.raw]))
    else:
        src_scaler = Standardizer(mask).fit(s_tr.raw)
        tgt_scaler = Standardizer(mask).fit(t_tr.raw) if args.norm == "per_domain" else src_scaler
    for e in (s_tr, s_va, s_te):
        e.rescale(src_scaler)
    for e in (t_tr, t_te):
        e.rescale(tgt_scaler)
    (out / "models" / "scalers.json").write_text(json.dumps(
        {"features": s_tr.names, "source": src_scaler.state(), "target": tgt_scaler.state()}), encoding="utf-8")

    src_classes = sorted(set(s_tr.cls))
    src_index = {c: i for i, c in enumerate(src_classes)}
    ys_tr = np.array([src_index[c] for c in s_tr.cls])
    va_keep = np.isin(s_va.cls, src_classes)
    ys_va = np.array([src_index[c] for c in s_va.cls[va_keep]])
    unseen = tuple(c.lower() for c in args.unseen_classes)
    tgt_classes = sorted(c for c in set(t_tr.cls) | set(t_te.cls) if c not in unseen)
    tgt_index = {c: i for i, c in enumerate(tgt_classes)}
    print(f"source classes {src_classes} | target classes {tgt_classes} | unseen {list(unseen)}", flush=True)
    print(f"rows: source train {len(s_tr.cls):,} val {len(s_va.cls):,} test {len(s_te.cls):,} | "
          f"target pool {len(t_tr.cls):,} test {len(t_te.cls):,} | features {len(s_tr.names)}", flush=True)

    rows: list[dict] = []
    per_class: list[dict] = []

    def record(task, method, shots, seed, metrics, extra=None, cls_true=None, correct=None):
        row = {"task": task, "method": method, "shots": shots, "seed": seed, **metrics, **(extra or {})}
        rows.append(row)
        if cls_true is not None:
            for c, rate in per_class_rates(cls_true, correct).items():
                per_class.append({"task": task, "method": method, "shots": shots, "seed": seed, "cls": c,
                                  "rate": rate, "n": int((cls_true == c).sum())})

    def evaluate(task, method, shots, seed, proba=None, pred_labels=None, extra=None):
        """proba: (n_test, n_out) for the task's output space; binary uses column 1 = attack."""
        if task == "binary":
            p_attack = proba[:, 1]
            m = binary_metrics(t_te.binary, p_attack)
            correct = (p_attack >= 0.5).astype(int) == t_te.binary
            record(task, method, shots, seed, m, extra, t_te.cls, correct)
        else:
            keep = np.isin(t_te.cls, tgt_classes)
            y_true = np.array([tgt_index[c] for c in t_te.cls[keep]])
            y_pred = proba[keep].argmax(1) if pred_labels is None else pred_labels[keep]
            m = multiclass_metrics(y_true, y_pred)
            record(task, method, shots, seed, m, extra, t_te.cls[keep], y_true == y_pred)
        return rows[-1]

    def zero_shot(model, name, seed):
        proba_src = predict_proba(model, t_te.X, args.device)
        normal_col = src_index.get("normal")
        p_normal = proba_src[:, normal_col] if normal_col is not None else np.zeros(len(t_te.X))
        if "binary" in args.tasks:
            r = evaluate("binary", name, 0, seed, np.column_stack([p_normal, 1 - p_normal]))
            print(f"  [{name}] binary zero-shot macro-F1={r['macro_f1']:.3f} AUC={r['roc_auc']:.3f}", flush=True)
        # Multiclass only on the classes both taxonomies share (reconnaissance ~ scanning).
        shared_t = {merge.get(c, c): merge.get(v, v) for c, v in SHARED_CLASS_MAP[TON].items()}
        shared_s = {merge.get(c, c): merge.get(v, v) for c, v in SHARED_CLASS_MAP[BOT].items()}
        keep = np.isin(t_te.cls, list(shared_t))
        names = sorted(set(shared_t.values()))
        src_to_shared = {src_index[c]: names.index(s) for c, s in shared_s.items() if c in src_index}
        cols = list(src_to_shared)
        if keep.any() and cols:
            pred = np.array([src_to_shared[cols[j]] for j in proba_src[keep][:, cols].argmax(1)])
            y_true = np.array([names.index(shared_t[c]) for c in t_te.cls[keep]])
            m = multiclass_metrics(y_true, pred)
            record("multiclass_shared", name, 0, seed, m, {"classes": "|".join(names)},
                   t_te.cls[keep], y_true == pred)

    base = TrainConfig(batch_size=args.batch_size, device=args.device)
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        src_model = dann_model = None
        if NEEDS_SOURCE & set(args.methods):
            t0 = time.time()
            cfg = scaled(base, steps=args.steps_pretrain, eval_every=max(args.steps_pretrain // 20, 1), patience=5)
            src_model, info = train_supervised(args.arch, s_tr.X, ys_tr, len(src_classes), cfg, seed,
                                               s_va.X[va_keep], ys_va)
            te_keep = np.isin(s_te.cls, src_classes)
            pred = predict_proba(src_model, s_te.X[te_keep], args.device).argmax(1)
            m = multiclass_metrics(np.array([src_index[c] for c in s_te.cls[te_keep]]), pred)
            record("source_in_domain", "source_pretrain", 0, seed, m,
                   {"params": count_params(src_model), "train_s": round(time.time() - t0, 1), **info})
            print(f"  source pre-training: in-domain macro-F1={m['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            torch.save(src_model.encoder.state_dict(), out / "models" / f"source_encoder_seed{seed}.pt")
            if "source_only" in args.methods:
                zero_shot(src_model, "source_only", seed)
        if NEEDS_DANN & set(args.methods):
            t0 = time.time()
            cfg = scaled(base, steps=args.steps_dann, eval_every=max(args.steps_dann // 20, 1))
            dann_model, info = train_dann(args.arch, s_tr.X, ys_tr, len(src_classes), t_tr.X, cfg, seed)
            print(f"  DANN pre-training: final domain acc={info['final_domain_acc']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            torch.save(dann_model.encoder.state_dict(), out / "models" / f"dann_encoder_seed{seed}.pt")
            if "dann_zero" in args.methods:
                zero_shot(dann_model, "dann_zero", seed)

        budgets: list[int | None] = list(args.shots) + ([None] if args.include_full else [])
        for shots in budgets:
            label = "full" if shots is None else shots
            rng = np.random.default_rng(seed * 100_003 + (shots or 0))
            idx = sample_budget(t_tr.cls, shots, rng, exclude=unseen)
            fit_i, hold_i = holdout_split(t_tr.cls[idx], 0.2, rng)
            fit_i, hold_i = idx[fit_i], idx[hold_i]
            n_fit = len(fit_i)
            unlabelled = np.setdiff1d(np.arange(len(t_tr.cls)), idx)
            if len(unlabelled) == 0:  # 'full' budget: every pool flow is labelled
                unlabelled = np.arange(len(t_tr.cls))
            print(f"  budget {label}/class -> {len(idx):,} labelled flows ({n_fit} fit / {len(hold_i)} hold)", flush=True)
            for task in args.tasks:
                if task == "binary":
                    y_all, n_out = t_tr.binary, 2
                else:
                    y_all = np.array([tgt_index.get(c, -1) for c in t_tr.cls])
                    n_out = len(tgt_classes)
                Xf, yf, Xh, yh = t_tr.X[fit_i], y_all[fit_i], t_tr.X[hold_i], y_all[hold_i]
                cfg_t = scaled(base, steps=steps_for(args.steps_target, n_fit, args.batch_size), eval_every=50)
                cfg_lp = scaled(base, steps=steps_for(args.steps_lp, n_fit, args.batch_size), eval_every=50)
                cfg_ft = scaled(base, steps=steps_for(args.steps_ft, n_fit, args.batch_size), eval_every=50, lr=3e-4)
                cfg_m = scaled(base, steps=steps_for(args.steps_match, n_fit, args.batch_size), eval_every=100)
                common = {"n_labelled": int(len(idx))}
                for method in args.methods:
                    if method in ("source_only", "dann_zero"):
                        continue
                    t0 = time.time()
                    if method == "rf":
                        rf = RandomForestClassifier(n_estimators=args.rf_trees, class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=seed)
                        rf.fit(np.concatenate([Xf, Xh]), np.concatenate([yf, yh]))
                        proba = np.zeros((len(t_te.X), n_out))
                        proba[:, rf.classes_] = rf.predict_proba(t_te.X)
                        info = {}
                    elif method in ("target_only", "finetune_full", "dann_full"):
                        donor = {"finetune_full": src_model, "dann_full": dann_model}.get(method)
                        model, info = train_supervised(args.arch, Xf, yf, n_out, cfg_t, seed, Xh, yh,
                                                       encoder_state=None if donor is None else donor.encoder.state_dict())
                        proba = predict_proba(model, t_te.X, args.device)
                    elif method in ("tl_match", "match_scratch"):
                        enc = src_model.encoder.state_dict() if method == "tl_match" else None
                        model, info = train_consistency(enc, args.arch, Xf, yf, n_out, t_tr.X[unlabelled], cfg_m,
                                                        seed, Xh, yh, threshold=args.match_threshold,
                                                        corrupt=args.match_corrupt, unlabeled_ratio=args.match_ratio)
                        proba = predict_proba(model, t_te.X, args.device)
                    elif method in ("frozen", "finetune", "dann"):
                        donor = dann_model if method == "dann" else src_model
                        head = "mlp" if method == "frozen" else "linear"
                        model, info = finetune(donor.encoder.state_dict(), args.arch, Xf, yf, n_out, seed,
                                               cfg_lp if method != "frozen" else cfg_t,
                                               None if method == "frozen" else cfg_ft, Xh, yh, head=head)
                        proba = predict_proba(model, t_te.X, args.device)
                    r = evaluate(task, method, label, seed, proba,
                                 extra={**common, **info, "train_s": round(time.time() - t0, 1)})
                    print(f"    {task:10s} {method:12s} macro-F1={r['macro_f1']:.3f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out / "summary.csv", index=False)
    text = render_summary(summary, args, src, tgt, time.time() - t_start) + render_gains(results)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(source_classes=src_classes, target_classes=tgt_classes, features=s_tr.names,
                  source_split_counts=src.split_counts, target_split_counts=tgt.split_counts)
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    plot_curves(summary, out)
    print("\n" + text)
    print(f"results written to {out.resolve()}")
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ("macro_f1", "balanced_accuracy", "accuracy", "f1", "recall_dr", "far_fpr", "roc_auc",
                           "weighted_f1") if m in results]
    g = results.groupby(["task", "method", "shots"], sort=False)[metrics]
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = g.size()
    return summary.reset_index()


def render_summary(summary: pd.DataFrame, args, src: DomainData, tgt: DomainData, seconds: float) -> str:
    lines = ["BoT-IoT -> ToN-IoT transfer learning | summary (mean +- std over seeds)", "=" * 78,
             f"arch={args.arch} features={args.features} norm={args.norm} seeds={args.seeds} "
             f"unseen={args.unseen_classes or 'none'} merge={args.merge or 'none'} "
             f"test_files={args.test_files or 'hash split'}",
             f"ToN test rows by class: {tgt.test['cls'].value_counts().to_dict()}", ""]
    order = lambda c: (int(c) if str(c).isdigit() else float("inf"))  # noqa: E731  0,10,...,full
    for task in summary["task"].unique():
        sub = summary[summary["task"] == task].assign(shots=lambda d: d["shots"].astype(str))
        metrics = ["macro_f1", "roc_auc"] if task == "binary" else ["macro_f1"]
        for metric in metrics:
            lines.append(f"[{task}]  {metric}   (columns = labelled ToN flows per class; 0 = no ToN labels)")
            table = sub.pivot_table(index="method", columns="shots", values=f"{metric}_mean", aggfunc="first")
            std = sub.pivot_table(index="method", columns="shots", values=f"{metric}_std", aggfunc="first")
            columns = sorted(table.columns, key=order)
            table, std = table[columns], std.reindex(index=table.index, columns=columns)
            cells = table.copy().astype(object)
            for c in table.columns:
                for i in table.index:
                    v, s = table.loc[i, c], std.loc[i, c]
                    cells.loc[i, c] = "" if pd.isna(v) else (f"{v:.3f}" if pd.isna(s) else f"{v:.3f}+-{s:.3f}")
            lines.append(cells.to_string())
            lines.append("")
    lines.append(f"total time {seconds / 60:.1f} min")
    return "\n".join(lines) + "\n"


def render_gains(results: pd.DataFrame) -> str:
    """Paired comparison per seed: transfer method minus each no-transfer baseline."""
    order = lambda c: (int(c) if str(c).isdigit() else float("inf"))  # noqa: E731
    transfer = [m for m in ("tl_match", "match_scratch", "finetune_full", "dann_full", "finetune", "dann", "frozen")
                if m in set(results["method"])]
    lines = ["GAIN OVER target_only, paired by seed: mean macro-F1 difference (seeds where the method wins / seeds)",
             "target_only = same network trained from scratch on the same labelled flows", ""]
    for task in ("binary", "multiclass"):
        r = results[(results["task"] == task) & (results["shots"].astype(str) != "0")]
        if r.empty or not transfer:
            continue
        r = r.assign(shots=r["shots"].astype(str))
        piv = r.pivot_table(index=["shots", "seed"], columns="method", values="macro_f1", aggfunc="first")
        for base in ("target_only",):
            if base not in piv:
                continue
            table = {}
            for m in transfer:
                if m not in piv:
                    continue
                d = (piv[m] - piv[base]).dropna()
                g = d.groupby(level="shots")
                table[m] = {k: f"{v.mean():+.3f} ({int((v > 0).sum())}/{len(v)})" for k, v in g}
            frame = pd.DataFrame(table).T
            frame = frame[sorted(frame.columns, key=order)]
            lines += [f"[{task}] method minus {base}", frame.to_string(), ""]
    return "\n".join(lines) + "\n"


def plot_curves(summary: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task in ("binary", "multiclass"):
        sub = summary[(summary["task"] == task)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        numeric = sub[sub["shots"].apply(lambda s: isinstance(s, (int, np.integer)) and s > 0)]
        for method, g in numeric.groupby("method", sort=False):
            g = g.sort_values("shots")
            ax.errorbar(g["shots"], g["macro_f1_mean"], yerr=g["macro_f1_std"].fillna(0), marker="o",
                        capsize=3, label=method)
        for method, g in sub[sub["shots"] == 0].groupby("method", sort=False):
            ax.axhline(g["macro_f1_mean"].iloc[0], linestyle="--", linewidth=1, color="grey")
            ax.annotate(f"{method} (0 labels)", (1, g["macro_f1_mean"].iloc[0]), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey")
        ax.set_xscale("log")
        ticks = sorted(numeric["shots"].astype(int).unique())
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
        ax.set_xlabel("labelled ToN-IoT flows per class")
        ax.set_ylabel("macro-F1 on ToN-IoT test")
        ax.set_title(f"BoT-IoT -> ToN-IoT, {task}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"learning_curve_{task}.png", dpi=150)
        plt.close(fig)


def experiment_main(argv=None) -> int:
    run(experiment_parse_args(argv))
    return 0
# ==================================================================================================
# ORCHESTRATOR - runs the stages above from the SETTINGS block (or the command line)
# ==================================================================================================

QUICK_ROWS_PER_FILE = 100_000


class _Tee:
    """Mirror console output into pipeline_log.txt."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __getattr__(self, name):  # isatty, encoding, ... come from the real console
        return getattr(self.stream, name)


def _expected_files(root: Path) -> list[Path]:
    return ([root / BOT_SPEC.file_name(i) for i in BOT_SPEC.file_numbers]
            + [root / ton_spec().file_name(i) for i in ton_spec().file_numbers])


def _prepared_is_current(prepared: Path, max_rows: int | None) -> bool:
    report = prepared / "prepare_report.json"
    if not all((prepared / f).is_file() for f in ("bot.parquet", "ton.parquet", "prepare_report.json")):
        return False
    settings = json.loads(report.read_text(encoding="utf-8")).get("settings", {})
    return (settings.get("ton_bytes") == TON_BYTES and settings.get("bot_l2_bytes") == BOT_L2_BYTES
            and settings.get("dedup") is True and settings.get("max_rows_per_file") == max_rows
            and set(json.loads(report.read_text(encoding="utf-8")).get("domains", {})) == {BOT, TON})


def _experiment_argv(prepared: Path, out: Path, quick: bool, extra: list[str] = ()) -> list[str]:
    argv = ["--prepared", str(prepared), "--out", str(out),
            "--seeds", *map(str, SEEDS), "--shots", *map(str, SHOTS), "--tasks", *TASKS,
            "--methods", *METHODS, "--arch", ARCH, "--features", *FEATURES, "--device", DEVICE]
    if INCLUDE_FULL:
        argv.append("--include-full")
    if quick:
        argv.append("--quick")
    return argv + list(extra)  # later options override earlier ones (argparse keeps the last)


ABLATION_ARGS = {
    "no_timing": lambda: ["--features", *[g for g in FEATURES if g != "timing"]],
    "no_ports": lambda: ["--features", *[g for g in FEATURES if g != "ports"]],
    "unseen_classes": lambda: ["--unseen-classes", *UNSEEN_CLASSES],
    "merged_classes": lambda: ["--merge", *MERGE_CLASSES],
}


def run_pipeline(root: Path, out_dir: Path, stages: list[str], quick: bool, ablations: list[str],
                 reuse_prepared: bool = True) -> None:
    max_rows = QUICK_ROWS_PER_FILE if quick else None
    prepared = out_dir / ("prepared_quick" if quick else "prepared")
    results = out_dir / "results"
    tag = "trial" if quick else "main"
    diag_dir = results / ("diagnose_trial" if quick else "diagnose")
    print("=" * 90)
    print("BoT-IoT -> ToN-IoT transfer-learning pipeline")
    print(f"data root : {root}")
    print(f"output    : {out_dir}")
    print(f"stages    : {stages} | quick trial: {quick} | ablations: {ablations}")
    print(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
          f"pyarrow {pa.__version__} | torch {torch.__version__} (cuda: {torch.cuda.is_available()})")
    print("=" * 90, flush=True)

    if "prepare" in stages:
        if reuse_prepared and _prepared_is_current(prepared, max_rows):
            print(f"\n[prepare] reusing {prepared} (already prepared with the same settings)", flush=True)
        else:
            missing = [p for p in _expected_files(root) if not p.is_file()]
            if missing:
                found = sorted(p.name for p in root.glob("*.csv")) if root.is_dir() else []
                raise FileNotFoundError(
                    f"{len(missing)} expected CSV file(s) are missing in {root}:\n  "
                    + "\n  ".join(p.name for p in missing)
                    + f"\nCSV files found there: {found or 'none (check DATA_ROOT)'}")
            print(f"\n[prepare] -> {prepared}", flush=True)
            argv = ["--root", str(root), "--out", str(prepared), "--ton-bytes", TON_BYTES,
                    "--bot-l2-bytes", str(BOT_L2_BYTES), "--chunk-size", str(CHUNK_SIZE)]
            if max_rows:
                argv += ["--max-rows-per-file", str(max_rows)]
            prepare_main(argv)

    if "diagnose" in stages:
        print(f"\n[diagnose] -> {diag_dir}", flush=True)
        diagnose_main(["--prepared", str(prepared), "--features", *FEATURES, "--out", str(diag_dir)])

    if "experiment" in stages:
        print(f"\n[experiment] -> {results / tag}", flush=True)
        experiment_main(_experiment_argv(prepared, results / tag, quick))
        for name in ablations:
            print(f"\n[ablation: {name}] -> {results / (tag + '_' + name)}", flush=True)
            experiment_main(_experiment_argv(prepared, results / (tag + "_" + name), quick, ABLATION_ARGS[name]()))

    print("\n" + "=" * 90)
    print("DONE. Files to look at / send back for interpretation:")
    for f in (prepared / "prepare_report.txt", diag_dir / "diagnose_report.txt",
              results / tag / "summary.txt", results / tag / "learning_curve_binary.png",
              results / tag / "learning_curve_multiclass.png", results / tag / "per_class.csv",
              *[results / (tag + "_" + a) / "summary.txt" for a in ablations],
              out_dir / "pipeline_log.txt"):
        print(f"  {f}{'' if f.exists() else '   (not produced in this run)'}")
    print("=" * 90, flush=True)


def orchestrator_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BoT-IoT -> ToN-IoT transfer learning (single file). "
                                             "Without options the SETTINGS block at the top is used.")
    ap.add_argument("--root", type=Path, default=Path(DATA_ROOT))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stages", nargs="+", choices=["prepare", "diagnose", "experiment"], default=list(STAGES))
    ap.add_argument("--quick", action="store_true", default=QUICK_TRIAL)
    ap.add_argument("--ablations", nargs="*", default=list(ABLATIONS), choices=list(ABLATION_ARGS))
    ap.add_argument("--self-test", action="store_true",
                    help="generate synthetic CSVs with the real headers in a temp folder and run everything on them")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles without UTF-8
    if args.self_test:
        tmp = Path(tempfile.mkdtemp(prefix="bot_ton_selftest_"))
        root, out_dir = write_synthetic(tmp / "raw", bot_rows_per_file=1500, ton_rows_per_file=400), tmp / "out"
        quick, reuse = True, False
    else:
        root = args.root
        out_dir = args.out or (Path(OUTPUT_DIR) if OUTPUT_DIR else root / "TL_BoT_ToN_pipeline")
        quick, reuse = args.quick, REUSE_PREPARED
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(sys.stdout, out_dir / "pipeline_log.txt")
    sys.stdout = tee
    t0 = time.time()
    try:
        print(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} #####")
        run_pipeline(root, out_dir, args.stages, quick, args.ablations, reuse_prepared=reuse)
        print(f"total time {(time.time() - t0) / 60:.1f} min")
        if args.self_test:
            print("SELF-TEST PASSED (synthetic data - the scores are meaningless, only the mechanics are tested).")
        return 0
    except FileNotFoundError as exc:
        print(f"\nERROR - input files not found. Check DATA_ROOT in the SETTINGS block.\n{exc}")
        return 1
    except Exception:
        print("\nERROR - the run stopped. Copy everything below when asking for help:\n")
        print(traceback.format_exc())
        return 1
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    sys.exit(orchestrator_main())
