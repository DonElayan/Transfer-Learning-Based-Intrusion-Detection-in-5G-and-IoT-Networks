"""Vectorised, policy-explicit parsing of raw CSV tokens.

Every parser returns the parsed float64 values *and* boolean masks describing what was
seen, so the caller decides (and reports) what happens to blanks, Zeek "-" values,
hexadecimal ports, sentinels and invalid tokens such as ToN-IoT's src_bytes='0.0.0.0'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
