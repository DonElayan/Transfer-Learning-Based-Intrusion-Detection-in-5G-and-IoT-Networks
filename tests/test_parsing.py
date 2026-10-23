import numpy as np
import pandas as pd

from bot_ton_tl.data import assign_split, hash_uniform, holdout_split, sample_budget
from bot_ton_tl.features import Standardizer, build_features, continuous_mask
from bot_ton_tl.parsing import parse_binary_label, parse_numeric, parse_port
from bot_ton_tl.schema import canonical_proto, normalize_col


def test_parse_numeric_policies():
    raw = pd.Series(["12", " 3.5 ", "", "-", "0.0.0.0", "nan", "inf", "-7"])
    p = parse_numeric(raw)
    assert p.values[0] == 12 and p.values[1] == 3.5 and p.values[7] == -7
    assert p.blank.tolist() == [False, False, True, False, False, True, False, False]
    assert p.dash.tolist()[3]
    assert p.invalid.tolist() == [False, False, False, False, True, False, True, False]
    assert np.isnan(p.values[3])
    assert parse_numeric(raw, dash_value=0.0).values[3] == 0.0


def test_parse_port_hex_and_sentinels():
    values, counts, bad = parse_port(pd.Series(["80", "0x0050", "-1", "0X1bb", "70000", "abc", ""]))
    assert values[0] == 80 and values[1] == 80 and values[3] == 443
    assert np.isnan(values[2]) and np.isnan(values[4]) and np.isnan(values[5]) and np.isnan(values[6])
    assert counts == {"hex": 2, "sentinel_minus1": 1, "out_of_range": 1, "missing": 1, "invalid": 1}
    assert bad.tolist() == ["abc"]


def test_labels_and_names():
    assert parse_binary_label(pd.Series(["0", "1", "1.0", "x"])).tolist() == [0, 1, 1, -1]
    assert normalize_col("src_ip_bytes") == "srcipbytes"
    assert normalize_col("﻿TnP_PSrcIP") == "tnppsrcip"
    assert [canonical_proto(p) for p in ("TCP", "ipv6-icmp", "arp", "udp")] == ["tcp", "icmp", "other", "udp"]


def test_hash_split_is_deterministic_and_groups_duplicates():
    h = np.array([5, 5, 5, 99, 1234567890123], dtype=np.uint64)
    u = hash_uniform(h)
    assert ((u >= 0) & (u < 1)).all() and u[0] == u[1] == u[2]
    s1 = assign_split(h, np.ones(5))
    assert (s1 == assign_split(h, np.ones(5))).all() and s1[0] == s1[1] == s1[2]
    many = assign_split(np.arange(100_000, dtype=np.uint64), np.ones(100_000))
    assert abs((many == 2).mean() - 0.2) < 0.01 and abs((many == 1).mean() - 0.1) < 0.01
    by_file = assign_split(np.arange(10, dtype=np.uint64), np.array([1] * 5 + [7] * 5), test_files=(7,))
    assert (by_file[5:] == 2).all() and (by_file[:5] != 2).all()


def test_budget_sampling():
    labels = np.array(["a"] * 50 + ["b"] * 3 + ["c"] * 20, dtype=object)
    rng = np.random.default_rng(0)
    idx = sample_budget(labels, 10, rng, exclude=("c",))
    assert pd.Series(labels[idx]).value_counts().to_dict() == {"a": 10, "b": 3}
    fit, hold = holdout_split(labels[idx], 0.2, rng)
    assert len(set(fit) & set(hold)) == 0 and len(fit) + len(hold) == len(idx)
    assert set(labels[idx][hold]) == {"a", "b"}


def test_features_and_scaler():
    df = pd.DataFrame({
        "proto": ["tcp", "udp", "icmp", "other"], "src_port": [1234, 53, 8, np.nan], "dst_port": [80, 53, 0, np.nan],
        "duration": [0.5, 0.0, 0.01, 1.0], "src_bytes": [600, 80, 98, 60], "dst_bytes": [1200, 0, 98, 0],
        "src_pkts": [5, 1, 1, 1], "dst_pkts": [4, 0, 1, 0], "state": ["CON", "INT", "ECO", "REQ"],
    })
    X, names = build_features(df, "bot", ("volume", "shape", "proto", "ports", "state"))
    assert X.shape == (4, len(names)) and np.isfinite(X).all()
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert col["dst_svc_http"].tolist() == [1, 0, 0, 0]
    assert col["dst_port_na"].tolist() == [0, 0, 1, 1]  # icmp / other have no ports
    assert col["state_established"].tolist() == [1, 0, 0, 0]
    assert col["state_no_reply"].tolist() == [0, 1, 0, 1] and col["state_other"].tolist() == [0, 0, 1, 0]
    mask = continuous_mask(names)
    Z = Standardizer(mask).fit(X).transform(X)
    assert np.allclose(Z[:, ~mask], X[:, ~mask])  # indicators untouched
    assert np.allclose(Z[:, mask].mean(0), 0, atol=1e-5)
