"""Small synthetic stand-ins for the 27 raw CSVs, with the real headers and quirks.

Used by the tests and for a dry run of the whole pipeline without the real data:

    python -m bot_ton_tl.synthetic --out synthetic_raw

Quirks reproduced: BoT-IoT hexadecimal ICMP 'ports' and -1 ARP ports, header-inclusive
Argus byte counts, ToN-IoT payload vs IP byte columns, Zeek '-' durations, the
src_bytes='0.0.0.0' records in Network_dataset_1/22/23, the extra 'uid' column in
Network_dataset_6 and exact duplicate rows. Numbers are invented - only the format is real.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("synthetic_raw"))
    ap.add_argument("--bot-rows", type=int, default=2000)
    ap.add_argument("--ton-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    print(f"wrote {write_synthetic(args.out, args.bot_rows, args.ton_rows, args.seed).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
