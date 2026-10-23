"""Dataset schemas and the canonical (shared) flow representation.

Only fields that exist with comparable semantics in BOTH datasets are kept:

    canonical        BoT-IoT (Argus)   ToN-IoT (Zeek)            notes
    ---------        ---------------   --------------            -----
    proto            proto             proto                     tcp/udp/icmp/other
    src_port         sport             src_port                  hex decoded, -1 -> missing
    dst_port         dport             dst_port                  hex decoded, -1 -> missing
    duration         dur               duration                  seconds in both
    src_bytes        sbytes - 14*spkts src_ip_bytes  (default)   IP-level bytes incl. headers
    dst_bytes        dbytes - 14*dpkts dst_ip_bytes  (default)   IP-level bytes incl. headers
    src_pkts         spkts             src_pkts
    dst_pkts         dpkts             dst_pkts
    state            state             conn_state                different taxonomies,
                                                                 only a coarse mapping

Why src_ip_bytes and not src_bytes: Zeek's ``orig_bytes`` (ToN ``src_bytes``) counts
*payload* bytes only and is 0 for SYN scans / floods, while Argus ``sbytes`` counts the
bytes of every packet including headers and is never 0 when ``spkts > 0``. Zeek's
``orig_ip_bytes`` (ToN ``src_ip_bytes``) is the header-inclusive counterpart. The
prepare report prints the evidence (rows with packets > 0 but bytes == 0) so the choice
can be checked on the real files; ``--ton-bytes payload`` switches back.

Identifiers, addresses, timestamps, BoT-IoT's window-aggregate columns (TnBPSrcIP,
N_IN_Conn_P_DstIP, ...), the *_number encodings and all labels are never features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

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
    return re.sub(r"[^a-z0-9]+", "", str(name).replace("﻿", "").strip().lower())


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
