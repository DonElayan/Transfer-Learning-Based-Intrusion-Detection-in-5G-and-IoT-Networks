"""Stage 1: stream the raw CSVs into one harmonised, de-duplicated parquet per dataset.

    python -m bot_ton_tl.prepare --root "D:\\Abdallah\\Pycharm folder\\pythonProject\\ToN-IoT FULL" --out prepared

Reads only the columns in schema.py (so ToN-IoT Network_dataset_6.csv's extra 'uid'
column is irrelevant), applies the parsing policy below, removes exact duplicate
(feature-vector, label) rows and writes:

    <out>/bot.parquet, <out>/ton.parquet   canonical rows + vec_hash (for group splits)
    <out>/prepare_report.json / .txt       every count needed for a methods section

Row policy (each case is counted in the report):
  * core numeric (duration, bytes, packets) blank / unparseable / negative -> row dropped
    (this is what removes the ToN-IoT src_bytes='0.0.0.0' records)
  * Zeek '-' in duration / byte columns -> 0 (Zeek leaves them unset for one-packet flows)
  * ports: hexadecimal decoded, -1 / out of range / unparseable -> missing (row kept)
  * binary label not 0/1, blank class, or binary label contradicting the class
    (normal <-> 0) -> row dropped
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .parsing import parse_binary_label, parse_numeric, parse_port
from .schema import (BOT, CANONICAL_NUMERIC, DOMAINS, VECTOR_COLUMNS, DatasetSpec,
                     canonical_class, canonical_proto, get_spec, normalize_col)

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


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    sys.exit(main())
