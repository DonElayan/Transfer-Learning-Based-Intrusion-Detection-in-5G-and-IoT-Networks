"""End-to-end run of all three stages on the synthetic CSVs."""
import json

import pandas as pd
import pytest

from bot_ton_tl import diagnose, experiment, prepare
from bot_ton_tl.synthetic import write_synthetic


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    write_synthetic(root, bot_rows_per_file=600, ton_rows_per_file=200, seed=3)
    out = tmp_path_factory.mktemp("prepared")
    assert prepare.main(["--root", str(root), "--out", str(out), "--chunk-size", "500"]) == 0
    return out


def test_prepare_outputs(prepared):
    report = json.loads((prepared / "prepare_report.json").read_text())
    bot, ton = report["domains"]["bot"], report["domains"]["ton"]
    # corrupt src_bytes='0.0.0.0' rows (3 per file in files 1, 22, 23, plus their duplicates)
    # are the only rows dropped
    assert list(ton["dropped_by_reason"]) == ["invalid_numeric:src_alt_bytes"]
    assert ton["dropped_by_reason"]["invalid_numeric:src_alt_bytes"] >= 9
    assert ton["invalid_tokens_top"]["src_alt_bytes"][0][0] == "0.0.0.0"
    assert bot["port_parsing"]["src_port"]["hex"] > 0
    assert ton["dedup"]["rows_after"] < ton["rows_after_row_policy"]
    assert bot["bytes_evidence"]["of_which_src_bytes_eq0"] == 0
    df = pd.read_parquet(prepared / "ton.parquet")
    assert not df.duplicated(["proto", "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
                              "src_pkts", "dst_pkts", "cls"]).any()
    assert set(df["cls"]) >= {"normal", "scanning", "mitm"} and df["binary"].isin([0, 1]).all()
    assert ((df["cls"] == "normal") == (df["binary"] == 0)).all()


def test_diagnose_runs(prepared, tmp_path):
    assert diagnose.main(["--prepared", str(prepared), "--out", str(tmp_path)]) == 0
    assert "Domain classifier" in (tmp_path / "diagnose_report.txt").read_text()


def test_experiment_quick(prepared, tmp_path):
    out = tmp_path / "exp"
    experiment.main(["--prepared", str(prepared), "--out", str(out), "--quick", "--shots", "5",
                     "--unseen-classes", "mitm"])
    res = pd.read_csv(out / "results.csv")
    expected = {"source_only", "dann_zero", "target_only", "rf", "frozen", "finetune", "dann"}
    assert expected <= set(res.loc[res.task == "binary", "method"])
    assert res["macro_f1"].between(0, 1).all()
    per_class = pd.read_csv(out / "per_class.csv")
    # the unseen class is still scored for binary detection, and absent from multiclass
    assert "mitm" in set(per_class.loc[per_class.task == "binary", "cls"])
    assert "mitm" not in set(per_class.loc[per_class.task == "multiclass", "cls"])
    assert (out / "summary.txt").exists() and (out / "models" / "source_encoder_seed0.pt").exists()
