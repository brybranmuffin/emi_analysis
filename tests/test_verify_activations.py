from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.verify_activations import main as verify_main

from test_collect_activations import _write_speeches
from test_train import _make_run_inputs


def _collection(tmp_path: Path, extra_text: str | None = None) -> tuple[Path, Path]:
    config_path = _make_run_inputs(tmp_path, "bert", max_length=32)
    speeches = tmp_path / "speeches.jsonl"
    _write_speeches(speeches)
    if extra_text:
        with speeches.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"speech_id": "extra", "congress": 99, "text": extra_text}) + "\n")
    summary = collect_main(["--model", "bert", "--config", str(config_path), "--base", "--input", str(speeches),
                            "--tag", "t", "--output-root", str(tmp_path / "act"), "--block-speeches", "3"])
    return Path(summary["output_dir"]), config_path


def _verify(out: Path, config_path: Path) -> dict:
    return verify_main([str(out), "--config", str(config_path), "--spot-check", "100", "--report", str(out.parent / "report.json")])


def _checks(out: Path, config_path: Path) -> dict:
    return _verify(out, config_path)["collections"][str(out)]["checks"]


def test_complete_collection_passes_every_check(tmp_path):
    out, config_path = _collection(tmp_path)
    report = _verify(out, config_path)
    checks = report["collections"][str(out)]["checks"]
    assert report["ok"] is True
    assert {"status", "speech_ids", "files", "token_counts", "rows_written", "row_identity", "fidelity"} <= set(checks)
    assert checks["row_identity"]["speeches"] == 7
    assert checks["fidelity"]["worst_centered_cosine"] > 0.999


def test_unwritten_row_is_detected(tmp_path):
    out, config_path = _collection(tmp_path)
    layer = np.load(out / "layer_04.npy", mmap_mode="r+")
    layer[5] = 0
    layer.flush()
    checks = _checks(out, config_path)
    assert checks["rows_written"]["ok"] is False
    assert "layer 4" in checks["rows_written"]["detail"]


def test_imprecise_layer_fails_fidelity_and_is_named(tmp_path):
    out, config_path = _collection(tmp_path)
    layer = np.load(out / "layer_03.npy", mmap_mode="r+")
    noise = np.random.default_rng(0).normal(size=layer[2].shape) * float(np.abs(layer[2]).max())
    layer[2] = (layer[2] + noise).astype(layer.dtype)
    layer.flush()
    checks = _checks(out, config_path)
    assert checks["rows_written"]["ok"] is True
    assert checks["row_identity"]["ok"] is True
    assert checks["fidelity"]["ok"] is False
    assert checks["fidelity"]["failing_layers"] == [3]


def test_misplaced_rows_fail_row_identity(tmp_path):
    out, config_path = _collection(tmp_path)
    for path in out.glob("layer_*.npy"):
        layer = np.load(path, mmap_mode="r+")
        layer[[1, 4]] = layer[[4, 1]]
        layer.flush()
    checks = _checks(out, config_path)
    assert checks["speech_ids"]["ok"] is True
    assert checks["row_identity"]["ok"] is False


def test_row_order_must_match_the_input_files(tmp_path):
    out, config_path = _collection(tmp_path)
    ids = (out / "speech_ids.txt").read_text().split()
    ids[0], ids[1] = ids[1], ids[0]
    (out / "speech_ids.txt").write_text("\n".join(ids) + "\n")
    assert _checks(out, config_path)["speech_ids"]["ok"] is False


def test_in_text_special_tokens_are_excluded_by_design(tmp_path):
    out, config_path = _collection(tmp_path, extra_text="the committee [UNK] report data study")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["special_tokens_in_text_excluded"] >= 1
    checks = _checks(out, config_path)
    assert checks["token_counts"]["ok"] is True


def test_altered_token_count_is_detected_per_speech(tmp_path):
    out, config_path = _collection(tmp_path)
    counts = np.load(out / "token_counts.npy", mmap_mode="r+")
    counts[3] += 1
    counts.flush()
    checks = _checks(out, config_path)
    assert checks["token_counts"]["ok"] is False
    assert "1 speech" in checks["token_counts"]["detail"]
