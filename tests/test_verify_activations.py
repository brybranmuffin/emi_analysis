from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.verify_activations import main as verify_main

from test_collect_activations import _write_speeches
from test_train import _make_run_inputs


def _collection(tmp_path: Path) -> tuple[Path, Path]:
    config_path = _make_run_inputs(tmp_path, "bert", max_length=32)
    _write_speeches(tmp_path / "speeches.jsonl")
    summary = collect_main(["--model", "bert", "--config", str(config_path), "--base", "--input", str(tmp_path / "speeches.jsonl"),
                            "--tag", "t", "--output-root", str(tmp_path / "act"), "--block-speeches", "3"])
    return Path(summary["output_dir"]), config_path


def _verify(out: Path, config_path: Path) -> dict:
    return verify_main([str(out), "--config", str(config_path), "--spot-check", "100", "--report", str(out.parent / "report.json")])


def test_complete_collection_passes_every_check(tmp_path):
    out, config_path = _collection(tmp_path)
    report = _verify(out, config_path)
    result = report["collections"][str(out)]
    assert report["ok"] is True
    assert result["checks"]["rows_written"]["ok"] is True
    assert result["checks"]["spot_check"]["speeches"] == 7


def test_unwritten_row_is_detected(tmp_path):
    out, config_path = _collection(tmp_path)
    layer = np.load(out / "layer_04.npy", mmap_mode="r+")
    layer[5] = 0
    layer.flush()
    result = _verify(out, config_path)["collections"][str(out)]
    assert result["checks"]["rows_written"]["ok"] is False
    assert "layer 4" in result["checks"]["rows_written"]["detail"]


def test_wrong_values_are_caught_by_recomputation(tmp_path):
    out, config_path = _collection(tmp_path)
    last = json.loads((out / "meta.json").read_text())["n_layers"] - 1
    layer = np.load(out / f"layer_{last:02d}.npy", mmap_mode="r+")
    layer[2] = layer[2] + np.float16(1.0)
    layer.flush()
    result = _verify(out, config_path)["collections"][str(out)]
    assert result["checks"]["rows_written"]["ok"] is True
    assert result["checks"]["spot_check"]["ok"] is False


def test_row_order_must_match_the_input_files(tmp_path):
    out, config_path = _collection(tmp_path)
    ids = (out / "speech_ids.txt").read_text().split()
    ids[0], ids[1] = ids[1], ids[0]
    (out / "speech_ids.txt").write_text("\n".join(ids) + "\n")
    result = _verify(out, config_path)["collections"][str(out)]
    assert result["checks"]["speech_ids"]["ok"] is False
