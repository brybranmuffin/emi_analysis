from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from emi_analysis.collect_activations import check_float16_range, main as collect_main, select_peak_checkpoint
from emi_analysis.data_pipeline import chunk_token_ids
from emi_analysis.train import load_model_and_tokenizer, main as train_main, parse_config

from test_train import _make_run_inputs, _read_metrics, _speech


def test_peak_checkpoint_is_lowest_congressional_loss_and_earliest_on_ties():
    rows = [
        {"step": "0", "congressional_loss": "3.0", "checkpoint": ""},
        {"step": "10", "congressional_loss": "2.0", "checkpoint": "a"},
        {"step": "20", "congressional_loss": "2.5", "checkpoint": "b"},
        {"step": "30", "congressional_loss": "2.0", "checkpoint": "c"},
    ]
    step, row = select_peak_checkpoint(rows)
    assert step == 10 and row["checkpoint"] == "a"


def test_float16_guard_names_the_overflowing_layer():
    with pytest.raises(ValueError, match="layer 5.*--dtype float32"):
        check_float16_range({0: 10.0, 5: 70_000.0})
    check_float16_range({0: 10.0, 1: 60_000.0})


def _write_speeches(path: Path) -> list[dict]:
    rng = random.Random(3)
    rows = [{"speech_id": f"s{i}", "congress": 46 + i, "text": _speech(rng, n)} for i, n in enumerate([1, 4, 30, 90, 200, 12, 150])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def _reference_pooled(config, text: str) -> torch.Tensor:
    """Run every chunk of the speech alone through the base model and average its non-special tokens per layer."""
    model, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    wrap = tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks, _ = chunk_token_ids(ids, config.max_length - (2 if wrap else 0), min_chunk_tokens=1)
    total, count = None, 0
    for chunk in chunks:
        full = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id] if wrap else chunk
        with torch.no_grad():
            states = model.base_model(input_ids=torch.tensor([full]), output_hidden_states=True).hidden_states
        span = slice(1, len(full) - 1) if wrap else slice(None)
        summed = torch.stack([h[0, span].sum(0) for h in states])
        total = summed if total is None else total + summed
        count += len(chunk)
    return total / count


@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        ("bert", {"max_length": 32}),
        ("modernbert", {"max_length": 128, "packing": True}),
        ("qwen3", {"max_length": 64, "packing": True}),
    ],
)
def test_pooled_activations_equal_per_chunk_reference(tmp_path, family, overrides, tiny_qwen3_path):
    if family == "qwen3":
        overrides = {**overrides, "model_name": tiny_qwen3_path}
    config_path = _make_run_inputs(tmp_path, family, **overrides)
    speeches = _write_speeches(tmp_path / "speeches.jsonl")

    summary = collect_main(["--model", family, "--config", str(config_path), "--base", "--input", str(tmp_path / "speeches.jsonl"),
                            "--tag", "t", "--output-root", str(tmp_path / "act"), "--dtype", "float32",
                            "--block-speeches", "3", "--tokens-per-batch", "256"])

    out = Path(summary["output_dir"])
    meta = json.loads((out / "meta.json").read_text())
    config = parse_config(str(config_path))
    assert (out / "speech_ids.txt").read_text().split() == [s["speech_id"] for s in speeches]
    assert meta["status"] == "complete"
    layers = [np.load(out / f"layer_{i:02d}.npy", mmap_mode="r") for i in range(meta["n_layers"])]
    for row, speech in enumerate(speeches):
        expected = _reference_pooled(config, speech["text"])
        assert expected.shape[0] == meta["n_layers"]
        got = torch.tensor(np.stack([layer[row] for layer in layers]))
        assert torch.allclose(got, expected, atol=1e-4), f"speech {row} differs"


def test_collection_resumes_after_interruption(tmp_path):
    config_path = _make_run_inputs(tmp_path, "bert", max_length=32)
    _write_speeches(tmp_path / "speeches.jsonl")
    argv = ["--model", "bert", "--config", str(config_path), "--base", "--input", str(tmp_path / "speeches.jsonl"),
            "--tag", "t", "--output-root", str(tmp_path / "act"), "--dtype", "float32", "--block-speeches", "3"]
    out = Path(collect_main(argv)["output_dir"])
    complete = np.load(out / "layer_05.npy").copy()

    # Simulate a job killed after the first block: progress says 3 speeches done, later rows never written.
    (out / "progress.json").write_text(json.dumps({"completed_speeches": 3}), encoding="utf-8")
    for path in out.glob("layer_*.npy"):
        layer = np.load(path, mmap_mode="r+")
        layer[3:] = 0
        layer.flush()

    collect_main(argv)

    assert np.array_equal(np.load(out / "layer_05.npy"), complete)
    assert "Resuming" in (out / "collect.log").read_text()


def test_default_checkpoint_is_peak_adaptation(tmp_path):
    config_path = _make_run_inputs(tmp_path, "gpt2", seed_drift_threshold=2.0, general_loss_max_rel_increase=100.0)
    run_dir = train_main(["--config", str(config_path), "--max-steps", "4", "--checkpoint-every", "2", "--run-name", "p"])
    _write_speeches(tmp_path / "speeches.jsonl")

    summary = collect_main(["--model", "gpt2", "--config", str(config_path), "--run-dir", str(run_dir),
                            "--input", str(tmp_path / "speeches.jsonl"), "--tag", "t", "--output-root", str(tmp_path / "act")])

    rows = [r for r in _read_metrics(run_dir) if r["checkpoint"]]
    best = min(rows, key=lambda r: float(r["congressional_loss"]))
    meta = json.loads((Path(summary["output_dir"]) / "meta.json").read_text())
    assert meta["checkpoint"]["step"] == int(best["step"])
    assert Path(summary["output_dir"]).parent.name == f"step{best['step']}"
    assert np.load(Path(summary["output_dir"]) / "layer_00.npy").dtype == np.float16
