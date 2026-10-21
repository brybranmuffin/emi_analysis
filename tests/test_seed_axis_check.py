from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from emi_analysis.clean_text_check import main as clean_main
from emi_analysis.seed_axis_check import main as axis_main, seed_axis_metrics
from emi_analysis.train import main as train_main

from test_train import _make_run_inputs, _read_metrics


def _clusters(n: int = 10, dim: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    centre_ev, centre_int = torch.zeros(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64)
    centre_ev[0], centre_int[1] = 1.0, 1.0
    noise = lambda: 0.1 * torch.randn(n, dim, generator=gen, dtype=torch.float64)  # noqa: E731
    return centre_ev + noise(), centre_int + noise()


def test_unchanged_embeddings_score_as_perfectly_stable():
    ev, intu = _clusters()
    m = seed_axis_metrics(ev, intu, ev.clone(), intu.clone())
    assert m["mean_drift"] == pytest.approx(0.0, abs=1e-12)
    assert m["axis_cosine"] == pytest.approx(1.0)
    assert m["relational_spearman"] == pytest.approx(1.0)
    assert m["axis_order_spearman"] == pytest.approx(1.0)
    assert m["separation_accuracy"] == 1.0


def test_shared_shift_inflates_raw_drift_but_leaves_the_axis_in_place():
    ev, intu = _clusters()
    shift = torch.zeros(16, dtype=torch.float64)
    shift[5] = 2.0  # every embedding moves the same way
    m = seed_axis_metrics(ev, intu, ev + shift, intu + shift)
    assert m["mean_drift"] > 0.3
    assert m["axis_cosine"] == pytest.approx(1.0)
    assert m["separation_accuracy"] == 1.0


def test_swapped_poles_reverse_the_axis():
    ev, intu = _clusters()
    m = seed_axis_metrics(ev, intu, intu.clone(), ev.clone())
    # The axis direction flips; word order *relative to each group's own pole* is preserved by a pure swap.
    assert m["axis_cosine"] == pytest.approx(-1.0, abs=0.05)


def test_collapsed_poles_lose_separation():
    ev, intu = _clusters()
    mixed_ev, mixed_int = _clusters(seed=1)
    merged = torch.cat([mixed_ev, mixed_int])[torch.randperm(20, generator=torch.Generator().manual_seed(2))]
    m = seed_axis_metrics(ev, intu, merged[:10], merged[10:])
    assert m["separation_accuracy"] < 0.8
    assert m["separation_accuracy_step0"] == 1.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_axis_check_reproduces_training_drift_and_merges_the_adaptation_table(tmp_path):
    config_path = _make_run_inputs(tmp_path, "gpt2", seed_drift_threshold=2.0, general_loss_max_rel_increase=100.0)
    run_dir = train_main(["--config", str(config_path), "--max-steps", "4", "--checkpoint-every", "2", "--run-name", "a"])
    out = tmp_path / "out"
    clean_main(["--model", "gpt2", "--config", str(config_path), "--run-dir", str(run_dir), "--all-checkpoints",
                "--output-dir", str(out / "clean")])

    summary = axis_main(["--model", "gpt2", "--config", str(config_path), "--run-dir", str(run_dir),
                         "--clean-text-dir", str(out / "clean"), "--output-dir", str(out / "axis")])

    rows = {int(r["step"]): r for r in _read_csv(Path(summary["outputs"]["csv"]))}
    metrics = {int(r["step"]): r for r in _read_metrics(run_dir)}
    assert sorted(rows) == [0, 2, 4]
    for step in (2, 4):
        # Raw drift is recomputed from the saved checkpoint and matches what training logged.
        assert float(rows[step]["mean_drift"]) == pytest.approx(float(metrics[step]["mean_seed_drift"]), abs=1e-4)
        assert rows[step]["matches_training_log"] == "True"
    assert float(rows[0]["axis_cosine"]) == pytest.approx(1.0)
    # The combined table carries congressional loss from training and clean general loss from the clean-text check.
    assert float(rows[4]["congressional"]) == pytest.approx(float(metrics[4]["congressional_loss"]))
    assert rows[4]["general_clean"] != ""
    assert rows[4]["verdict"] in {"stable", "watch", "moved"}
