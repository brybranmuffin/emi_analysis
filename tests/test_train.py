from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest
import torch

from emi_analysis.train import (
    LengthGroupedBatchSampler,
    checkpoint_schedule,
    main as train_main,
    mask_tokens_for_mlm,
    parse_slurm_time,
)
from emi_analysis.verification import check_stop_conditions

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = REPO_ROOT / ".cache" / "huggingface"
TINY_MODELS = {
    "bert": "hf-internal-testing/tiny-random-BertForMaskedLM",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
}


# ---------------------------------------------------------------- unit tests


def test_checkpoint_schedule_every_ten_percent():
    assert checkpoint_schedule(1000, interval=0.10) == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def test_checkpoint_schedule_short_run_has_no_step_zero_or_duplicates():
    assert checkpoint_schedule(5, interval=0.10) == [1, 2, 3, 4, 5]


def test_checkpoint_schedule_fixed_interval_always_ends_on_final_step():
    assert checkpoint_schedule(5, every=2) == [2, 4, 5]


def _mlm_fixture():
    rows, cols = 64, 50
    input_ids = torch.randint(1000, 2000, (rows, cols))
    input_ids[:, 0] = 101
    input_ids[:, 39] = 102
    input_ids[::2, 40:] = 0  # padding on every other row
    input_ids[1::2, 49] = 102
    maskable = (input_ids >= 1000) & (input_ids < 2000)
    return input_ids, maskable


def test_mlm_masking_only_selects_maskable_positions():
    input_ids, maskable = _mlm_fixture()
    original = input_ids.clone()
    masked, labels = mask_tokens_for_mlm(input_ids, maskable, 103, 2000, 0.15, torch.Generator().manual_seed(0))

    assert torch.equal(input_ids, original)
    assert torch.all(labels[~maskable] == -100)
    selected = labels != -100
    assert torch.equal(labels[selected], original[selected])
    assert torch.equal(masked[~selected], original[~selected])
    rate = selected.sum().item() / maskable.sum().item()
    assert 0.12 < rate < 0.18


def test_mlm_masking_hides_most_selected_tokens_from_the_model():
    input_ids, maskable = _mlm_fixture()
    masked, labels = mask_tokens_for_mlm(input_ids, maskable, 103, 2000, 0.15, torch.Generator().manual_seed(0))
    selected = labels != -100
    n = selected.sum().item()
    replaced_with_mask = (masked[selected] == 103).sum().item() / n
    left_unchanged = (masked[selected] == input_ids[selected]).sum().item() / n
    assert 0.74 < replaced_with_mask < 0.86
    assert left_unchanged < 0.17


def test_length_grouped_sampler_covers_every_chunk_once_per_epoch():
    sampler = LengthGroupedBatchSampler([10] * 37 + [50] * 40, batch_size=8, seed=0)
    batches = sampler.epoch_batches(0)
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(77))
    assert len(batches) == len(sampler) == 10
    assert sampler.epoch_batches(1) != batches


def test_length_grouped_sampler_batches_similar_lengths():
    lengths = [10] * 40 + [50] * 40
    sampler = LengthGroupedBatchSampler(lengths, batch_size=8, seed=0, megabatch_mult=100)
    for batch in sampler.epoch_batches(0):
        assert len({lengths[i] for i in batch}) == 1


@pytest.mark.parametrize(
    ("current_general", "expect_flag"),
    [(3.9, True), (3.3, False)],
)
def test_stop_condition_on_general_loss_rise(current_general, expect_flag):
    flags = check_stop_conditions(
        baseline={"general_loss": 3.0, "mean_seed_drift": 0.0},
        previous={"general_loss": 3.0, "mean_seed_drift": 0.0},
        current={"general_loss": current_general, "mean_seed_drift": 0.0},
        general_rel_increase=0.20,
        drift_threshold=0.10,
    )
    assert any("general" in f for f in flags) is expect_flag


@pytest.mark.parametrize(
    ("previous_drift", "current_drift", "expect_flag"),
    [(0.12, 0.15, True), (0.16, 0.15, False), (0.01, 0.05, False), (0.12, "", False)],
)
def test_stop_condition_on_large_climbing_seed_drift(previous_drift, current_drift, expect_flag):
    flags = check_stop_conditions(
        baseline={"general_loss": 3.0, "mean_seed_drift": 0.0},
        previous={"general_loss": 3.0, "mean_seed_drift": previous_drift},
        current={"general_loss": 3.0, "mean_seed_drift": current_drift},
        general_rel_increase=0.20,
        drift_threshold=0.10,
    )
    assert any("drift" in f for f in flags) is expect_flag


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("1-02:03:04", 93784), ("02:03:04", 7384), ("03:04", 184), ("UNLIMITED", None), ("", None)],
)
def test_parse_slurm_time(value, seconds):
    assert parse_slurm_time(value) == seconds


# --------------------------------------------------------- integration tests

WORDS = "the committee bill evidence data study budget nation report people district believe feel heart vote law".split()


def _speech(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _make_run_inputs(tmp_path: Path, family: str, **overrides) -> Path:
    rng = random.Random(0)
    train_rows = [{"speech_id": f"t{i}", "congress": 46 + i % 3, "text": _speech(rng, rng.randint(3, 120))} for i in range(30)]
    _write_jsonl(tmp_path / "data" / "train.jsonl", train_rows)
    _write_jsonl(tmp_path / "data" / "heldout" / "congressional.jsonl",
                 [{"speech_id": f"h{i}", "congress": 46, "text": _speech(rng, 60)} for i in range(6)])
    _write_jsonl(tmp_path / "data" / "heldout" / "general.jsonl",
                 [{"id": f"g{i}", "text": _speech(rng, 50)} for i in range(6)])
    lexicons = tmp_path / "data" / "lexicons"
    lexicons.mkdir(parents=True)
    (lexicons / "evidence_lexicon.csv").write_text("evidence_keywords\nevidence\ndata\nstudy\n", encoding="utf-8")
    (lexicons / "intuition_lexicon.csv").write_text("intuition_keywords\nbelieve\nfeel\nheart\n", encoding="utf-8")

    config = {
        "model_family": family,
        "model_name": TINY_MODELS[family],
        "max_length": 32,
        "objective": "mlm" if family == "bert" else "clm",
        "epochs": 1.0,
        "checkpoint_interval": 0.10,
        "train_data_path": str(tmp_path / "data" / "train.jsonl"),
        "heldout_data_path": str(tmp_path / "data" / "heldout" / "congressional.jsonl"),
        "general_data_path": str(tmp_path / "data" / "heldout" / "general.jsonl"),
        "evidence_lexicon_path": str(lexicons / "evidence_lexicon.csv"),
        "intuition_lexicon_path": str(lexicons / "intuition_lexicon.csv"),
        "output_dir": str(tmp_path / "checkpoints" / family),
        "local_cache_dir": str(MODEL_CACHE),
        "datasets_cache_dir": str(tmp_path / "datasets_cache"),
        "batch_size": 4,
        "min_chunk_tokens": 4,
        "eval_max_chunks": 8,
        "learning_rate": 1e-3,
        "log_every": 1,
        "num_proc": 1,
        "seed": 0,
        **overrides,
    }
    config_path = tmp_path / f"{family}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("family", ["bert", "gpt2"])
def test_training_run_records_measured_metrics_and_checkpoints(tmp_path, family):
    # Tiny random models drift a lot at this learning rate; loosen the stop limits so the run completes.
    config_path = _make_run_inputs(tmp_path, family, seed_drift_threshold=2.0, general_loss_max_rel_increase=100.0)

    run_dir = train_main(["--config", str(config_path), "--max-steps", "4", "--checkpoint-every", "2", "--run-name", "t"])

    assert run_dir == tmp_path / "checkpoints" / family / "run-t"
    rows = _read_metrics(run_dir)
    assert [r["step"] for r in rows] == ["0", "2", "4"]
    for row in rows:
        assert row["model"] == family
        assert math.isfinite(float(row["congressional_loss"])) and float(row["congressional_loss"]) > 0
        assert math.isfinite(float(row["general_loss"])) and float(row["general_loss"]) > 0
        assert row["loss_metric"] == ("mlm_loss" if family == "bert" else "perplexity")
        assert float(row["max_seed_drift"]) >= float(row["mean_seed_drift"])
    if family == "gpt2":
        assert float(rows[0]["congressional_loss"]) > 1.0
    assert float(rows[0]["mean_seed_drift"]) == 0.0
    assert float(rows[-1]["mean_seed_drift"]) > 0.0
    assert rows[0]["congressional_loss"] != rows[-1]["congressional_loss"]

    for step in (2, 4):
        checkpoint = run_dir / f"checkpoint-{step}"
        assert (checkpoint / "config.json").exists()
        info = json.loads((checkpoint / "checkpoint_info.json").read_text())
        assert info["step"] == step
        assert info["total_steps"] == 4

    run_config = json.loads((run_dir / "run_config.json").read_text())
    stats = run_config["data"]["train"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODELS[family], cache_dir=str(MODEL_CACHE))
    texts = [json.loads(line)["text"] for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]
    assert stats["tokens_in"] == sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    assert stats["tokens_in"] == stats["tokens_out"] + stats["tokens_dropped"]
    assert stats["max_chunk_length"] <= 32

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["steps_completed"] == 4
    log_text = (run_dir / "train.log").read_text()
    assert str(run_dir / "checkpoint-2") in log_text and str(run_dir / "checkpoint-4") in log_text


def test_training_stops_and_exits_nonzero_when_a_flag_fires(tmp_path):
    config_path = _make_run_inputs(tmp_path, "gpt2", seed_drift_threshold=0.0)

    with pytest.raises(SystemExit) as excinfo:
        train_main(["--config", str(config_path), "--max-steps", "4", "--checkpoint-every", "2", "--run-name", "s"])

    assert excinfo.value.code == 3
    run_dir = tmp_path / "checkpoints" / "gpt2" / "run-s"
    rows = _read_metrics(run_dir)
    assert [r["step"] for r in rows] == ["0", "2"]
    assert "drift" in rows[-1]["flags"]
    assert json.loads((run_dir / "run_summary.json").read_text())["status"] == "stopped_on_flag"


def test_training_refuses_to_run_without_prepared_training_data(tmp_path):
    config_path = _make_run_inputs(tmp_path, "bert")
    (tmp_path / "data" / "train.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        train_main(["--config", str(config_path), "--max-steps", "2", "--run-name", "m"])


def test_training_refuses_to_run_without_seed_lexicons(tmp_path):
    config_path = _make_run_inputs(tmp_path, "bert")
    (tmp_path / "data" / "lexicons" / "evidence_lexicon.csv").unlink()
    with pytest.raises(FileNotFoundError, match="lexicon"):
        train_main(["--config", str(config_path), "--max-steps", "2", "--run-name", "m"])
