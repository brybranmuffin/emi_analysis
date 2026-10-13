from __future__ import annotations

from collections import Counter

from emi_analysis.data_pipeline import (
    chunk_speech_text,
    chunk_token_ids,
    congress_from_year,
    dedup_key,
    stratified_holdout_ids,
)
from emi_analysis.train import parse_config


def test_chunk_token_ids_splits_long_speech_into_full_windows_without_loss():
    # Handoff example: a 1400-token speech at window 512 becomes 512, 512, 376.
    chunks, dropped = chunk_token_ids(list(range(1400)), window=512, min_chunk_tokens=32)
    assert [len(c) for c in chunks] == [512, 512, 376]
    assert dropped == 0
    assert [t for c in chunks for t in c] == list(range(1400))


def test_chunk_token_ids_drops_short_trailing_chunk_and_counts_it():
    chunks, dropped = chunk_token_ids(list(range(530)), window=512, min_chunk_tokens=32)
    assert [len(c) for c in chunks] == [512]
    assert dropped == 18


def test_chunk_token_ids_drops_speech_shorter_than_minimum():
    chunks, dropped = chunk_token_ids(list(range(20)), window=512, min_chunk_tokens=32)
    assert chunks == []
    assert dropped == 20


def test_chunk_token_ids_legacy_truncation_keeps_only_first_window():
    chunks, dropped = chunk_token_ids(list(range(1400)), window=512, min_chunk_tokens=32, truncate=True)
    assert [len(c) for c in chunks] == [512]
    assert dropped == 888
    short_chunks, short_dropped = chunk_token_ids(list(range(20)), window=512, min_chunk_tokens=32, truncate=True)
    assert [len(c) for c in short_chunks] == [20]
    assert short_dropped == 0


def test_congress_from_year_matches_emi_convention():
    assert congress_from_year(1879) == 46
    assert congress_from_year(1880) == 46
    assert congress_from_year(1881) == 47
    assert congress_from_year(2022) == 117


def test_dedup_key_ignores_case_punctuation_and_whitespace():
    assert dedup_key("Mr. Speaker, I yield back.") == dedup_key("mr speaker   i yield back")
    assert dedup_key("Mr. Speaker, I yield back.") != dedup_key("Mr. Speaker, I object.")


def test_stratified_holdout_allocates_proportionally_per_congress():
    ids = [f"s{i}" for i in range(1000)]
    congresses = [46] * 100 + [47] * 300 + [48] * 600
    congress_of = dict(zip(ids, congresses))

    held = stratified_holdout_ids(ids, congresses, holdout_count=100, seed=7)

    assert len(held) == 100
    assert len(set(held)) == 100
    assert Counter(congress_of[s] for s in held) == {46: 10, 47: 30, 48: 60}


def test_stratified_holdout_covers_small_congresses():
    ids = [f"s{i}" for i in range(10_003)]
    congresses = [46] * 3 + [47] * 10_000
    congress_of = dict(zip(ids, congresses))

    held = stratified_holdout_ids(ids, congresses, holdout_count=100, seed=7)

    assert len(held) == 100
    assert Counter(congress_of[s] for s in held)[46] == 1


def test_stratified_holdout_with_fewer_slots_than_congresses():
    ids = [f"s{i}" for i in range(200)]
    congresses = [46 + i // 20 for i in range(200)]  # 10 Congresses x 20 speeches
    congress_of = dict(zip(ids, congresses))

    held = stratified_holdout_ids(ids, congresses, holdout_count=5, seed=3)

    assert len(held) == 5
    assert len({congress_of[s] for s in held}) == 5


def test_stratified_holdout_is_deterministic_for_a_seed():
    ids = [f"s{i}" for i in range(500)]
    congresses = [46 + (i % 5) for i in range(500)]
    first = stratified_holdout_ids(ids, congresses, holdout_count=50, seed=1)
    again = stratified_holdout_ids(ids, congresses, holdout_count=50, seed=1)
    other = stratified_holdout_ids(ids, congresses, holdout_count=50, seed=2)
    assert first == again
    assert set(first) != set(other)


def test_chunking_preserves_tokens_without_truncation():
    text = " ".join(f"token{i}" for i in range(700))
    chunks = chunk_speech_text(text, max_tokens=256)
    total = sum(len(chunk.split()) for chunk in chunks)
    assert total == 700
    assert all(len(chunk.split()) <= 256 for chunk in chunks)


def test_short_trailing_chunk_is_merged_or_dropped_cleanly():
    text = " ".join(f"w{i}" for i in range(270))
    chunks = chunk_speech_text(text, max_tokens=256)
    assert sum(len(chunk.split()) for chunk in chunks) == 270
    assert len(chunks) >= 1


def test_parse_config_accepts_training_metadata_fields():
    config = {
        "model_family": "bert",
        "model_name": "bert-base-uncased",
        "max_length": 256,
        "objective": "mlm",
        "epochs": 2.5,
        "checkpoint_interval": 0.10,
        "train_data_path": "data/train.jsonl",
        "heldout_data_path": "data/heldout/congressional.jsonl",
        "general_data_path": "data/heldout/general.jsonl",
        "output_dir": "checkpoints/bert",
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "seed": 42,
    }

    parsed = parse_config(config)
    assert parsed.model_family == "bert"
    assert parsed.train_data_path == "data/train.jsonl"
    assert parsed.batch_size == 32
    assert parsed.seed == 42
