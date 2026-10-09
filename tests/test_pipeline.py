from __future__ import annotations

from emi_analysis.data_pipeline import chunk_speech_text
from emi_analysis.train import parse_config


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
