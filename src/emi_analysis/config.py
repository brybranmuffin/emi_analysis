from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BERT_MAX_LENGTH = 256
GPT2_MAX_LENGTH = 512
MIN_CHUNK_TOKENS = 32

DATASET_URL = "https://zenodo.org/records/11127530/files/saroyehun%2FEvidenceMinusIntuition-v0.2.zip"
LOCAL_CACHE_DIR = Path(".cache/huggingface")
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HELDOUT_DIR = DATA_DIR / "heldout"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    model_family: Literal["bert", "gpt2"]
    max_length: int
    checkpoint_interval: float
    epochs: float
    objective: str
    tokenizer_name: str
    output_dir: str
    local_cache_dir: str = str(LOCAL_CACHE_DIR)
    datasets_cache_dir: str = str(LOCAL_CACHE_DIR / "datasets")
    train_data_path: str = "data/train.jsonl"
    heldout_data_path: str = "data/heldout/congressional.jsonl"
    general_data_path: str = "data/heldout/general.jsonl"
    evidence_lexicon_path: str = "data/lexicons/evidence_lexicon.csv"
    intuition_lexicon_path: str = "data/lexicons/intuition_lexicon.csv"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    seed: int = 42
    min_chunk_tokens: int = MIN_CHUNK_TOKENS
    mlm_probability: float = 0.15
    legacy_truncation: bool = False
    # Held-out congressional chunks scored at every checkpoint (a fixed random subset, for speed).
    eval_max_chunks: int = 10_000
    log_every: int = 100
    num_proc: int | None = None
    # Stop conditions from the handoff. "Sharp" and "large" are not defined there; these are starting values.
    # Flags are always logged and recorded; stop_on_flag decides whether a flag also ends the run.
    general_loss_max_rel_increase: float = 0.20
    seed_drift_threshold: float = 0.10
    stop_on_flag: bool = False


BERT_CONFIG = ModelConfig(
    model_name="google-bert/bert-base-uncased",
    model_family="bert",
    max_length=BERT_MAX_LENGTH,
    checkpoint_interval=0.10,
    epochs=2.5,
    objective="mlm",
    tokenizer_name="google-bert/bert-base-uncased",
    output_dir="checkpoints/bert",
    local_cache_dir=str(LOCAL_CACHE_DIR),
)

GPT2_CONFIG = ModelConfig(
    model_name="gpt2",
    model_family="gpt2",
    max_length=GPT2_MAX_LENGTH,
    checkpoint_interval=0.10,
    epochs=1.5,
    objective="clm",
    tokenizer_name="gpt2",
    output_dir="checkpoints/gpt2",
    local_cache_dir=str(LOCAL_CACHE_DIR),
)
