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
    train_data_path: str = "data/train.jsonl"
    heldout_data_path: str = "data/heldout/congressional.jsonl"
    general_data_path: str = "data/heldout/general.jsonl"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    seed: int = 42

    @property
    def checkpoint_every(self) -> int:
        return max(1, int(round(self.checkpoint_interval * 100)))


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
