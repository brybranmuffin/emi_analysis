"""EMI continued-pretraining package."""

from .config import BERT_MAX_LENGTH, GPT2_MAX_LENGTH, MIN_CHUNK_TOKENS, ModelConfig
from .data_pipeline import build_chunked_dataset, create_speech_holdout, deduplicate_speeches

__all__ = [
    "ModelConfig",
    "BERT_MAX_LENGTH",
    "GPT2_MAX_LENGTH",
    "MIN_CHUNK_TOKENS",
    "build_chunked_dataset",
    "create_speech_holdout",
    "deduplicate_speeches",
]
