from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = REPO_ROOT / ".cache" / "huggingface"


@pytest.fixture(scope="session")
def tiny_qwen3_path(tmp_path_factory) -> str:
    """A 2-layer random Qwen3 with the real Qwen3-0.6B-Base tokenizer (no tiny Qwen3 exists on the Hub)."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    path = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", cache_dir=str(MODEL_CACHE))
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_qwen3_bf16_path(tiny_qwen3_path, tmp_path_factory) -> str:
    """The tiny Qwen3 stored in bfloat16, like the real Qwen3-0.6B-Base checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = tmp_path_factory.mktemp("tiny-qwen3-bf16")
    AutoModelForCausalLM.from_pretrained(tiny_qwen3_path).to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(tiny_qwen3_path).save_pretrained(path)
    return str(path)
