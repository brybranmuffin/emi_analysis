from __future__ import annotations

import random

import pytest
import torch

from emi_analysis.config import ModelConfig
from emi_analysis.packing import build_packed_batch, model_inputs, pack_sequences
from emi_analysis.train import make_batch, split_weight_decay

from conftest import MODEL_CACHE

TINY_MODERNBERT = "hf-internal-testing/tiny-random-ModernBertForMaskedLM"


def _config(**overrides) -> ModelConfig:
    base = dict(model_name="x", model_family="qwen3", max_length=16, checkpoint_interval=0.1, epochs=1.0,
                objective="clm", tokenizer_name="x", output_dir="x", packing=True)
    return ModelConfig(**{**base, **overrides})


# ------------------------------------------------------------------ bin packing


def test_pack_sequences_places_every_piece_once_within_capacity():
    lengths = [300, 1200, 50, 2048, 700, 700, 33, 1900, 140]
    packs = pack_sequences(lengths, capacity=2048)
    placed = sorted(i for pack in packs for i in pack)
    assert placed == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in pack) <= 2048 for pack in packs)


def test_pack_sequences_fills_packs_tightly():
    rng = random.Random(0)
    lengths = [rng.randint(32, 600) for _ in range(5000)]
    packs = pack_sequences(lengths, capacity=2048)
    fill = sum(lengths) / (len(packs) * 2048)
    assert fill > 0.98


def test_pack_sequences_rejects_piece_longer_than_capacity():
    with pytest.raises(ValueError, match="longer than"):
        pack_sequences([100, 3000], capacity=2048)


# ------------------------------------------------------------------ packed batches


def test_packed_clm_batch_restarts_positions_and_never_predicts_across_speeches(tiny_qwen3_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen3_path)
    pad = tokenizer.pad_token_id
    packs = [[[11, 12, 13], [14, 15]], [[21, 22, 23, 24, 25, 26, 27, 28]]]
    batch = make_batch(packs, "clm", tokenizer, _config(max_length=8), torch.Generator().manual_seed(0))

    assert batch["input_ids"].tolist() == [[11, 12, 13, 14, 15, pad, pad, pad], [21, 22, 23, 24, 25, 26, 27, 28]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7]]
    # Each label is the target for the prediction made at the previous position; a speech's first token
    # (predicted from the end of the previous speech) and padding are never scored.
    assert batch["labels"].tolist() == [[-100, 12, 13, -100, 15, -100, -100, -100], [-100, 22, 23, 24, 25, 26, 27, 28]]
    assert int(batch["_real"].sum()) == 13


def test_packed_mlm_batch_never_masks_special_tokens_or_padding():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODERNBERT, cache_dir=str(MODEL_CACHE))
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    speech = lambda n, start: [cls] + list(range(start, start + n)) + [sep]  # noqa: E731
    packs = [[speech(40, 1000), speech(30, 2000)], [speech(60, 3000)]]
    batch = make_batch(packs, "mlm", tokenizer, _config(max_length=80, objective="mlm", mlm_probability=0.3),
                       torch.Generator().manual_seed(0))

    original = build_packed_batch(packs, tokenizer.pad_token_id)["input_ids"]
    special_or_pad = torch.isin(original, torch.tensor([cls, sep])) | ~batch["_real"]
    assert torch.all(batch["labels"][special_or_pad] == -100)
    assert torch.equal(batch["input_ids"][special_or_pad], original[special_or_pad])
    rate = (batch["labels"] != -100).sum().item() / (~special_or_pad).sum().item()
    assert 0.2 < rate < 0.4


# ------------------------------------------------- speech isolation inside a real model


def _isolation_check(model, objective, pad_id, speech_a, speech_b, other_a):
    """Return (B's logits packed after A, B's logits alone, B's logits packed after a different A, naive concat)."""
    model.eval()
    with torch.no_grad():
        def packed(first):
            batch = build_packed_batch([[first, speech_b]], pad_id)
            out = model(**model_inputs(batch, model, objective, torch.device("cpu"))).logits
            return out[0, len(first) : len(first) + len(speech_b)]

        alone = model(input_ids=torch.tensor([speech_b])).logits[0]
        naive = model(input_ids=torch.tensor([speech_a + speech_b])).logits[0, len(speech_a) :]
        return packed(speech_a), alone, packed(other_a), naive


def test_modernbert_speech_is_isolated_inside_a_pack():
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(TINY_MODERNBERT, cache_dir=str(MODEL_CACHE), attn_implementation="sdpa")
    rng = random.Random(0)
    # Longer than the tiny model's 64-token local-attention window, so sliding-window layers are exercised.
    a, b, other = ([rng.randrange(1000, 9000) for _ in range(n)] for n in (100, 90, 100))
    in_pack, alone, after_other, naive = _isolation_check(model, "mlm", 50283, a, b, other)

    assert torch.allclose(in_pack, alone, atol=1e-4)
    assert torch.allclose(in_pack, after_other, atol=1e-5)
    assert not torch.allclose(naive, alone, atol=1e-2), "naive concatenation should leak; the check is not sensitive"


def test_qwen3_speech_is_isolated_inside_a_pack(tiny_qwen3_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_qwen3_path, attn_implementation="sdpa")
    rng = random.Random(0)
    a, b, other = ([rng.randrange(1000, 9000) for _ in range(n)] for n in (100, 90, 100))
    in_pack, alone, after_other, naive = _isolation_check(model, "clm", 151643, a, b, other)

    assert torch.allclose(in_pack, alone, atol=1e-4)
    assert torch.allclose(in_pack, after_other, atol=1e-5)
    assert not torch.allclose(naive, alone, atol=1e-2), "naive concatenation should leak; the check is not sensitive"


# ------------------------------------------------------------------ weight decay


def _old_name_rule(model) -> tuple[list[str], list[str]]:
    no = [n for n, _ in model.named_parameters() if n.endswith("bias") or "LayerNorm" in n or "ln_" in n]
    return [n for n, _ in model.named_parameters() if n not in no], no


@pytest.mark.parametrize("name", ["hf-internal-testing/tiny-random-BertForMaskedLM", "hf-internal-testing/tiny-random-GPT2LMHeadModel"])
def test_weight_decay_grouping_unchanged_for_bert_and_gpt2(name):
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

    cls = AutoModelForMaskedLM if "Bert" in name else AutoModelForCausalLM
    model = cls.from_pretrained(name, cache_dir=str(MODEL_CACHE))
    assert split_weight_decay(model) == _old_name_rule(model)


def test_weight_decay_excludes_modernbert_norm_weights():
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(TINY_MODERNBERT, cache_dir=str(MODEL_CACHE))
    decay, no_decay = split_weight_decay(model)
    assert any("norm" in n for n in no_decay)
    assert not any("norm" in n for n in decay)
