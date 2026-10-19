"""Pack several speeches into one long sequence without letting them see each other.

ModernBERT and Qwen3 read far more tokens than a typical speech contains, so training puts several
speech pieces back to back in each sequence. Each piece still behaves exactly as if it were alone:
position ids restart at 0 for every piece, attention never crosses a piece boundary (the handoff's
requirement), and for causal LM the first token of a piece is never predicted from the previous one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def pack_sequences(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Best-fit-decreasing bin packing: group piece indices into packs of at most ``capacity`` tokens.

    Longest pieces are placed first; each piece goes into the open pack whose free space it fills most
    tightly, so packs end up almost full.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(lengths) and lengths.max() > capacity:
        raise ValueError(f"A piece of {int(lengths.max())} tokens is longer than the pack capacity {capacity}.")

    packs: list[list[int]] = []
    open_by_space: list[list[int]] = [[] for _ in range(capacity + 1)]  # free space -> open pack ids
    has_space = np.zeros(capacity + 1, dtype=bool)
    for index in np.argsort(-lengths, kind="stable"):
        length = int(lengths[index])
        space = length + int(has_space[length:].argmax())
        if has_space[space]:
            pack = open_by_space[space].pop()
            if not open_by_space[space]:
                has_space[space] = False
        else:
            pack, space = len(packs), capacity
            packs.append([])
        packs[pack].append(int(index))
        left = space - length
        if left > 0:
            open_by_space[left].append(pack)
            has_space[left] = True
    return packs


def build_packed_batch(packs: Sequence[Sequence[Sequence[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Lay out packs (each a list of token-id pieces) as one padded batch.

    Keys starting with ``_`` are bookkeeping, not model inputs: ``_doc_ids`` numbers the pieces within
    each row (trailing padding gets its own id), ``_real`` marks real tokens, ``_doc_start`` marks the
    first token of each piece.
    """
    width = max(sum(len(piece) for piece in pack) for pack in packs)
    shape = (len(packs), width)
    input_ids = torch.full(shape, pad_id, dtype=torch.long)
    position_ids = torch.zeros(shape, dtype=torch.long)
    doc_ids = torch.zeros(shape, dtype=torch.long)
    real = torch.zeros(shape, dtype=torch.bool)
    doc_start = torch.zeros(shape, dtype=torch.bool)
    for row, pack in enumerate(packs):
        start = 0
        for doc, piece in enumerate(pack):
            end = start + len(piece)
            input_ids[row, start:end] = torch.as_tensor(piece, dtype=torch.long)
            position_ids[row, start:end] = torch.arange(len(piece))
            doc_ids[row, start:end] = doc
            real[row, start:end] = True
            doc_start[row, start] = True
            start = end
        # Padding forms its own segment with restarted positions, so no real token can attend to it.
        position_ids[row, start:] = torch.arange(width - start)
        doc_ids[row, start:] = len(pack)
    return {"input_ids": input_ids, "position_ids": position_ids, "_doc_ids": doc_ids, "_real": real, "_doc_start": doc_start}


def model_inputs(batch: dict[str, torch.Tensor], model, objective: str, device: torch.device, labels: bool = True) -> dict:
    """Move a batch to ``device`` and add whatever attention masking the model needs.

    Unpacked batches (BERT, GPT-2) pass their padding mask through unchanged. Packed encoder batches get
    per-speech masks for ModernBERT's global and sliding-window layers. Packed decoder batches get no
    mask: transformers builds the per-speech causal mask from the restarted ``position_ids`` (cache disabled).
    """
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if not k.startswith("_") and (labels or k != "labels")}
    if "_doc_ids" not in batch:
        return inputs
    if objective != "mlm":
        # transformers only derives the per-speech mask from position_ids when no KV cache exists; decoders
        # such as Qwen3 create one by default, which would silently let speeches attend to each other.
        inputs["use_cache"] = False
        return inputs

    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
        packed_sequence_mask_function,
    )

    doc_ids = batch["_doc_ids"].to(device)
    kwargs = {
        "config": model.config,
        "inputs_embeds": torch.empty((*doc_ids.shape, 1), dtype=model.dtype, device=device),  # shape/dtype/device only
        "attention_mask": None,
        "and_mask_function": packed_sequence_mask_function(doc_ids),
        "allow_is_bidirectional_skip": False,
    }
    inputs["attention_mask"] = {
        "full_attention": create_bidirectional_mask(**kwargs),
        "sliding_attention": create_bidirectional_sliding_window_mask(**kwargs),
    }
    return inputs
