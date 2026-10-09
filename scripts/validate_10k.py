#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emi_analysis.config import BERT_MAX_LENGTH, GPT2_MAX_LENGTH
from emi_analysis.data_pipeline import build_chunked_dataset, create_speech_holdout, deduplicate_speeches


def load_jsonl(path: str | Path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the speech chunking pipeline on a 10k-speech subset.")
    parser.add_argument("--input", type=str, required=True, help="Path to a JSONL file with speech rows containing speech_id, congress, text.")
    parser.add_argument("--max-length", type=int, default=256, help="Model max length used for chunking. Defaults to BERT's 256.")
    parser.add_argument("--holdout-count", type=int, default=20000, help="Target held-out count for speech split.")
    args = parser.parse_args()

    speeches = [
        {"speech_id": row["speech_id"], "congress": int(row["congress"]), "text": row["text"]}
        for row in load_jsonl(args.input)
    ]

    deduped, dedup_counts = deduplicate_speeches([
        type("Speech", (), {"speech_id": item["speech_id"], "congress": item["congress"], "text": item["text"]})()
        for item in speeches
    ])

    unique_ids = {speech.speech_id for speech in deduped}
    holdout_ids, train_speeches, eval_speeches = create_speech_holdout(deduped, holdout_count=args.holdout_count)

    input_tokens = sum(len(item["text"].split()) for item in speeches)
    chunked = build_chunked_dataset(deduped, args.max_length)
    output_tokens = sum(len(chunk.text.split()) for chunk in chunked)

    print("=== validation summary ===")
    print(f"input_speeches={len(speeches)}")
    print(f"deduplicated_speeches={len(deduped)}")
    print(f"dedup_counts={dedup_counts}")
    print(f"train_speeches={len(train_speeches)}")
    print(f"eval_speeches={len(eval_speeches)}")
    print(f"input_tokens={input_tokens}")
    print(f"output_tokens={output_tokens}")
    print(f"token_ratio={output_tokens / max(1, input_tokens):.4f}")
    print(f"max_chunk_length={max(len(chunk.text.split()) for chunk in chunked) if chunked else 0}")
    print(f"heldout_ids_present={bool(set(holdout_ids) & unique_ids)}")

    assert max(len(chunk.text.split()) for chunk in chunked) <= args.max_length, "A chunk exceeded the model max length."
    assert len(set(holdout_ids) & unique_ids) >= 1, "Held-out speech IDs were not isolated from the train set."


if __name__ == "__main__":
    main()
