#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from emi_analysis.config import BERT_CONFIG, GPT2_CONFIG
from emi_analysis.download_assets import ensure_dataset_download, ensure_local_model
from emi_analysis.train import run_training


def build_sample_texts() -> list[str]:
    base = [
        "Mr. Speaker, I yield myself such time as I may consume and begin by noting the importance of this measure.",
        "The committee has reviewed the record and the practical impact of this proposal on the district and the nation.",
        "I rise in support of this legislation because it expands opportunity, protects public trust, and strengthens accountability.",
        "This bill ensures local communities receive timely assistance and preserves the constitutional role of the people’s representatives.",
        "Members of this body have a responsibility to act with prudence and to keep the interests of their constituents in view.",
    ]
    return base * 8


def main() -> None:
    dataset_files = ensure_dataset_download()
    print(f"Dataset files found: {[str(p) for p in dataset_files][:3]}")

    for model_cfg in (BERT_CONFIG, GPT2_CONFIG):
        ensure_local_model(model_cfg.model_name, model_cfg.local_cache_dir)
        print(f"Model cache prepared for {model_cfg.model_name}")

    run_training(
        model_family="bert",
        model_name=BERT_CONFIG.model_name,
        texts=build_sample_texts(),
        output_dir="checkpoints/bert-smoke",
        max_steps=2,
        checkpoint_every=1,
        max_length=BERT_CONFIG.max_length,
        local_cache_dir=BERT_CONFIG.local_cache_dir,
    )

    run_training(
        model_family="gpt2",
        model_name=GPT2_CONFIG.model_name,
        texts=build_sample_texts(),
        output_dir="checkpoints/gpt2-smoke",
        max_steps=2,
        checkpoint_every=1,
        max_length=GPT2_CONFIG.max_length,
        local_cache_dir=GPT2_CONFIG.local_cache_dir,
    )

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
