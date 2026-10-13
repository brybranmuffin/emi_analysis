#!/usr/bin/env python3
"""End-to-end smoke test: synthetic corpus -> prepare_data -> a few training steps for BERT and GPT-2.

Exercises exactly the code path of the full run, on throwaway data under data/smoke/ and
checkpoints/smoke/. The seed lexicons written here are placeholders, not the EMI lexicons.

    python scripts/run_smoke_test.py [--max-steps 4] [--checkpoint-every 2]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from datetime import datetime
from pathlib import Path

from emi_analysis.prepare_data import main as prepare_main
from emi_analysis.train import main as train_main

SMOKE_DATA = Path("data/smoke")
SMOKE_CHECKPOINTS = Path("checkpoints/smoke")
SENTENCES = [
    "Mr. Speaker, I yield myself such time as I may consume.",
    "The committee has reviewed the record and the practical impact of this proposal on the district.",
    "I rise in support of this legislation because it expands opportunity and strengthens accountability.",
    "The data from the budget office show that the program reduced costs by twelve percent.",
    "I believe in my heart that the people of this nation deserve better than this bill.",
    "Members of this body have a responsibility to act with prudence and keep their constituents in view.",
]


def write_synthetic_corpus(path: Path, n_speeches: int = 400) -> None:
    rng = random.Random(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speech_id", "text", "year"])
        writer.writeheader()
        for i in range(n_speeches):
            text = " ".join(rng.choice(SENTENCES) for _ in range(rng.randint(1, 60)))
            writer.writerow({"speech_id": f"smoke{i}", "text": text, "year": rng.choice(range(1879, 2023))})


def write_smoke_config(family: str, root: Path) -> Path:
    config = json.loads(Path(f"configs/{family}_continued_pretraining.json").read_text(encoding="utf-8"))
    config.update(
        train_data_path=str(root / "train.jsonl"),
        heldout_data_path=str(root / "heldout" / "congressional.jsonl"),
        general_data_path=str(root / "heldout" / "general.jsonl"),
        evidence_lexicon_path=str(root / "lexicons" / "evidence_lexicon.csv"),
        intuition_lexicon_path=str(root / "lexicons" / "intuition_lexicon.csv"),
        output_dir=str(SMOKE_CHECKPOINTS / family),
        batch_size=4,
        eval_max_chunks=16,
        log_every=1,
    )
    path = root / f"{family}_smoke_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    corpus = SMOKE_DATA / "raw" / "smoke_corpus.csv.gzip"
    write_synthetic_corpus(corpus)
    general = SMOKE_DATA / "raw" / "general.txt"
    general.write_text("\n".join(f"General domain paragraph {i}. " + " ".join(SENTENCES[::-1]) for i in range(40)), encoding="utf-8")
    prepare_main(["--input", str(corpus), "--out-root", str(SMOKE_DATA), "--holdout-count", "40",
                  "--general-source", str(general), "--general-count", "20", "--regenerate-holdout", "--regenerate-general"])

    lexicons = SMOKE_DATA / "lexicons"
    lexicons.mkdir(parents=True, exist_ok=True)
    (lexicons / "evidence_lexicon.csv").write_text("evidence_keywords\nevidence\ndata\nstudy\nreport\n", encoding="utf-8")
    (lexicons / "intuition_lexicon.csv").write_text("intuition_keywords\nbelieve\nfeel\nheart\nhope\n", encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dirs = []
    for family in ("bert", "gpt2"):
        argv = ["--config", str(write_smoke_config(family, SMOKE_DATA)), "--run-name", f"smoke-{stamp}",
                "--max-steps", str(args.max_steps), "--checkpoint-every", str(args.checkpoint_every)]
        if args.require_gpu:
            argv.append("--require-gpu")
        run_dirs.append(train_main(argv))
    print(f"Smoke test completed successfully. Runs: {', '.join(str(p) for p in run_dirs)}")


if __name__ == "__main__":
    main()
