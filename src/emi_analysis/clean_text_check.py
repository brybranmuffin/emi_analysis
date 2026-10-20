"""Clean-text check: how much of the general-domain loss rise is real forgetting, and how much is WikiText formatting?

The general-domain held-out set is raw WikiText, which spaces out punctuation ("film , television", "player 's",
"( 1990 )"). Byte-level tokenizers (GPT-2, ModernBERT, Qwen3) see that spacing; a model adapted to normally
punctuated speech can find it increasingly odd, inflating the measured forgetting. This script scores the same
2,000 paragraphs twice, raw and detokenized, for the unadapted model and the adapted checkpoint(s), using exactly
the evaluation code of train.py (so the raw scores reproduce the training log).

    python -m emi_analysis.clean_text_check --model qwen3                       # latest completed run, step 0 + final
    python -m emi_analysis.clean_text_check --model bert --all-checkpoints      # every checkpoint of that run
    python -m emi_analysis.clean_text_check --model gpt2 --run-dir checkpoints/gpt2/run-job9843505

Writes <output-dir>/<model>_<run>.csv (one row per checkpoint x text set) and .json (summary); the log is appended
to <output-dir>/<model>_check.log.
BERT is a built-in control: its tokenizer splits off punctuation regardless of spacing, so raw = clean.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .runinfo import banner, environment_info, format_duration, setup_logging
from .train import autocast_for, build_eval_batches, evaluate, load_model_and_tokenizer, parse_config, tokenize_and_chunk

logger = logging.getLogger("emi_analysis")

TEXT_SETS = ("wikitext_raw", "wikitext_clean")
GENERAL_EVAL_OFFSET = 2  # train.py seeds the general-domain evaluation set with config.seed + 2

_DETOKENIZE_RULES = [
    (re.compile(r" @-@ "), "-"),
    (re.compile(r" @,@ "), ","),
    (re.compile(r" @\.@ "), "."),
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),                # " quoted words " -> "quoted words"
    (re.compile(r" ([.,;:!?%)\]}])"), r"\1"),              # no space before closing punctuation
    (re.compile(r"([(\[{$]) "), r"\1"),                     # no space after opening brackets or $
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),  # did n't -> didn't, player 's -> player's
    (re.compile(r"(\w) ' "), r"\1' "),                      # players ' union -> players' union
    (re.compile(r"\s+"), " "),
]


def detokenize_wikitext(text: str) -> str:
    """Undo WikiText's tokenized spacing so the paragraph reads like ordinary prose."""
    for pattern, replacement in _DETOKENIZE_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the general-domain set raw and detokenized, before and after adaptation.")
    parser.add_argument("--model", required=True, choices=["bert", "gpt2", "modernbert", "qwen3"])
    parser.add_argument("--config", default=None, help="Training config (default: configs/<model>_continued_pretraining.json).")
    parser.add_argument("--run-dir", default=None, help="Training run folder (default: latest completed run in the config's output_dir).")
    parser.add_argument("--all-checkpoints", action="store_true", help="Score every checkpoint, not just step 0 and the final one.")
    parser.add_argument("--general-data", default=None, help="General-domain JSONL (default: the config's general_data_path).")
    parser.add_argument("--output-dir", default="results/clean_text_check")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def latest_completed_run(output_dir: Path) -> Path:
    runs = [p.parent for p in output_dir.glob("run-*/run_summary.json")
            if json.loads(p.read_text(encoding="utf-8")).get("status") == "completed"]
    if not runs:
        raise FileNotFoundError(f"No completed training run under {output_dir}; pass --run-dir.")
    return max(runs, key=lambda p: (p / "run_summary.json").stat().st_mtime)


def read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_checkpoint(path: str, run_dir: Path) -> Path:
    """Checkpoint paths are stored relative to where training ran; fall back to the run folder."""
    candidate = Path(path)
    return candidate if candidate.exists() else run_dir / candidate.name


def score(model, tokenizer, texts: list[str], config, device, autocast, num_proc: int) -> dict:
    """Held-out score exactly as train.py computes it for the general-domain set."""
    from datasets import Dataset

    n_special = 2 if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None else 0
    chunks, _, stats = tokenize_and_chunk(Dataset.from_dict({"text": texts}), tokenizer, config.max_length - n_special,
                                          config.min_chunk_tokens, False, num_proc)
    if len(chunks) > config.eval_max_chunks:
        keep = np.random.default_rng(config.seed + GENERAL_EVAL_OFFSET).choice(len(chunks), size=config.eval_max_chunks, replace=False)
        chunks = chunks.select(sorted(keep.tolist()))
    batches, _ = build_eval_batches(chunks, config.objective, tokenizer, config, seed=config.seed + GENERAL_EVAL_OFFSET)
    value = evaluate(model, batches, config.objective, device, autocast)
    shift = 1 if config.objective == "clm" else 0
    scored = int(sum((b["labels"][:, shift:] != -100).sum().item() for b in batches))
    return {"value": value, "loss": math.log(value) if config.objective == "clm" else value,
            "tokens_scored": scored, "model_tokens": stats["tokens_in"], "chunks": len(chunks)}


def pct(new: float, old: float) -> float:
    return round(100 * (new - old) / old, 2) if old else float("nan")


def run_check(args: argparse.Namespace) -> dict:
    started = time.time()
    config = parse_config(args.config or f"configs/{args.model}_continued_pretraining.json")
    run_dir = Path(args.run_dir) if args.run_dir else latest_completed_run(Path(config.output_dir))
    general_path = Path(args.general_data or config.general_data_path)
    output_dir = Path(args.output_dir)
    stem = f"{args.model}_{run_dir.name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    autocast, use_bf16 = autocast_for(config, device)
    num_proc = config.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    metric = "mlm_loss" if config.objective == "mlm" else "perplexity"

    raw_texts = [json.loads(line)["text"] for line in general_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = {"wikitext_raw": raw_texts, "wikitext_clean": [detokenize_wikitext(t) for t in raw_texts]}
    changed = sum(a != b for a, b in zip(raw_texts, texts["wikitext_clean"]))

    metrics = read_metrics(run_dir)
    logged = {int(r["step"]): r for r in metrics}
    final_step = max(logged)
    steps = sorted(logged) if args.all_checkpoints else [0, final_step]
    banner(logger, f"CLEAN-TEXT CHECK: {args.model}", {
        "training run": run_dir,
        "general-domain set": f"{general_path} ({len(raw_texts):,} paragraphs; {changed:,} changed by detokenizing)",
        "example raw": raw_texts[0][:110],
        "example clean": texts["wikitext_clean"][0][:110],
        "checkpoints scored": ", ".join(f"step {s:,}" for s in steps),
        "metric": f"{metric} (same evaluation code and masks as training)",
        "device": f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}, "
                  f"{'bf16 autocast' if use_bf16 else 'fp32'}",
        **environment_info(),
    })

    rows: list[dict] = []
    for step in steps:
        name_or_path = config.model_name if step == 0 else str(resolve_checkpoint(logged[step]["checkpoint"], run_dir))
        model, tokenizer = load_model_and_tokenizer(config, name_or_path, device)
        for text_set in TEXT_SETS:
            t0 = time.time()
            result = score(model, tokenizer, texts[text_set], config, device, autocast, num_proc)
            row = {"model": args.model, "step": step, "is_final": step == final_step, "checkpoint": name_or_path,
                   "text_set": text_set, "metric": metric, "value": round(result["value"], 6),
                   "loss": round(result["loss"], 6), "tokens_scored": result["tokens_scored"],
                   "model_tokens": result["model_tokens"], "chunks": result["chunks"],
                   "training_log_value": "", "matches_training_log": ""}
            if text_set == "wikitext_raw" and general_path == Path(config.general_data_path):
                row["training_log_value"] = float(logged[step]["general_loss"])
                row["matches_training_log"] = abs(row["value"] - row["training_log_value"]) <= 1e-3 * abs(row["training_log_value"])
            rows.append(row)
            logger.info(f"step {step:>9,} | {text_set:<14} | {metric} {row['value']:.4f} | {result['model_tokens']:,} tokens "
                        f"| {format_duration(time.time() - t0)}"
                        + (f" | training log {row['training_log_value']:.4f} ({'matches' if row['matches_training_log'] else 'DIFFERS'})"
                           if row["training_log_value"] != "" else ""))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by = {(r["step"], r["text_set"]): r for r in rows}
    rise = {}
    for text_set in TEXT_SETS:
        base, final = by[(0, text_set)], by[(final_step, text_set)]
        rise[text_set] = {"step0": base["value"], "final": final["value"], "change_pct": pct(final["value"], base["value"]),
                          "loss_step0": base["loss"], "loss_final": final["loss"],
                          "loss_change": round(final["loss"] - base["loss"], 6),
                          "loss_change_pct": pct(final["loss"], base["loss"])}
    raw_rise, clean_rise = rise["wikitext_raw"]["loss_change"], rise["wikitext_clean"]["loss_change"]
    formatting_share = round(1 - clean_rise / raw_rise, 4) if raw_rise > 0 else None

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model, "run_dir": str(run_dir), "final_step": final_step, "metric": metric,
        "rise": rise, "share_of_rise_from_formatting": formatting_share,
        "raw_reproduces_training_log": all(r["matches_training_log"] for r in rows if r["matches_training_log"] != ""),
        "outputs": {"csv": str(csv_path), "json": str(json_path), "log": str(output_dir / f"{args.model}_check.log")},
        "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": round(time.time() - started, 1),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def describe(text_set: str) -> str:
        r = rise[text_set]
        return (f"{r['step0']:.4f} -> {r['final']:.4f} ({r['change_pct']:+.1f}% {metric}; "
                f"loss {r['loss_step0']:.4f} -> {r['loss_final']:.4f}, {r['loss_change_pct']:+.1f}%)")

    banner(logger, f"CLEAN-TEXT CHECK RESULT: {args.model} (step 0 -> step {final_step:,})", {
        "raw WikiText (as in training)": describe("wikitext_raw"),
        "clean WikiText (same paragraphs)": describe("wikitext_clean"),
        "share of the loss rise due to formatting": "n/a (no rise on raw text)" if formatting_share is None
        else f"{formatting_share:.0%}  (1 - clean rise / raw rise, in loss)",
        "raw scores reproduce training log": summary["raw_reproduces_training_log"],
        "results": f"{csv_path}, {json_path}",
        "elapsed": format_duration(time.time() - started),
    })
    if not summary["raw_reproduces_training_log"]:
        logger.warning("Raw scores differ from the training log; check that the config and general-domain file match the run.")
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    setup_logging(Path(args.output_dir) / f"{args.model}_check.log", mode="a")
    try:
        return run_check(args)
    except Exception:
        logger.exception("Clean-text check failed")
        raise


if __name__ == "__main__":
    main()
