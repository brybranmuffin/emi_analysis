"""Continued pretraining of BERT (MLM) or GPT-2 (CLM) on chunked congressional speeches.

    python -m emi_analysis.train --config configs/bert_continued_pretraining.json

Inputs come from ``python -m emi_analysis.prepare_data``. Each run writes to
``<output_dir>/run-<name>/``:
    train.log              same lines as the SLURM .out file
    run_config.json        resolved config, environment, token accounting, step schedule
    metrics.csv            one row per evaluation: step 0 (unadapted model) and every checkpoint
    checkpoint-<step>/     model + tokenizer + checkpoint_info.json
    run_summary.json       final status (completed / stopped_on_flag / failed)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import BERT_CONFIG, GPT2_CONFIG, ModelConfig
from .data_pipeline import chunk_token_ids
from .runinfo import banner, environment_info, format_duration, setup_logging
from .verification import append_metric_row, check_stop_conditions

logger = logging.getLogger("emi_analysis")

STOP_EXIT_CODE = 3
PREP_HINT = "Run `python -m emi_analysis.prepare_data --input <corpus>` first (see README)."


def parse_config(config_path: str | Path | dict) -> ModelConfig:
    if isinstance(config_path, dict):
        config = dict(config_path)
    else:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

    model_family = config.get("model_family")
    if model_family == "bert":
        defaults = BERT_CONFIG.__dict__.copy()
    elif model_family == "gpt2":
        defaults = GPT2_CONFIG.__dict__.copy()
    else:
        raise ValueError(f"Unsupported model family in config: {model_family}")

    allowed_fields = {field.name for field in fields(ModelConfig)}
    merged = {**defaults, **{key: value for key, value in config.items() if key in allowed_fields}}
    return ModelConfig(**merged)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch continued-pretraining for BERT or GPT-2.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for checkpoint directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name (default: job<SLURM_JOB_ID> or a timestamp).")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap on optimizer steps (default: config epochs).")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint every N steps (default: every 10%% of training).")
    parser.add_argument("--log-every", type=int, default=None, help="Progress line every N optimizer steps.")
    parser.add_argument("--num-proc", type=int, default=None, help="Tokenization processes (default: SLURM_CPUS_PER_TASK or 1).")
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU.")
    parser.add_argument("--legacy-truncation", action="store_true", help="Reproduce the old loader: keep only the first window of each speech.")
    parser.add_argument("--allow-missing-lexicons", action="store_true", help="Smoke tests only: skip seed drift if lexicons are missing.")
    return parser


# ------------------------------------------------------------------ schedule


def checkpoint_schedule(total_steps: int, interval: float = 0.10, every: int | None = None) -> list[int]:
    """Optimizer steps at which to checkpoint and evaluate; always ends on ``total_steps``."""
    if every:
        steps = list(range(every, total_steps + 1, every))
        if not steps or steps[-1] != total_steps:
            steps.append(total_steps)
        return steps
    n = max(1, round(1 / interval))
    return sorted({round(total_steps * k / n) for k in range(1, n + 1)} - {0})


def parse_slurm_time(value: str) -> int | None:
    """Seconds in a SLURM time string such as ``1-02:03:04``, ``02:03:04`` or ``03:04``."""
    value = (value or "").strip()
    if not value or not value[0].isdigit():
        return None
    days, _, clock = value.rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return int(days or 0) * 86_400 + hours * 3_600 + minutes * 60 + seconds


def slurm_seconds_left() -> int | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%L"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_slurm_time(out)


# ---------------------------------------------------------------------- data


class LengthGroupedBatchSampler:
    """Shuffle, then sort by length inside mega-batches so each batch holds similar lengths.

    This is the handoff's "length bucketing" alternative to packing: little padding, and no
    sequence ever attends across a speech boundary.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, megabatch_mult: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.megabatch = batch_size * megabatch_mult

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def epoch_batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.lengths))
        batches: list[np.ndarray] = []
        for start in range(0, len(order), self.megabatch):
            mega = order[start : start + self.megabatch]
            mega = mega[np.argsort(-self.lengths[mega], kind="stable")]
            batches.extend(mega[i : i + self.batch_size] for i in range(0, len(mega), self.batch_size))
        return [batches[i].tolist() for i in rng.permutation(len(batches))]


def load_jsonl_dataset(path: str | Path, cache_dir: str | Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train", cache_dir=str(cache_dir))


def tokenize_and_chunk(dataset, tokenizer, window: int, min_chunk_tokens: int, truncate: bool, num_proc: int | None):
    """Tokenize whole speeches, split them into windows, and account for every token.

    Returns ``(chunk_dataset, lengths, stats)``. Each chunk row holds ``input_ids`` including
    [CLS]/[SEP] for BERT. Raises if tokens out + tokens dropped != tokens in.
    """
    from datasets import Features, Sequence as Seq, Value

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    wrap = cls_id is not None and sep_id is not None
    num_proc = num_proc if num_proc and num_proc > 1 else None

    def split_batch(batch):
        encoded = tokenizer(batch["text"], add_special_tokens=False, verbose=False)["input_ids"]
        out = {"chunks": [], "n_tokens": [], "n_dropped": []}
        for ids in encoded:
            chunks, dropped = chunk_token_ids(ids, window, min_chunk_tokens, truncate)
            out["chunks"].append(chunks)
            out["n_tokens"].append(len(ids))
            out["n_dropped"].append(dropped)
        return out

    def flatten(batch):
        rows = [chunk for chunks in batch["chunks"] for chunk in chunks]
        if wrap:
            rows = [[cls_id] + chunk + [sep_id] for chunk in rows]
        return {"input_ids": rows, "length": [len(r) for r in rows]}

    per_speech = dataset.map(
        split_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        features=Features({"chunks": Seq(Seq(Value("int32"))), "n_tokens": Value("int64"), "n_dropped": Value("int64")}),
    )
    chunks = per_speech.map(
        flatten,
        batched=True,
        num_proc=num_proc,
        remove_columns=per_speech.column_names,
        features=Features({"input_ids": Seq(Value("int32")), "length": Value("int32")}),
    )

    counts = per_speech.select_columns(["n_tokens", "n_dropped"]).with_format("numpy")[:]
    lengths = chunks.select_columns(["length"]).with_format("numpy")[:]["length"].astype(np.int64)
    n_special = 2 if wrap else 0
    stats = {
        "speeches": len(per_speech),
        "tokens_in": int(counts["n_tokens"].sum()),
        "tokens_dropped": int(counts["n_dropped"].sum()),
        "tokens_out": int(lengths.sum()) - n_special * len(lengths),
        "chunks": len(chunks),
        "max_chunk_length": int(lengths.max()) if len(lengths) else 0,
        "mean_chunk_length": round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        "special_tokens_per_chunk": n_special,
        "legacy_truncation": truncate,
    }
    if stats["tokens_out"] + stats["tokens_dropped"] != stats["tokens_in"]:
        raise RuntimeError(f"Token accounting mismatch: {stats}")
    return chunks, lengths, stats


def collate(rows: Sequence[Sequence[int]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.as_tensor(row, dtype=torch.long)
        attention_mask[i, : len(row)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mask_tokens_for_mlm(
    input_ids: torch.Tensor,
    maskable: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking: select ``mlm_probability`` of maskable tokens; 80% -> [MASK], 10% random, 10% kept."""
    shape = input_ids.shape
    selected = torch.bernoulli(torch.full(shape, mlm_probability), generator=generator).bool() & maskable
    labels = torch.where(selected, input_ids, torch.full_like(input_ids, -100))
    masked = input_ids.clone()
    to_mask = torch.bernoulli(torch.full(shape, 0.8), generator=generator).bool() & selected
    masked[to_mask] = mask_token_id
    to_random = torch.bernoulli(torch.full(shape, 0.5), generator=generator).bool() & selected & ~to_mask
    masked[to_random] = torch.randint(vocab_size, shape, generator=generator)[to_random]
    return masked, labels


def make_batch(rows, family: str, tokenizer, config: ModelConfig, generator: torch.Generator) -> dict[str, torch.Tensor]:
    batch = collate(rows, tokenizer.pad_token_id)
    if family == "bert":
        special = torch.as_tensor(tokenizer.all_special_ids)
        maskable = batch["attention_mask"].bool() & ~torch.isin(batch["input_ids"], special)
        batch["input_ids"], batch["labels"] = mask_tokens_for_mlm(
            batch["input_ids"], maskable, tokenizer.mask_token_id, len(tokenizer), config.mlm_probability, generator
        )
    else:
        batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    return batch


def build_eval_batches(chunks, family: str, tokenizer, config: ModelConfig, seed: int) -> list[dict[str, torch.Tensor]]:
    """Fixed evaluation batches; BERT masks are drawn once so every checkpoint sees the same task."""
    rows = chunks.with_format(None)[:]["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    generator = torch.Generator().manual_seed(seed)
    return [
        make_batch([rows[i] for i in order[s : s + config.batch_size]], family, tokenizer, config, generator)
        for s in range(0, len(order), config.batch_size)
    ]


def load_lexicon(path: str | Path) -> list[str]:
    """Seed words from the EMI repo's lexicon CSV (column ``*_keywords``) or a one-word-per-line text file."""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            column = next((c for c in reader.fieldnames or [] if "keyword" in c.lower()), (reader.fieldnames or [None])[0])
            words = [row[column] for row in reader]
    else:
        words = path.read_text(encoding="utf-8").splitlines()
    return list(dict.fromkeys(w.strip() for w in words if w and w.strip()))


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, batches, family: str, device: torch.device, autocast) -> float:
    """Token-weighted held-out MLM loss (BERT) or perplexity (GPT-2)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        if family == "gpt2":
            logits, labels = logits[:, :-1], labels[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += int((labels != -100).sum().item())
    model.train()
    nll = total_loss / max(1, total_tokens)
    return math.exp(nll) if family == "gpt2" else nll


@torch.no_grad()
def embed_words(model, tokenizer, words: list[str], device: torch.device) -> torch.Tensor:
    """Mean-pooled last hidden state of each seed word encoded on its own (sentence-transformers style)."""
    model.eval()
    encoded = tokenizer(words, padding=True, return_tensors="pt").to(device)
    hidden = model.base_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    model.train()
    return pooled.double().cpu()


def seed_drift(start: torch.Tensor, now: torch.Tensor) -> tuple[float, float]:
    distance = (1.0 - F.cosine_similarity(start, now, dim=-1)).clamp(min=0.0)
    return float(distance.mean()), float(distance.max())


# ------------------------------------------------------------------ training


def _change(value: float, base: float) -> str:
    return f"{value:.4f} ({(value - base) / base:+.2%} vs step 0)" if base else f"{value:.4f}"


def run(config: ModelConfig, args: argparse.Namespace, run_dir: Path) -> str:
    import datasets
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, get_scheduler

    started = time.time()
    datasets.disable_progress_bars()
    transformers.utils.logging.disable_progress_bar()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    family = config.model_family

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    use_bf16 = config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    environment = {**environment_info(), "device": device_name, "precision": "bf16 autocast" if use_bf16 else "fp32",
                   "torch": torch.__version__, "transformers": transformers.__version__}
    banner(logger, f"EMI continued pretraining: {family} ({config.model_name})",
           {"run directory": run_dir, "config file": args.config, **environment})
    if device.type != "cuda":
        logger.warning("Training on CPU. This is only reasonable for smoke tests.")

    for label, path in (("training data", config.train_data_path), ("held-out congressional data", config.heldout_data_path),
                        ("general-domain data", config.general_data_path)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found at {path}. {PREP_HINT}")
    lexicon_paths = (config.evidence_lexicon_path, config.intuition_lexicon_path)
    missing_lexicons = [p for p in lexicon_paths if not Path(p).exists()]
    if missing_lexicons and not args.allow_missing_lexicons:
        raise FileNotFoundError(
            f"Seed-word lexicon(s) not found: {missing_lexicons}. Seed drift is a required checkpoint metric; copy "
            "evidence_lexicon.csv / intuition_lexicon.csv from the EMI OSF repository (doi 10.17605/OSF.IO/Z6UTW) "
            "to these paths, or set evidence_lexicon_path / intuition_lexicon_path in the config."
        )
    seed_words = [] if missing_lexicons else [w for p in lexicon_paths for w in load_lexicon(p)]
    if missing_lexicons:
        logger.warning(f"Seed drift will NOT be measured: missing {missing_lexicons} (--allow-missing-lexicons)")
    else:
        logger.info(f"Loaded {len(seed_words)} seed words from {lexicon_paths[0]} and {lexicon_paths[1]}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.local_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForMaskedLM if family == "bert" else AutoModelForCausalLM
    model = model_cls.from_pretrained(config.model_name, cache_dir=config.local_cache_dir).to(device)
    model.train()
    logger.info(f"Loaded {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters onto {device_name}")

    # ---- data: tokenize with the model's own tokenizer and chunk; every token is accounted for.
    num_proc = config.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    n_special = 2 if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None else 0
    window = config.max_length - n_special
    if config.min_chunk_tokens > window:
        raise ValueError(f"min_chunk_tokens ({config.min_chunk_tokens}) exceeds the chunk window ({window}); every chunk would be dropped.")
    t0 = time.time()
    logger.info(f"Tokenizing and chunking {config.train_data_path} (window {window} + {n_special} special tokens, {num_proc} processes)")
    train_chunks, train_lengths, train_stats = tokenize_and_chunk(
        load_jsonl_dataset(config.train_data_path, config.datasets_cache_dir), tokenizer, window,
        config.min_chunk_tokens, config.legacy_truncation, num_proc,
    )
    banner(logger, f"TOKEN ACCOUNTING: training set, {family} tokenizer ({format_duration(time.time() - t0)})", {
        "speeches": f"{train_stats['speeches']:,}",
        "tokens in": f"{train_stats['tokens_in']:,}",
        "tokens out (in chunks)": f"{train_stats['tokens_out']:,} ({train_stats['tokens_out'] / max(1, train_stats['tokens_in']):.4%} of tokens in)",
        "tokens dropped": f"{train_stats['tokens_dropped']:,} "
        + ("(legacy truncation past the first window)" if config.legacy_truncation else f"(chunks < {config.min_chunk_tokens} tokens)"),
        "chunks": f"{train_stats['chunks']:,} (mean {train_stats['mean_chunk_length']} tokens incl. specials)",
        "max chunk length": f"{train_stats['max_chunk_length']} (limit {config.max_length})",
        "check": "tokens in = tokens out + tokens dropped: OK",
    })
    if config.legacy_truncation:
        logger.warning("LEGACY TRUNCATION is on: everything after the first window of each speech is discarded.")
    if train_stats["chunks"] == 0:
        raise RuntimeError(f"No training chunks were produced from {config.train_data_path}.")
    if train_stats["max_chunk_length"] > config.max_length:
        raise RuntimeError(f"A chunk has {train_stats['max_chunk_length']} tokens, above max_length {config.max_length}.")

    eval_sets: dict[str, list[dict[str, torch.Tensor]]] = {}
    eval_stats: dict[str, dict] = {}
    for offset, (name, path) in enumerate((("congressional", config.heldout_data_path), ("general", config.general_data_path)), start=1):
        chunks, _, stats = tokenize_and_chunk(load_jsonl_dataset(path, config.datasets_cache_dir), tokenizer, window,
                                              config.min_chunk_tokens, False, num_proc)
        if len(chunks) > config.eval_max_chunks:
            keep = np.random.default_rng(config.seed + offset).choice(len(chunks), size=config.eval_max_chunks, replace=False)
            chunks = chunks.select(sorted(keep.tolist()))
        stats["chunks_evaluated"] = len(chunks)
        eval_sets[name] = build_eval_batches(chunks, family, tokenizer, config, seed=config.seed + offset)
        eval_stats[name] = stats
        logger.info(f"Held-out {name} set: {stats['speeches']:,} texts -> {stats['chunks']:,} chunks; "
                    f"evaluating a fixed {len(chunks):,} of them at every checkpoint")
        if len(chunks) == 0:
            raise RuntimeError(f"Held-out {name} set at {path} produced no chunks.")

    # ---- schedule
    sampler = LengthGroupedBatchSampler(train_lengths, config.batch_size, config.seed)
    batches_per_epoch = len(sampler)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    epoch_steps = math.ceil(config.epochs * steps_per_epoch)
    total_steps = min(args.max_steps, epoch_steps) if args.max_steps else epoch_steps
    schedule = checkpoint_schedule(total_steps, config.checkpoint_interval, args.checkpoint_every)
    log_every = max(1, config.log_every)

    decay = [p for n, p in model.named_parameters() if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n or "ln_" in n)]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n or "ln_" in n)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate
    )
    warmup_steps = round(config.warmup_ratio * total_steps)
    scheduler = get_scheduler(config.lr_scheduler, optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    schedule_info = {
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "epochs": config.epochs,
        "total_steps": total_steps,
        "max_steps_override": args.max_steps,
        "checkpoint_steps": schedule,
        "warmup_steps": warmup_steps,
    }
    run_config = {
        "config": asdict(config),
        "cli": vars(args),
        "environment": environment,
        "data": {"train": train_stats, **{f"{k}_eval": v for k, v in eval_stats.items()}},
        "seed_words": len(seed_words),
        "schedule": schedule_info,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")
    banner(logger, "TRAINING PLAN", {
        "objective": "masked LM (fresh masks every batch)" if family == "bert" else "causal LM",
        "epochs": f"{config.epochs} x {steps_per_epoch:,} steps/epoch",
        "total optimizer steps": f"{total_steps:,}" + (f" (capped by --max-steps; full run is {epoch_steps:,})" if total_steps < epoch_steps else ""),
        "effective batch": f"{config.batch_size * config.gradient_accumulation_steps} chunks",
        "learning rate": f"{config.learning_rate} ({config.lr_scheduler}, {warmup_steps} warmup steps)",
        "checkpoints": f"{len(schedule)} at steps {schedule}",
        "stop if": f"general loss > +{config.general_loss_max_rel_increase:.0%} vs step 0, or mean seed drift "
        f"> {config.seed_drift_threshold} and climbing" + ("" if config.stop_on_flag else " (flag only, no stop)"),
        "run config": run_dir / "run_config.json",
    })

    metrics_path = run_dir / "metrics.csv"
    loss_metric = "mlm_loss" if family == "bert" else "perplexity"
    base_words = embed_words(model, tokenizer, seed_words, device) if seed_words else None
    state = {"step": 0, "micro": 0, "tokens": 0, "eval_seconds": 0.0, "status": "running"}
    rows: list[dict] = []

    def evaluate_point(step: int, checkpoint_path: str, train_loss: float | None) -> dict:
        eval_started = time.time()
        row = {
            "model": family,
            "step": step,
            "congressional_loss": round(evaluate(model, eval_sets["congressional"], family, device, autocast), 6),
            "general_loss": round(evaluate(model, eval_sets["general"], family, device, autocast), 6),
            "mean_seed_drift": "",
            "max_seed_drift": "",
            "loss_metric": loss_metric,
            "epoch": round(state["micro"] / batches_per_epoch, 4),
            "train_loss": "" if train_loss is None else round(train_loss, 6),
            "tokens_seen": state["tokens"],
            "elapsed_seconds": round(time.time() - started, 1),
            "flags": "",
            "checkpoint": checkpoint_path,
        }
        if base_words is not None:
            mean_drift, max_drift = seed_drift(base_words, embed_words(model, tokenizer, seed_words, device))
            row["mean_seed_drift"], row["max_seed_drift"] = round(mean_drift, 6), round(max_drift, 6)
        state["eval_seconds"] += time.time() - eval_started
        row["_eval_seconds"] = time.time() - eval_started
        return row

    def record(row: dict) -> list[str]:
        flags = check_stop_conditions(rows[0], rows[-1], row, config.general_loss_max_rel_increase,
                                      config.seed_drift_threshold) if rows else []
        row["flags"] = "; ".join(flags)
        append_metric_row(metrics_path, row)
        rows.append(row)
        return flags

    logger.info(f"Evaluating the unadapted model (step 0) on {len(eval_sets['congressional'])} + {len(eval_sets['general'])} batches")
    baseline = evaluate_point(0, "", None)
    record(baseline)
    eval_cost = baseline["_eval_seconds"]
    banner(logger, "BASELINE (step 0, unadapted model)", {
        f"congressional {loss_metric}": f"{baseline['congressional_loss']:.4f}",
        f"general {loss_metric}": f"{baseline['general_loss']:.4f}",
        "seed drift": "0 by definition" if base_words is not None else "not measured",
        "evaluation time": format_duration(eval_cost),
        "metrics file": metrics_path,
    })

    # ---- training loop
    train_generator = torch.Generator().manual_seed(config.seed)
    epoch, position = 0, 0
    epoch_batches = sampler.epoch_batches(epoch)
    logger.info(f"Starting epoch 1 ({batches_per_epoch:,} batches)")
    window_loss, window_micro = 0.0, 0
    ckpt_loss, ckpt_micro = 0.0, 0
    train_started = time.time()
    walltime_checked = False
    checkpoint_steps = set(schedule)

    while state["step"] < total_steps:
        if position >= len(epoch_batches):
            epoch, position = epoch + 1, 0
            epoch_batches = sampler.epoch_batches(epoch)
            logger.info(f"Starting epoch {epoch + 1} ({batches_per_epoch:,} batches, reshuffled with seed {config.seed + epoch})")
        rows_in_batch = train_chunks[epoch_batches[position]]["input_ids"]
        position += 1
        batch = make_batch(rows_in_batch, family, tokenizer, config, train_generator)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss ({loss.item()}) at step {state['step'] + 1}, epoch {epoch + 1}.")
        (loss / config.gradient_accumulation_steps).backward()
        state["micro"] += 1
        state["tokens"] += int(batch["attention_mask"].sum().item())
        window_loss += loss.item()
        window_micro += 1
        ckpt_loss += loss.item()
        ckpt_micro += 1
        if state["micro"] % config.gradient_accumulation_steps:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["step"] += 1
        step = state["step"]

        if step % log_every == 0 or step == total_steps:
            train_seconds = time.time() - train_started - (state["eval_seconds"] - eval_cost)
            rate = step / max(train_seconds, 1e-9)
            remaining_ckpts = sum(1 for s in schedule if s > step)
            eta = (total_steps - step) / rate + remaining_ckpts * eval_cost
            gpu = f" | gpu mem {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB" if device.type == "cuda" else ""
            logger.info(
                f"step {step:,}/{total_steps:,} ({step / total_steps:.1%}) | epoch {state['micro'] / batches_per_epoch:.3f} | "
                f"loss {window_loss / window_micro:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {rate:.2f} steps/s | "
                f"{state['tokens'] / max(train_seconds, 1e-9):,.0f} tok/s | elapsed {format_duration(time.time() - started)} | "
                f"ETA {format_duration(eta)}{gpu}"
            )
            window_loss, window_micro = 0.0, 0
            if not walltime_checked and step >= min(total_steps, 5 * log_every):
                walltime_checked = True
                left = slurm_seconds_left()
                if left is not None and eta > left:
                    logger.warning(
                        f"Projected time to finish ({format_duration(eta)}) exceeds the SLURM time left ({format_duration(left)}). "
                        "The job will be killed before the final checkpoint; resubmit with a longer --time or fewer epochs."
                    )

        if step in checkpoint_steps:
            ckpt_dir = run_dir / f"checkpoint-{step}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            row = evaluate_point(step, str(ckpt_dir), ckpt_loss / max(1, ckpt_micro))
            ckpt_loss, ckpt_micro = 0.0, 0
            flags = record(row)
            info = {k: v for k, v in row.items() if not k.startswith("_")}
            info.update(total_steps=total_steps, checkpoint_index=schedule.index(step) + 1, checkpoints_planned=len(schedule),
                        saved_at=datetime.now().isoformat(timespec="seconds"))
            (ckpt_dir / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            details = {
                "saved to": ckpt_dir,
                f"congressional {loss_metric}": _change(row["congressional_loss"], baseline["congressional_loss"]),
                f"general {loss_metric}": _change(row["general_loss"], baseline["general_loss"]),
                "seed drift mean / max": f"{row['mean_seed_drift']} / {row['max_seed_drift']}" if base_words is not None else "not measured",
                "train loss since last checkpoint": f"{row['train_loss']:.4f}",
                "epoch": row["epoch"],
                "tokens seen": f"{row['tokens_seen']:,}",
                "evaluation time": format_duration(row["_eval_seconds"]),
                "flags": row["flags"] or "none",
            }
            banner(logger, f"CHECKPOINT {schedule.index(step) + 1}/{len(schedule)} | step {step:,}/{total_steps:,} ({step / total_steps:.0%})", details)
            if flags:
                for flag in flags:
                    logger.warning(f"STOP CONDITION: {flag}")
                if config.stop_on_flag:
                    logger.error("Stopping the run as the handoff requires; report these numbers before continuing.")
                    state["status"] = "stopped_on_flag"
                    break

    if state["status"] == "running":
        state["status"] = "completed"
    final = {k: v for k, v in rows[-1].items() if not k.startswith("_")}
    summary = {
        "status": state["status"],
        "model": family,
        "steps_completed": state["step"],
        "total_steps": total_steps,
        "epochs_completed": round(state["micro"] / batches_per_epoch, 4),
        "tokens_seen": state["tokens"],
        "checkpoints": [r["checkpoint"] for r in rows if r["checkpoint"]],
        "baseline": {k: v for k, v in baseline.items() if not k.startswith("_")},
        "final": final,
        "elapsed_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    banner(logger, f"TRAINING {state['status'].upper().replace('_', ' ')}", {
        "steps": f"{state['step']:,}/{total_steps:,}",
        "checkpoints written": len(summary["checkpoints"]),
        f"congressional {loss_metric}": _change(final["congressional_loss"], baseline["congressional_loss"]),
        f"general {loss_metric}": _change(final["general_loss"], baseline["general_loss"]),
        "final seed drift mean / max": f"{final['mean_seed_drift']} / {final['max_seed_drift']}",
        "metrics": metrics_path,
        "summary": run_dir / "run_summary.json",
        "elapsed": format_duration(time.time() - started),
    })
    return state["status"]


def main(argv: list[str] | None = None) -> Path:
    args = build_cli().parse_args(argv)
    config = parse_config(args.config)
    overrides = {
        "output_dir": args.output_dir,
        "log_every": args.log_every,
        "num_proc": args.num_proc,
        "legacy_truncation": True if args.legacy_truncation else None,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})

    job_id = os.environ.get("SLURM_JOB_ID")
    run_name = args.run_name or (f"job{job_id}" if job_id else datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = Path(config.output_dir) / f"run-{run_name}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already contains a run; choose a different --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")

    try:
        status = run(config, args, run_dir)
    except Exception as error:
        logger.exception("Training failed")
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}", "failed_at": datetime.now().isoformat(timespec="seconds")}
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    if status == "stopped_on_flag":
        raise SystemExit(STOP_EXIT_CODE)
    return run_dir


if __name__ == "__main__":
    main()
