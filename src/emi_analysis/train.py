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
import signal
import subprocess
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import BERT_CONFIG, GPT2_CONFIG, MODERNBERT_CONFIG, QWEN3_CONFIG, ModelConfig
from .data_pipeline import chunk_token_ids
from .packing import build_packed_batch, model_inputs, pack_sequences
from .runinfo import banner, environment_info, format_duration, setup_logging
from .verification import append_metric_row, check_stop_conditions

logger = logging.getLogger("emi_analysis")

STOP_EXIT_CODE = 3
CANCEL_EXIT_CODE = 143  # 128 + SIGTERM, what SLURM reports for scancel / time limit
PREP_HINT = "Run `python -m emi_analysis.prepare_data --input <corpus>` first (see README)."
FAMILY_DEFAULTS = {"bert": BERT_CONFIG, "gpt2": GPT2_CONFIG, "modernbert": MODERNBERT_CONFIG, "qwen3": QWEN3_CONFIG}


def parse_config(config_path: str | Path | dict) -> ModelConfig:
    if isinstance(config_path, dict):
        config = dict(config_path)
    else:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

    model_family = config.get("model_family")
    if model_family not in FAMILY_DEFAULTS:
        raise ValueError(f"Unsupported model family in config: {model_family}")
    defaults = FAMILY_DEFAULTS[model_family].__dict__.copy()

    allowed_fields = {field.name for field in fields(ModelConfig)}
    merged = {**defaults, **{key: value for key, value in config.items() if key in allowed_fields}}
    return ModelConfig(**merged)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch continued-pretraining for BERT or GPT-2.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for checkpoint directory.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Read <root>/train.jsonl and <root>/heldout/{congressional,general}.jsonl (a prepare_data --out-root) "
        "and write checkpoints to checkpoints/<root name>/<model> unless --output-dir is given.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name (default: job<SLURM_JOB_ID> or a timestamp).")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap on optimizer steps (default: config epochs).")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint every N steps (default: every 10%% of training).")
    parser.add_argument("--log-every", type=int, default=None, help="Progress line every N optimizer steps.")
    parser.add_argument("--num-proc", type=int, default=None, help="Tokenization processes (default: SLURM_CPUS_PER_TASK or 1).")
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU.")
    parser.add_argument("--legacy-truncation", action="store_true", help="Reproduce the old loader: keep only the first window of each speech.")
    parser.add_argument("--allow-missing-lexicons", action="store_true", help="Smoke tests only: skip seed drift if lexicons are missing.")
    parser.add_argument(
        "--stop-on-flag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="End the run when a stop condition fires (default: config stop_on_flag). Flags are always logged and recorded.",
    )
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


def epoch_end_steps(batches_per_epoch: int, accumulation: int, total_steps: int) -> list[int]:
    """Optimizer step on which each full epoch completes (the first step at or after its last batch)."""
    steps = []
    epoch = 1
    while (step := math.ceil(epoch * batches_per_epoch / accumulation)) <= total_steps:
        steps.append(step)
        epoch += 1
    return steps


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


def heldout_overlap(train_dataset, heldout_dataset) -> int | None:
    """Number of training rows whose speech_id is a held-out speech; None if either file lacks speech_id."""
    if "speech_id" not in train_dataset.column_names or "speech_id" not in heldout_dataset.column_names:
        return None
    heldout_ids = {str(s) for s in heldout_dataset.select_columns(["speech_id"]).with_format(None)[:]["speech_id"]}
    train_ids = train_dataset.select_columns(["speech_id"]).with_format(None)[:]["speech_id"]
    return sum(1 for s in train_ids if str(s) in heldout_ids)


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


def make_batch(rows, objective: str, tokenizer, config: ModelConfig, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Training/eval batch with labels. ``rows`` are chunks, or packs (lists of chunks) when ``config.packing``.

    ``_real`` marks real (non-padding) tokens; other ``_`` keys are packing bookkeeping, not model inputs.
    """
    if config.packing:
        batch = build_packed_batch(rows, tokenizer.pad_token_id)
    else:
        batch = collate(rows, tokenizer.pad_token_id)
        batch["_real"] = batch["attention_mask"].bool()
    if objective == "mlm":
        special = torch.as_tensor(tokenizer.all_special_ids)
        maskable = batch["_real"] & ~torch.isin(batch["input_ids"], special)
        batch["input_ids"], batch["labels"] = mask_tokens_for_mlm(
            batch["input_ids"], maskable, tokenizer.mask_token_id, len(tokenizer), config.mlm_probability, generator
        )
    elif config.packing:
        # A speech's first token would be predicted from the end of the previous speech: never score it.
        batch["labels"] = batch["input_ids"].masked_fill(~batch["_real"] | batch["_doc_start"], -100)
    else:
        batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    return batch


def pack_stats(packs: list[list[int]], lengths: np.ndarray, capacity: int) -> dict:
    tokens = int(sum(int(lengths[i]) for pack in packs for i in pack))
    return {
        "pieces": int(sum(len(pack) for pack in packs)),
        "packs": len(packs),
        "tokens": tokens,
        "capacity": capacity,
        "fill_rate": round(tokens / max(1, len(packs) * capacity), 4),
        "mean_pieces_per_pack": round(sum(len(pack) for pack in packs) / max(1, len(packs)), 2),
    }


def build_eval_batches(chunks, objective: str, tokenizer, config: ModelConfig, seed: int) -> tuple[list[dict[str, torch.Tensor]], dict | None]:
    """Fixed evaluation batches; MLM masks are drawn once so every checkpoint sees the same task.

    With packing, the evaluation chunks are packed the same way as training (returns the pack stats too).
    """
    rows = chunks.with_format(None)[:]["input_ids"]
    generator = torch.Generator().manual_seed(seed)
    if config.packing:
        lengths = np.asarray([len(r) for r in rows])
        packs = pack_sequences(lengths, config.max_length)
        units = [[rows[i] for i in pack] for pack in packs]
        stats = pack_stats(packs, lengths, config.max_length)
    else:
        units = [rows[i] for i in sorted(range(len(rows)), key=lambda i: len(rows[i]))]
        stats = None
    batches = [make_batch(units[s : s + config.batch_size], objective, tokenizer, config, generator)
               for s in range(0, len(units), config.batch_size)]
    return batches, stats


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
def evaluate(model, batches, objective: str, device: torch.device, autocast) -> float:
    """Token-weighted held-out MLM loss (masked LM) or perplexity (causal LM)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        labels = batch["labels"].to(device)
        with autocast():
            # No labels passed: ModernBERT's sparse prediction would otherwise return logits for masked tokens only.
            logits = model(**model_inputs(batch, model, objective, device, labels=False)).logits
        if objective == "clm":
            logits, labels = logits[:, :-1], labels[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += int((labels != -100).sum().item())
    model.train()
    nll = total_loss / max(1, total_tokens)
    return math.exp(nll) if objective == "clm" else nll


def split_weight_decay(model) -> tuple[list[str], list[str]]:
    """Parameter names with and without weight decay: biases and norm weights (1-D tensors) get none."""
    decay = [n for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [n for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    return decay, no_decay


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


def autocast_for(config: ModelConfig, device: torch.device):
    """bf16 autocast context factory on GPUs that support it (weights stay float32); a no-op otherwise."""
    use_bf16 = config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext), use_bf16


def load_model_and_tokenizer(config: ModelConfig, name_or_path: str, device: torch.device):
    """Load a base model (hub name) or a saved checkpoint exactly as training does."""
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, cache_dir=config.local_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForMaskedLM if config.objective == "mlm" else AutoModelForCausalLM
    # Packed batches rely on SDPA/eager attention masks; flash-attention kernels would ignore them.
    load_kwargs = {"attn_implementation": "sdpa"} if config.packing else {}
    # Always train float32 weights (bf16 only inside autocast). transformers otherwise keeps a checkpoint's stored
    # dtype, and Qwen3 is stored in bf16, where updates of size lr x gradient are largely rounded away.
    model = model_cls.from_pretrained(name_or_path, cache_dir=config.local_cache_dir, dtype=torch.float32, **load_kwargs)
    return model.to(device), tokenizer


def _change(value: float, base: float) -> str:
    return f"{value:.4f} ({(value - base) / base:+.2%} vs step 0)" if base else f"{value:.4f}"


def run(config: ModelConfig, args: argparse.Namespace, run_dir: Path) -> str:
    import datasets
    import transformers
    from transformers import get_scheduler

    started = time.time()
    datasets.disable_progress_bars()
    transformers.utils.logging.disable_progress_bar()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    family = config.model_family
    objective = config.objective

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    autocast, use_bf16 = autocast_for(config, device)
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

    model, tokenizer = load_model_and_tokenizer(config, config.model_name, device)
    model.train()
    environment["weights_dtype"] = str(model.dtype)
    logger.info(f"Loaded {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters onto {device_name} "
                f"(weights {model.dtype}, attention {model.config._attn_implementation})")

    # ---- data: tokenize with the model's own tokenizer and chunk; every token is accounted for.
    num_proc = config.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    n_special = 2 if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None else 0
    window = config.max_length - n_special
    if config.min_chunk_tokens > window:
        raise ValueError(f"min_chunk_tokens ({config.min_chunk_tokens}) exceeds the chunk window ({window}); every chunk would be dropped.")
    train_raw = load_jsonl_dataset(config.train_data_path, config.datasets_cache_dir)
    eval_raw = {
        "congressional": load_jsonl_dataset(config.heldout_data_path, config.datasets_cache_dir),
        "general": load_jsonl_dataset(config.general_data_path, config.datasets_cache_dir),
    }
    overlap = heldout_overlap(train_raw, eval_raw["congressional"])
    if overlap is None:
        logger.warning("Held-out leakage check skipped: speech_id column missing from the training or held-out file")
    elif overlap:
        raise RuntimeError(
            f"{overlap:,} held-out speech IDs from {config.heldout_data_path} also appear in {config.train_data_path}. "
            "These files must come from the same prepare_data run (check --data-root and the config paths)."
        )
    else:
        logger.info(f"Held-out leakage check: 0 of {len(eval_raw['congressional']):,} held-out speech IDs appear in "
                    f"{config.train_data_path}: OK")

    t0 = time.time()
    logger.info(f"Tokenizing and chunking {config.train_data_path} (window {window} + {n_special} special tokens, {num_proc} processes)")
    train_chunks, train_lengths, train_stats = tokenize_and_chunk(
        train_raw, tokenizer, window, config.min_chunk_tokens, config.legacy_truncation, num_proc,
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

    # ---- packing: several chunks per max_length sequence, each isolated from the others
    pack_info: dict[str, dict] = {}
    if config.packing:
        t0 = time.time()
        train_packs = pack_sequences(train_lengths, config.max_length)
        pack_info["train"] = pack_stats(train_packs, train_lengths, config.max_length)
        unit_lengths = np.asarray([sum(int(train_lengths[i]) for i in pack) for pack in train_packs])
        banner(logger, f"PACKING: training set ({format_duration(time.time() - t0)})", {
            "chunks": f"{pack_info['train']['pieces']:,}",
            "packs": f"{pack_info['train']['packs']:,} of up to {config.max_length:,} tokens",
            "fill rate": f"{pack_info['train']['fill_rate']:.2%} real tokens ({1 - pack_info['train']['fill_rate']:.2%} padding)",
            "chunks per pack": f"{pack_info['train']['mean_pieces_per_pack']} on average",
            "isolation": "positions restart per chunk; attention and next-token targets never cross chunks",
            "check": f"tokens in packs = tokens in chunks ({pack_info['train']['tokens']:,}): OK",
        })
        if pack_info["train"]["tokens"] != int(train_lengths.sum()) or pack_info["train"]["pieces"] != len(train_lengths):
            raise RuntimeError(f"Packing lost chunks or tokens: {pack_info['train']}")
    else:
        train_packs = None
        unit_lengths = train_lengths

    eval_sets: dict[str, list[dict[str, torch.Tensor]]] = {}
    eval_stats: dict[str, dict] = {}
    for offset, (name, path) in enumerate((("congressional", config.heldout_data_path), ("general", config.general_data_path)), start=1):
        chunks, _, stats = tokenize_and_chunk(eval_raw[name], tokenizer, window, config.min_chunk_tokens, False, num_proc)
        if len(chunks) > config.eval_max_chunks:
            keep = np.random.default_rng(config.seed + offset).choice(len(chunks), size=config.eval_max_chunks, replace=False)
            chunks = chunks.select(sorted(keep.tolist()))
        stats["chunks_evaluated"] = len(chunks)
        eval_sets[name], eval_pack_stats = build_eval_batches(chunks, objective, tokenizer, config, seed=config.seed + offset)
        eval_stats[name] = stats
        if eval_pack_stats:
            pack_info[f"{name}_eval"] = eval_pack_stats
        logger.info(f"Held-out {name} set: {stats['speeches']:,} texts -> {stats['chunks']:,} chunks; "
                    f"evaluating a fixed {len(chunks):,} of them at every checkpoint")
        if len(chunks) == 0:
            raise RuntimeError(f"Held-out {name} set at {path} produced no chunks.")

    # ---- schedule
    # One sampler unit is a chunk (bucketed by length) or, with packing, a whole pack.
    sampler = LengthGroupedBatchSampler(unit_lengths, config.batch_size, config.seed)
    batches_per_epoch = len(sampler)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    epoch_steps = math.ceil(config.epochs * steps_per_epoch)
    total_steps = min(args.max_steps, epoch_steps) if args.max_steps else epoch_steps
    schedule = checkpoint_schedule(total_steps, config.checkpoint_interval, args.checkpoint_every)
    epoch_ends = epoch_end_steps(batches_per_epoch, config.gradient_accumulation_steps, total_steps)
    epochs_planned = round(total_steps * config.gradient_accumulation_steps / batches_per_epoch, 3)
    log_every = max(1, config.log_every)

    params = dict(model.named_parameters())
    decay_names, no_decay_names = split_weight_decay(model)
    decay, no_decay = [params[n] for n in decay_names], [params[n] for n in no_decay_names]
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
        "epoch_end_steps": epoch_ends,
        "epochs_planned": epochs_planned,
        "warmup_steps": warmup_steps,
    }
    run_config = {
        "config": asdict(config),
        "cli": vars(args),
        "environment": environment,
        "data": {"train": train_stats, **{f"{k}_eval": v for k, v in eval_stats.items()}},
        "packing": pack_info or None,
        "seed_words": len(seed_words),
        "schedule": schedule_info,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")
    banner(logger, "TRAINING PLAN", {
        "objective": f"masked LM, {config.mlm_probability:.0%} of tokens (fresh masks every batch)" if objective == "mlm" else "causal LM",
        "epochs": f"{config.epochs} x {steps_per_epoch:,} steps/epoch",
        "total optimizer steps": f"{total_steps:,}" + (f" (capped by --max-steps; full run is {epoch_steps:,})" if total_steps < epoch_steps else ""),
        "effective batch": (f"{config.batch_size * config.gradient_accumulation_steps} packs of up to {config.max_length:,} tokens"
                            if config.packing else f"{config.batch_size * config.gradient_accumulation_steps} chunks"),
        "learning rate": f"{config.learning_rate} ({config.lr_scheduler}, {warmup_steps} warmup steps)",
        "checkpoints": f"{len(schedule)} at steps {schedule}",
        "epoch ends": f"steps {epoch_ends} (evaluated and checkpointed too)" if epoch_ends else "none within this run",
        "flag if": f"general loss > +{config.general_loss_max_rel_increase:.0%} vs step 0, or mean seed drift "
        f"> {config.seed_drift_threshold} and climbing",
        "on a flag": "stop the run (stop_on_flag on)" if config.stop_on_flag else "log it and keep training (stop_on_flag off)",
        "run config": run_dir / "run_config.json",
    })

    metrics_path = run_dir / "metrics.csv"
    progress_path = run_dir / "progress.json"
    loss_metric = "mlm_loss" if objective == "mlm" else "perplexity"
    base_words = embed_words(model, tokenizer, seed_words, device) if seed_words else None
    state = {"step": 0, "micro": 0, "tokens": 0, "eval_seconds": 0.0, "status": "running",
             "rate": None, "eta": None, "recent_loss": None, "lr": config.learning_rate}
    rows: list[dict] = []
    flags_raised: list[dict] = []

    def public(row: dict) -> dict:
        return {k: v for k, v in row.items() if not k.startswith("_")}

    def latest_checkpoint() -> str:
        return next((r["checkpoint"] for r in reversed(rows) if r["checkpoint"]), "")

    def write_progress() -> None:
        """Snapshot for deciding mid-run whether to stop; rewritten at every progress line and evaluation."""
        snapshot = {
            "model": family,
            "status": state["status"],
            "run_dir": str(run_dir),
            "log": str(run_dir / "train.log"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "step": state["step"],
            "total_steps": total_steps,
            "percent": round(100 * state["step"] / total_steps, 2),
            "epoch": round(state["micro"] / batches_per_epoch, 4),
            "epochs_planned": epochs_planned,
            "loss_metric": loss_metric,
            "recent_train_loss": state["recent_loss"],
            "learning_rate": state["lr"],
            "steps_per_second": state["rate"],
            "elapsed_seconds": round(time.time() - started, 1),
            "eta_seconds": state["eta"],
            "latest_checkpoint": latest_checkpoint(),
            "baseline": public(rows[0]) if rows else None,
            "latest_evaluation": public(rows[-1]) if rows else None,
            "flags_raised": flags_raised,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        tmp.replace(progress_path)

    def evaluate_point(step: int, checkpoint_path: str, train_loss: float | None, event: str) -> dict:
        eval_started = time.time()
        row = {
            "model": family,
            "step": step,
            "congressional_loss": round(evaluate(model, eval_sets["congressional"], objective, device, autocast), 6),
            "general_loss": round(evaluate(model, eval_sets["general"], objective, device, autocast), 6),
            "mean_seed_drift": "",
            "max_seed_drift": "",
            "loss_metric": loss_metric,
            "epoch": round(state["micro"] / batches_per_epoch, 4),
            "train_loss": "" if train_loss is None else round(train_loss, 6),
            "tokens_seen": state["tokens"],
            "elapsed_seconds": round(time.time() - started, 1),
            "flags": "",
            "checkpoint": checkpoint_path,
            "event": event,
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
        if flags:
            flags_raised.append({"step": row["step"], "epoch": row["epoch"], "flags": flags})
        write_progress()
        return flags

    logger.info(f"Evaluating the unadapted model (step 0) on {len(eval_sets['congressional'])} + {len(eval_sets['general'])} batches")
    baseline = evaluate_point(0, "", None, "baseline")
    record(baseline)
    eval_cost = baseline["_eval_seconds"]
    banner(logger, "BASELINE (step 0, unadapted model)", {
        f"congressional {loss_metric}": f"{baseline['congressional_loss']:.4f}",
        f"general {loss_metric}": f"{baseline['general_loss']:.4f}",
        "seed drift": "0 by definition" if base_words is not None else "not measured",
        "evaluation time": format_duration(eval_cost),
        "metrics file": metrics_path,
        "live progress": progress_path,
    })

    # ---- training loop
    cancel: dict[str, str | None] = {"signal": None}

    def request_cancel(signum, _frame) -> None:
        cancel["signal"] = signal.Signals(signum).name

    previous_handler = signal.signal(signal.SIGTERM, request_cancel)
    train_generator = torch.Generator().manual_seed(config.seed)
    epoch, position = 0, 0
    epoch_batches = sampler.epoch_batches(epoch)
    logger.info(f"Starting epoch 1 ({batches_per_epoch:,} batches)")
    window_loss, window_micro = 0.0, 0
    ckpt_loss, ckpt_micro = 0.0, 0
    epoch_loss, epoch_micro = 0.0, 0
    train_started = time.time()
    walltime_checked = False
    checkpoint_steps = set(schedule)
    epoch_end_set = set(epoch_ends)

    try:
        while state["step"] < total_steps:
            if cancel["signal"]:
                state["status"] = "cancelled"
                logger.warning(
                    f"Received {cancel['signal']} (scancel or SLURM time limit) at step {state['step']:,}/{total_steps:,}; "
                    f"stopping. Latest checkpoint: {latest_checkpoint() or 'none yet'}"
                )
                break
            if position >= len(epoch_batches):
                epoch, position = epoch + 1, 0
                epoch_batches = sampler.epoch_batches(epoch)
                logger.info(f"Starting epoch {epoch + 1} ({batches_per_epoch:,} batches, reshuffled with seed {config.seed + epoch})")
            if train_packs is None:
                rows_in_batch = train_chunks[epoch_batches[position]]["input_ids"]
            else:
                batch_packs = [train_packs[i] for i in epoch_batches[position]]
                fetched = iter(train_chunks[[i for pack in batch_packs for i in pack]]["input_ids"])
                rows_in_batch = [[next(fetched) for _ in pack] for pack in batch_packs]
            position += 1
            batch = make_batch(rows_in_batch, objective, tokenizer, config, train_generator)
            with autocast():
                loss = model(**model_inputs(batch, model, objective, device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss ({loss.item()}) at step {state['step'] + 1}, epoch {epoch + 1}.")
            (loss / config.gradient_accumulation_steps).backward()
            state["micro"] += 1
            state["tokens"] += int(batch["_real"].sum().item())
            window_loss += loss.item()
            window_micro += 1
            ckpt_loss += loss.item()
            ckpt_micro += 1
            epoch_loss += loss.item()
            epoch_micro += 1
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
                remaining_evals = len({s for s in checkpoint_steps | epoch_end_set if s > step})
                eta = (total_steps - step) / rate + remaining_evals * eval_cost
                state.update(rate=round(rate, 3), eta=round(eta), recent_loss=round(window_loss / window_micro, 6),
                             lr=scheduler.get_last_lr()[0])
                gpu = f" | gpu mem {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB" if device.type == "cuda" else ""
                logger.info(
                    f"step {step:,}/{total_steps:,} ({step / total_steps:.1%}) | epoch {state['micro'] / batches_per_epoch:.3f} | "
                    f"loss {window_loss / window_micro:.4f} | lr {state['lr']:.2e} | {rate:.2f} steps/s | "
                    f"{state['tokens'] / max(train_seconds, 1e-9):,.0f} tok/s | elapsed {format_duration(time.time() - started)} | "
                    f"ETA {format_duration(eta)}{gpu}"
                )
                window_loss, window_micro = 0.0, 0
                write_progress()
                if not walltime_checked and step >= min(total_steps, 5 * log_every):
                    walltime_checked = True
                    left = slurm_seconds_left()
                    if left is not None and eta > left:
                        logger.warning(
                            f"Projected time to finish ({format_duration(eta)}) exceeds the SLURM time left ({format_duration(left)}). "
                            "The job will be killed before the final checkpoint; resubmit with a longer --time or fewer epochs."
                        )

            if step not in checkpoint_steps and step not in epoch_end_set:
                continue

            events, titles = [], []
            if step in checkpoint_steps:
                events.append("checkpoint")
                titles.append(f"CHECKPOINT {schedule.index(step) + 1}/{len(schedule)}")
            completed_epoch = int(state["micro"] // batches_per_epoch)
            if step in epoch_end_set:
                events.append("epoch_end")
                titles.append(f"END OF EPOCH {completed_epoch} (of {epochs_planned:g} planned)")
            ckpt_dir = run_dir / f"checkpoint-{step}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            row = evaluate_point(step, str(ckpt_dir), ckpt_loss / max(1, ckpt_micro), "+".join(events))
            ckpt_loss, ckpt_micro = 0.0, 0
            flags = record(row)
            info = public(row)
            info.update(total_steps=total_steps, saved_at=datetime.now().isoformat(timespec="seconds"))
            (ckpt_dir / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            details = {
                "saved to": ckpt_dir,
                f"congressional {loss_metric}": _change(row["congressional_loss"], baseline["congressional_loss"]),
                f"general {loss_metric}": _change(row["general_loss"], baseline["general_loss"]),
                "seed drift mean / max": f"{row['mean_seed_drift']} / {row['max_seed_drift']}" if base_words is not None else "not measured",
                "train loss since last evaluation": f"{row['train_loss']:.4f}",
            }
            if step in epoch_end_set:
                details[f"train loss over epoch {completed_epoch}"] = f"{epoch_loss / max(1, epoch_micro):.4f}"
                epoch_loss, epoch_micro = 0.0, 0
            details.update({
                "epoch": row["epoch"],
                "tokens seen": f"{row['tokens_seen']:,}",
                "evaluation time": format_duration(row["_eval_seconds"]),
                "flags": row["flags"] or "none",
            })
            banner(logger, f"{' + '.join(titles)} | step {step:,}/{total_steps:,} ({step / total_steps:.0%})", details)
            for flag in flags:
                suffix = "" if config.stop_on_flag else " (stop_on_flag is off; the run continues)"
                logger.warning(f"FLAG at step {step:,}: {flag}{suffix}")
            if flags and config.stop_on_flag:
                logger.error(f"STOP CONDITION at step {step:,}: ending the run because stop_on_flag is on. "
                             f"Latest checkpoint: {ckpt_dir}")
                state["status"] = "stopped_on_flag"
                break
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    if state["status"] == "running":
        state["status"] = "completed"
    write_progress()
    final = public(rows[-1])
    summary = {
        "status": state["status"],
        "model": family,
        "steps_completed": state["step"],
        "total_steps": total_steps,
        "epochs_completed": round(state["micro"] / batches_per_epoch, 4),
        "epochs_planned": epochs_planned,
        "tokens_seen": state["tokens"],
        "latest_checkpoint": latest_checkpoint(),
        "checkpoints": [r["checkpoint"] for r in rows if r["checkpoint"]],
        "flags_raised": flags_raised,
        "baseline": public(baseline),
        "final": final,
        "elapsed_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    if state["status"] == "cancelled":
        summary["stop_reason"] = f"received {cancel['signal']}"
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    banner(logger, f"TRAINING {state['status'].upper().replace('_', ' ')}", {
        "steps": f"{state['step']:,}/{total_steps:,}",
        "epochs": f"{summary['epochs_completed']} of {epochs_planned:g}",
        "checkpoints written": len(summary["checkpoints"]),
        "latest checkpoint": summary["latest_checkpoint"] or "none",
        f"congressional {loss_metric} (last evaluation)": _change(final["congressional_loss"], baseline["congressional_loss"]),
        f"general {loss_metric} (last evaluation)": _change(final["general_loss"], baseline["general_loss"]),
        "seed drift mean / max (last evaluation)": f"{final['mean_seed_drift']} / {final['max_seed_drift']}",
        "flags raised": len(flags_raised),
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
        "stop_on_flag": args.stop_on_flag,
    }
    if args.data_root:
        root = Path(args.data_root)
        overrides.update(
            train_data_path=str(root / "train.jsonl"),
            heldout_data_path=str(root / "heldout" / "congressional.jsonl"),
            general_data_path=str(root / "heldout" / "general.jsonl"),
        )
        if not args.output_dir:
            overrides["output_dir"] = str(Path(config.output_dir).parent / root.name / config.model_family)
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
    if status == "cancelled":
        raise SystemExit(CANCEL_EXIT_CODE)
    return run_dir


if __name__ == "__main__":
    main()
