"""Speech-pooled activations from every layer of a model, at its peak-adaptation checkpoint.

For each speech and each hidden state (the embedding output plus every layer), stores the token-weighted mean over
all of the speech's real tokens across all of its chunks. Special tokens ([CLS]/[SEP]) and padding are excluded and
no speech is dropped for being short. Chunking and packing match training: BERT/GPT-2 use 256/512-token chunks,
ModernBERT/Qwen3 pack chunks into 2,048-token sequences with every chunk isolated (proven in tests/test_packing.py).
The last stored layer equals the model's final output (with its final norm), as returned by transformers.

    python -m emi_analysis.collect_activations --model bert --run-dir checkpoints/bert-continued/run-job4757959
    python -m emi_analysis.collect_activations --model gpt2 --input data/train.jsonl \\
        --input data/heldout/congressional.jsonl --tag corpus

The checkpoint is the one with the lowest held-out congressional loss in the run; for a continuation run its
starting checkpoint competes too. Output, under <output-root>/<model>/<checkpoint run>/step<N>/<tag>/:
    layer_00.npy ... layer_LL.npy   [n_speeches x hidden], float16 by default (row i = line i of speech_ids.txt)
    speech_ids.txt, congress.npy, token_counts.npy (tokens pooled per speech)
    meta.json (model, checkpoint and why it was chosen, layers, pooling, chunking, dtype, value ranges, timing)
    progress.json, collect.log
Collection runs in blocks of speeches and resumes from the last finished block if a job is interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .data_pipeline import chunk_token_ids
from .packing import build_packed_batch, model_inputs, pack_sequences
from .runinfo import banner, environment_info, format_duration, setup_logging
from .train import autocast_for, collate, load_jsonl_dataset, load_model_and_tokenizer, parse_config

logger = logging.getLogger("emi_analysis")

FLOAT16_MAX = 65_504.0


def select_peak_checkpoint(rows: list[dict[str, str]], init_from: str | None = None) -> tuple[int, dict[str, str]]:
    """The evaluation with the lowest held-out congressional loss (earliest on ties).

    Step 0 is a candidate only for a continuation run, where it is the checkpoint the run started from.
    """
    candidates = []
    for row in rows:
        step = int(row["step"])
        if step == 0 and init_from:
            candidates.append((float(row["congressional_loss"]), step, {**row, "checkpoint": init_from}))
        elif row.get("checkpoint"):
            candidates.append((float(row["congressional_loss"]), step, row))
    if not candidates:
        raise ValueError("The run has no saved checkpoints to choose from.")
    _, step, row = min(candidates, key=lambda c: (c[0], c[1]))
    return step, row


def check_float16_range(max_abs_by_layer: dict[int, float]) -> None:
    over = {layer: value for layer, value in max_abs_by_layer.items() if value > FLOAT16_MAX}
    if over:
        layer, value = next(iter(sorted(over.items())))
        raise ValueError(f"Pooled activations in layer {layer} reach {value:,.0f}, beyond float16's {FLOAT16_MAX:,.0f}; "
                         "rerun with --dtype float32.")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech-pooled activations for every layer at the peak checkpoint.")
    parser.add_argument("--model", required=True, choices=["bert", "gpt2", "modernbert", "qwen3"])
    parser.add_argument("--config", default=None, help="Training config (default: configs/<model>_continued_pretraining.json).")
    parser.add_argument("--run-dir", default=None, help="Training run to take the peak checkpoint from (default: latest completed run).")
    parser.add_argument("--checkpoint", default=None, help="Use this checkpoint folder instead of the peak one.")
    parser.add_argument("--base", action="store_true", help="Use the unadapted base model.")
    parser.add_argument("--input", action="append", default=None,
                        help="Speech JSONL (speech_id, congress, text); repeatable (default: the held-out speeches).")
    parser.add_argument("--tag", default=None, help="Name for this speech set (default: 'heldout', or the input file names).")
    parser.add_argument("--output-root", default="activations")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--block-speeches", type=int, default=20_000, help="Speeches per block (memory and resume granularity).")
    parser.add_argument("--tokens-per-batch", type=int, default=65_536)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Discard an existing collection with different settings.")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def choose_checkpoint(args, config) -> dict:
    if args.base:
        return {"path": config.model_name, "step": 0, "run": "base", "reason": "unadapted base model (--base)"}
    if args.checkpoint:
        path = Path(args.checkpoint)
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return {"path": str(path), "step": int(match.group(1)) if match else -1, "run": path.parent.name,
                "reason": "given with --checkpoint"}
    from .clean_text_check import latest_completed_run, read_metrics, resolve_checkpoint

    run_dir = Path(args.run_dir) if args.run_dir else latest_completed_run(Path(config.output_dir))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").exists() else {}
    init_from = run_config.get("init_from")
    rows = read_metrics(run_dir)
    step, row = select_peak_checkpoint(rows, init_from)
    path = Path(row["checkpoint"]) if step == 0 else resolve_checkpoint(row["checkpoint"], run_dir)
    match = re.search(r"checkpoint-(\d+)$", path.name)
    candidates = {int(r["step"]): float(r["congressional_loss"]) for r in rows if r.get("checkpoint") or (int(r["step"]) == 0 and init_from)}
    return {
        "path": str(path),
        "step": int(match.group(1)) if match else step,
        "run": path.parent.name,
        "selected_from_run": str(run_dir),
        "init_from": init_from,
        "reason": f"lowest held-out congressional {rows[0].get('loss_metric', 'loss')} ({float(row['congressional_loss']):.4f}) "
                  f"among {len(candidates)} candidates" + (" incl. the continuation's starting checkpoint" if init_from else ""),
        "candidates": candidates,
    }


def tokenize_speeches(dataset, tokenizer, window: int, num_proc: int | None):
    """One row per chunk: input_ids (with [CLS]/[SEP] where the model uses them), length, and its speech's row index."""
    from datasets import Features, Sequence as Seq, Value

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    wrap = cls_id is not None and sep_id is not None

    def split(batch, indices):
        encoded = tokenizer(batch["text"], add_special_tokens=False, verbose=False)["input_ids"]
        out = {"input_ids": [], "length": [], "speech": []}
        for index, ids in zip(indices, encoded):
            chunks, _ = chunk_token_ids(ids, window, min_chunk_tokens=1)
            for chunk in chunks:
                row = [cls_id, *chunk, sep_id] if wrap else chunk
                out["input_ids"].append(row)
                out["length"].append(len(row))
                out["speech"].append(index)
        return out

    return dataset.map(
        split, batched=True, with_indices=True, num_proc=num_proc if num_proc and num_proc > 1 else None,
        remove_columns=dataset.column_names,
        features=Features({"input_ids": Seq(Value("int32")), "length": Value("int32"), "speech": Value("int64")}),
    )


def batches_for_block(rows: list[list[int]], local_speech: np.ndarray, config, tokens_per_batch: int):
    """Yield (units, speech index per unit). Units are chunks (bucketed by length) or packs of chunks."""
    lengths = np.asarray([len(r) for r in rows])
    if config.packing:
        packs = pack_sequences(lengths, config.max_length)
        per_batch = max(1, tokens_per_batch // config.max_length)
        for start in range(0, len(packs), per_batch):
            group = packs[start : start + per_batch]
            yield [[rows[i] for i in pack] for pack in group], [[int(local_speech[i]) for i in pack] for pack in group]
    else:
        order = np.argsort(-lengths, kind="stable")
        start = 0
        while start < len(order):
            width = int(lengths[order[start]])
            count = max(1, tokens_per_batch // width)
            group = order[start : start + count]
            yield [rows[i] for i in group], [int(local_speech[i]) for i in group]
            start += count


def position_speech_index(batch: dict, unit_speech, packing: bool, special_ids: torch.Tensor) -> torch.Tensor:
    """Local speech index for every token position; -1 for padding and special tokens."""
    if packing:
        doc_ids = batch["_doc_ids"]
        lookup = torch.full((doc_ids.shape[0], int(doc_ids.max()) + 1), -1, dtype=torch.long)
        for row, speeches in enumerate(unit_speech):
            lookup[row, : len(speeches)] = torch.as_tensor(speeches)
        index = lookup.gather(1, doc_ids)
        real = batch["_real"]
    else:
        index = torch.as_tensor(unit_speech, dtype=torch.long)[:, None].expand_as(batch["input_ids"]).clone()
        real = batch["attention_mask"].bool()
    return index.masked_fill(~real | torch.isin(batch["input_ids"], special_ids), -1)


def run_collection(args: argparse.Namespace) -> dict:
    import datasets

    datasets.disable_progress_bars()
    started = time.time()
    config = parse_config(args.config or f"configs/{args.model}_continued_pretraining.json")
    checkpoint = choose_checkpoint(args, config)
    inputs = [Path(p) for p in (args.input or [config.heldout_data_path])]
    tag = args.tag or ("heldout" if not args.input else "+".join(p.stem for p in inputs))
    out = Path(args.output_root) / args.model / checkpoint["run"] / f"step{checkpoint['step']}" / tag
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "collect.log", mode="a")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    autocast, use_bf16 = autocast_for(config, device)
    num_proc = args.num_proc or config.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    model, tokenizer = load_model_and_tokenizer(config, checkpoint["path"], device)
    model.eval()
    base = model.base_model
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    speeches = datasets.concatenate_datasets([load_jsonl_dataset(p, config.datasets_cache_dir) for p in inputs])
    n = len(speeches)
    fingerprint = {
        "model": args.model, "checkpoint": checkpoint["path"], "inputs": [[str(p), p.stat().st_size] for p in inputs],
        "n_speeches": n, "n_layers": n_layers, "hidden_size": hidden, "dtype": args.dtype, "max_length": config.max_length,
    }
    meta_path, progress_path = out / "meta.json", out / "progress.json"
    done = 0
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            if not args.overwrite:
                raise RuntimeError(f"{out} holds a collection with different settings; pass --overwrite to replace it.")
            for path in out.glob("layer_*.npy"):
                path.unlink()
        elif progress_path.exists():
            done = int(json.loads(progress_path.read_text(encoding="utf-8")).get("completed_speeches", 0))

    n_special = 2 if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None else 0
    window = config.max_length - n_special
    t0 = time.time()
    chunks = tokenize_speeches(speeches, tokenizer, window, num_proc)
    columns = chunks.select_columns(["speech", "length"]).with_format("numpy")[:]
    chunk_speech, chunk_lengths = columns["speech"], columns["length"].astype(np.int64)
    pooled_tokens = int(chunk_lengths.sum()) - n_special * len(chunk_lengths)
    storage = n * n_layers * hidden * np.dtype(np_dtype).itemsize

    banner(logger, f"ACTIVATION COLLECTION: {args.model} ({tag})", {
        "checkpoint": checkpoint["path"],
        "why this checkpoint": checkpoint["reason"],
        "inputs": ", ".join(str(p) for p in inputs),
        "speeches": f"{n:,} -> {len(chunk_lengths):,} chunks of up to {config.max_length:,} tokens "
                    f"({'packed, speech-isolated' if config.packing else 'batched by length'})",
        "tokens pooled": f"{pooled_tokens:,} (special tokens and padding excluded; tokenized in {format_duration(time.time() - t0)})",
        "layers stored": f"{n_layers} (embedding output + {n_layers - 1} layers) x {hidden} dims",
        "storage": f"{storage / 1e9:,.1f} GB as {args.dtype}",
        "device": f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}, {'bf16 autocast' if use_bf16 else 'fp32'}",
        "output": out,
        **({"resuming": f"from speech {done:,}"} if done else {}),
        **environment_info(),
    })
    if done:
        logger.info(f"Resuming after {done:,} of {n:,} speeches already collected")

    if done == 0:
        (out / "speech_ids.txt").write_text("\n".join(str(s) for s in speeches["speech_id"]) + "\n", encoding="utf-8")
        np.save(out / "congress.npy", np.asarray(speeches["congress"], dtype=np.int16))
        np.save(out / "token_counts.npy", np.zeros(n, dtype=np.int32))
        layers = [np.lib.format.open_memmap(out / f"layer_{i:02d}.npy", mode="w+", dtype=np_dtype, shape=(n, hidden))
                  for i in range(n_layers)]
    else:
        layers = [np.load(out / f"layer_{i:02d}.npy", mmap_mode="r+") for i in range(n_layers)]
    token_counts = np.load(out / "token_counts.npy", mmap_mode="r+")
    meta = {"fingerprint": fingerprint, "status": "running", "model": args.model, "model_name": config.model_name,
            "checkpoint": checkpoint, "n_speeches": n, "n_layers": n_layers, "hidden_size": hidden, "dtype": args.dtype,
            "layer_index": "0 = embedding output; i = output of layer i; last = final output (final norm applied)",
            "pooling": "token-weighted mean over all real tokens of the speech across all its chunks; "
                       "special tokens and padding excluded",
            "chunking": {"max_length": config.max_length, "window": window, "packing": config.packing},
            "precision": "bf16 autocast forward, float32 accumulation" if use_bf16 else "float32",
            "inputs": [str(p) for p in inputs], "tag": tag, "tokens_pooled": pooled_tokens,
            "started_at": datetime.now().isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    special = torch.as_tensor(tokenizer.all_special_ids)
    max_abs = {i: 0.0 for i in range(n_layers)}
    collect_started, collected = time.time(), 0
    for s0 in range(done, n, args.block_speeches):
        s1 = min(s0 + args.block_speeches, n)
        c0, c1 = np.searchsorted(chunk_speech, [s0, s1])
        rows = chunks.select(range(int(c0), int(c1))).with_format(None)[:]["input_ids"]
        sums = torch.zeros((n_layers, s1 - s0, hidden), dtype=torch.float32, device=device)
        counts = torch.zeros(s1 - s0, dtype=torch.float32, device=device)
        for units, unit_speech in batches_for_block(rows, chunk_speech[c0:c1] - s0, config, args.tokens_per_batch):
            batch = build_packed_batch(units, tokenizer.pad_token_id) if config.packing else collate(units, tokenizer.pad_token_id)
            index = position_speech_index(batch, unit_speech, config.packing, special).to(device).reshape(-1)
            keep = index >= 0
            with torch.no_grad(), autocast():
                states = base(**model_inputs(batch, model, config.objective, device, labels=False), output_hidden_states=True).hidden_states
            for layer, state in enumerate(states):
                sums[layer].index_add_(0, index[keep], state.reshape(-1, hidden)[keep].float())
            counts.index_add_(0, index[keep], torch.ones(int(keep.sum()), device=device))

        means = (sums / counts.clamp(min=1)[None, :, None]).cpu().numpy()
        if not np.isfinite(means).all():
            raise RuntimeError(f"Non-finite pooled activations in speeches {s0:,}-{s1:,}.")
        block_max = {i: float(np.abs(means[i]).max(initial=0.0)) for i in range(n_layers)}
        if args.dtype == "float16":
            check_float16_range(block_max)
        for i in range(n_layers):
            layers[i][s0:s1] = means[i].astype(np_dtype)
            layers[i].flush()
            max_abs[i] = max(max_abs[i], block_max[i])
        token_counts[s0:s1] = counts.cpu().numpy().astype(np.int32)
        token_counts.flush()
        progress_path.write_text(json.dumps({"completed_speeches": s1, "n_speeches": n,
                                             "updated_at": datetime.now().isoformat(timespec="seconds")}) + "\n", encoding="utf-8")
        collected += s1 - s0
        rate = collected / max(time.time() - collect_started, 1e-9)
        logger.info(f"speeches {s1:,}/{n:,} ({s1 / n:.1%}) | {rate:,.0f} speeches/s | elapsed {format_duration(time.time() - started)} "
                    f"| ETA {format_duration((n - s1) / rate)} | max |activation| so far {max(max_abs.values()):,.1f}")

    counts_all = np.load(out / "token_counts.npy")
    empty = int((counts_all == 0).sum())
    if empty:
        logger.warning(f"{empty:,} speeches have no tokens; their rows are all zeros (token_counts = 0).")
    meta.update(status="complete", finished_at=datetime.now().isoformat(timespec="seconds"),
                elapsed_seconds=round(time.time() - started, 1), empty_speeches=empty,
                max_abs_by_layer={str(k): round(v, 3) for k, v in max_abs.items()} if done == 0 else "see earlier log for resumed blocks")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("layer_*.npy"))
    banner(logger, f"ACTIVATION COLLECTION COMPLETE: {args.model} ({tag})", {
        "speeches": f"{n:,} ({empty:,} empty)",
        "files": f"{n_layers} layer files, {size / 1e9:,.1f} GB, in {out}",
        "largest |pooled activation|": f"{max(max_abs.values()):,.1f}" if done == 0 else "n/a (resumed run)",
        "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "checkpoint": checkpoint, "n_speeches": n, "n_layers": n_layers}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run_collection(args)
    except Exception:
        logger.exception("Activation collection failed")
        raise


if __name__ == "__main__":
    main()
