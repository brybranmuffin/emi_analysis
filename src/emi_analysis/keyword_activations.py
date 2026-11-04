"""Hidden states of the sampled keyword occurrences, in context, for one model (every layer).

    python -m emi_analysis.keyword_activations --model bert --run-dir checkpoints/bert-continued/run-job4757959 \\
        --occurrences emi/occurrences/corpus

Each occurrence is run inside the chunk that contains it, built exactly as the speech collection built it (same
token window, same [CLS]/[SEP] wrapping), so the keyword's state is computed in the context its speech vector saw.
The occurrence's vector is the mean over the tokens overlapping its characters (multi-piece words averaged).
A span that crosses a chunk boundary keeps the pieces in the chunk of its first piece (counted in meta.json).
Each keyword is also run on its own ("alone": the paper's way of embedding the dictionary) for the context ablation.

Output in emi/keywords/<model>/<checkpoint run>/step<N>/<occurrence tag>/:
    occ_layer_XX.npy    [n_occurrences x hidden] float32, rows in occurrences.npz order
    alone_layer_XX.npy  [n_keywords x hidden] float32, rows in keywords.json order
    occ_sae_<name>.npy  with --sae: mean over the occurrence's pieces of SAE(piece) at that SAE's layer
    meta.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .collect_activations import choose_checkpoint
from .runinfo import banner, environment_info, format_duration, setup_logging
from .train import autocast_for, collate, load_jsonl_dataset, load_model_and_tokenizer, parse_config

logger = logging.getLogger("emi_analysis")


def resolve_precision(precision: str, model: str) -> str:
    """'auto' is strict float32 for ModernBERT (bf16 noise swamps its late layers) and bf16 otherwise."""
    if precision != "auto":
        return precision
    return "fp32" if model == "modernbert" else "bf16"


def precision_context(config, device: torch.device, precision: str):
    autocast, _ = autocast_for(config, device)
    if precision == "bf16":
        return autocast
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        torch.backends.cudnn.allow_tf32 = precision == "tf32"
    return contextlib.nullcontext


def plan_units(texts: dict[int, str], rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, tokenizer, window: int,
               wrap: bool) -> tuple[list[list[int]], list[tuple[int, list[int]]], int, list[int]]:
    """Chunks to run and, per occurrence, (unit index, positions in that unit).

    Returns (units, per-occurrence placement, straddling count, occurrences with no overlapping token).
    """
    unique_rows = sorted(texts)
    units: list[list[int]] = []
    unit_of: dict[tuple[int, int], int] = {}
    placement: list[tuple[int, list[int]]] = [(-1, [])] * len(rows)
    straddling, missing = 0, []
    by_row: dict[int, list[int]] = {}
    for i, row in enumerate(rows.tolist()):
        by_row.setdefault(row, []).append(i)
    for start in range(0, len(unique_rows), 1000):
        batch_rows = unique_rows[start : start + 1000]
        encoded = tokenizer([texts[r] for r in batch_rows], add_special_tokens=False, return_offsets_mapping=True,
                            verbose=False)
        for row, ids, offsets in zip(batch_rows, encoded["input_ids"], encoded["offset_mapping"]):
            offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
            for i in by_row[row]:
                pieces = np.flatnonzero((offsets[:, 0] < ends[i]) & (offsets[:, 1] > starts[i]))
                if len(pieces) == 0:
                    missing.append(i)
                    continue
                chunk = int(pieces[0]) // window
                inside = pieces[pieces // window == chunk]
                straddling += int(len(inside) < len(pieces))
                key = (row, chunk)
                if key not in unit_of:
                    body = ids[chunk * window : (chunk + 1) * window]
                    unit_of[key] = len(units)
                    units.append([tokenizer.cls_token_id, *body, tokenizer.sep_token_id] if wrap else list(body))
                placement[i] = (unit_of[key], (inside - chunk * window + (1 if wrap else 0)).tolist())
    return units, placement, straddling, missing


def run_units(base, units: list[list[int]], placement, n_out: int, pad_id: int, batch_chunks: int, device, context,
              sinks: list) -> None:
    """Forward every unit and write each occurrence's mean over its positions into the sinks.

    A sink is (transform, arrays): transform(states) -> list of [batch, seq, dim] tensors, one per array.
    """
    by_unit: dict[int, list[int]] = {}
    for i, (unit, _) in enumerate(placement):
        if unit >= 0:
            by_unit.setdefault(unit, []).append(i)
    order = sorted(range(len(units)), key=lambda u: len(units[u]))
    for start in range(0, len(order), batch_chunks):
        group = order[start : start + batch_chunks]
        batch = collate([units[u] for u in group], pad_id)
        with torch.no_grad(), context():
            states = base(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                          output_hidden_states=True).hidden_states
        occ_ids, b_idx, pos, owner = [], [], [], []
        for b, unit in enumerate(group):
            for i in by_unit.get(unit, []):
                positions = placement[i][1]
                b_idx += [b] * len(positions)
                pos += positions
                owner += [len(occ_ids)] * len(positions)
                occ_ids.append(i)
        if not occ_ids:
            continue
        b_t, p_t = torch.tensor(b_idx, device=device), torch.tensor(pos, device=device)
        o_t = torch.tensor(owner, device=device)
        counts = torch.zeros(len(occ_ids), device=device).index_add_(0, o_t, torch.ones(len(owner), device=device))
        for transform, arrays in sinks:
            for tensor, array in zip(transform(states, b_t, p_t), arrays):
                sums = torch.zeros((len(occ_ids), tensor.shape[-1]), device=device, dtype=torch.float32)
                sums.index_add_(0, o_t, tensor.float())
                array[np.asarray(occ_ids)] = (sums / counts[:, None]).cpu().numpy()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hidden states of sampled keyword occurrences, in context.")
    parser.add_argument("--model", required=True, choices=["bert", "gpt2", "modernbert", "qwen3"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--occurrences", required=True, help="Output folder of keyword_occurrences.")
    parser.add_argument("--precision", choices=["auto", "bf16", "tf32", "fp32"], default="auto")
    parser.add_argument("--sae", action="append", default=[], help="SAE checkpoint(s): also store SAE features of each occurrence.")
    parser.add_argument("--output-root", default="emi/keywords")
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    import datasets

    datasets.disable_progress_bars()
    started = time.time()
    config = parse_config(args.config or f"configs/{args.model}_continued_pretraining.json")
    checkpoint = choose_checkpoint(args, config)
    occ_dir = Path(args.occurrences)
    out = Path(args.output_root) / args.model / checkpoint["run"] / f"step{checkpoint['step']}" / occ_dir.name
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "keywords.log", mode="a")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    precision = resolve_precision(args.precision, args.model)
    context = precision_context(config, device, precision)

    occ = np.load(occ_dir / "occurrences.npz")
    occ_meta = json.loads((occ_dir / "meta.json").read_text(encoding="utf-8"))
    words = json.loads((occ_dir / "keywords.json").read_text(encoding="utf-8"))["words"]
    model, tokenizer = load_model_and_tokenizer(config, checkpoint["path"], device)
    model.eval()
    base = model.base_model
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    wrap = tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
    window = config.max_length - (2 if wrap else 0)

    speeches = datasets.concatenate_datasets([load_jsonl_dataset(p, config.datasets_cache_dir) for p in occ_meta["inputs"]])
    rows = occ["row"].astype(np.int64)
    unique_rows = np.unique(rows)
    texts = dict(zip(unique_rows.tolist(), speeches.select(unique_rows.tolist())["text"]))
    banner(logger, f"KEYWORD ACTIVATIONS: {args.model}", {
        "checkpoint": checkpoint["path"], "why this checkpoint": checkpoint["reason"],
        "occurrences": f"{len(rows):,} in {len(unique_rows):,} speeches ({occ_dir})",
        "keywords": len(words), "layers": f"{n_layers} x {hidden}",
        "precision": precision, "output": out, **environment_info(),
    })

    t0 = time.time()
    units, placement, straddling, missing = plan_units(texts, rows, occ["start"], occ["end"], tokenizer, window, wrap)
    logger.info(f"Planned {len(units):,} chunks in {format_duration(time.time() - t0)}; "
                f"{straddling:,} spans cross a chunk boundary; {len(missing):,} spans have no token")
    occ_arrays = [np.lib.format.open_memmap(out / f"occ_layer_{i:02d}.npy", mode="w+", dtype=np.float32,
                                            shape=(len(rows), hidden)) for i in range(n_layers)]
    for array in occ_arrays:
        array[:] = np.nan
    sinks = [(lambda states, b, p: [s[b, p] for s in states], occ_arrays)]
    sae_outputs = []
    for sae_path in args.sae:
        from .sae import load_sae

        sae, sae_meta = load_sae(sae_path, device)
        layer = int(sae_meta["layer"])
        name = Path(sae_path).parent.name if Path(sae_path).name == "sae.pt" else Path(sae_path).stem
        array = np.lib.format.open_memmap(out / f"occ_sae_{name}.npy", mode="w+", dtype=np.float32,
                                          shape=(len(rows), sae.n_features))
        array[:] = np.nan
        sinks.append(((lambda states, b, p, sae=sae, layer=layer: [sae.encode(states[layer][b, p].float())]), [array]))
        sae_outputs.append({"path": str(sae_path), "name": name, "layer": layer, "file": f"occ_sae_{name}.npy"})
    run_units(base, units, placement, len(rows), tokenizer.pad_token_id, args.batch_chunks, device, context, sinks)
    for array in occ_arrays:
        array.flush()

    alone_ids = [tokenizer(w, add_special_tokens=False)["input_ids"] for w in words]
    alone_units = [[tokenizer.cls_token_id, *ids, tokenizer.sep_token_id] if wrap else ids for ids in alone_ids]
    alone_place = [(k, list(range(1 if wrap else 0, (1 if wrap else 0) + len(ids)))) for k, ids in enumerate(alone_ids)]
    alone_arrays = [np.zeros((len(words), hidden), dtype=np.float32) for _ in range(n_layers)]
    alone_sinks = [(sinks[0][0], alone_arrays)]
    alone_sae = []
    for (transform, (occ_array,)), info in zip(sinks[1:], sae_outputs):
        array = np.zeros((len(words), occ_array.shape[1]), dtype=np.float32)
        alone_sinks.append((transform, [array]))
        alone_sae.append((info, array))
    run_units(base, alone_units, alone_place, len(words), tokenizer.pad_token_id, args.batch_chunks, device, context, alone_sinks)
    for i, array in enumerate(alone_arrays):
        np.save(out / f"alone_layer_{i:02d}.npy", array)
    for info, array in alone_sae:
        info["alone_file"] = f"alone_sae_{info['name']}.npy"
        np.save(out / info["alone_file"], array)

    meta = {
        "model": args.model, "model_name": config.model_name, "checkpoint": checkpoint, "precision": precision,
        "occurrences": str(occ_dir), "n_occurrences": int(len(rows)), "n_keywords": len(words), "words": words,
        "n_layers": n_layers, "hidden_size": hidden, "window": window, "wrapped": wrap, "n_chunks_run": len(units),
        "straddling_spans": straddling, "missing_spans": missing, "sae": sae_outputs,
        "layer_index": "0 = embedding output; i = output of layer i; last = final output (final norm applied)",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    if missing:
        logger.warning(f"{len(missing):,} occurrence spans map to no token; their rows are NaN")
    banner(logger, f"KEYWORD ACTIVATIONS COMPLETE: {args.model}", {
        "occurrences": f"{len(rows):,} ({straddling:,} straddling a chunk boundary, {len(missing):,} without tokens)",
        "files": f"{n_layers} occurrence layers + {n_layers} alone layers" + (f" + {len(sae_outputs)} SAE" if sae_outputs else ""),
        "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "n_occurrences": int(len(rows))}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run(args)
    except Exception:
        logger.exception("Keyword activation collection failed")
        raise


if __name__ == "__main__":
    main()
