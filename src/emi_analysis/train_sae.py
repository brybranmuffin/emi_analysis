"""Train a TopK SAE on one layer of one model: on individual tokens (--kind token) or on speech means (--kind speech).

    python -m emi_analysis.train_sae --model bert --kind token \\
        --activations activations/bert/run-job4757959/step350847/corpus \\
        --layer-from results/emi/raw/bert/run-job4757959/step350847/meta.json

The checkpoint is the one the speech activations were collected from (their meta.json), so the SAE, the keyword
activations and the speech vectors all come from the same model. Training data excludes the held-out speeches and is
era-balanced: token SAEs get the same number of tokens from every era (speeches drawn at random within each era until
its share is reached); speech SAEs draw each training example from a uniformly chosen era.

Token kind: the speeches are chunked and packed exactly as the activation collection does; token states at the layer
(special tokens and padding excluded) stream into a shuffling buffer and train the SAE directly, nothing token-level
is stored. Diagnostics on held-out speeches: fraction of variance unexplained (FVU), dead features, and the held-out
LM loss with the layer's real-token states replaced by their SAE reconstruction (vs the original loss).
Speech kind: trains on the stored speech vectors of the layer; diagnostics are FVU and dead features.

Output in sae/<model>/<run>/step<N>/<kind>/layer<L>_k<k>_x<expansion>_seed<seed>/:
    sae.pt, metrics.json, history.csv, train_sae.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .emi_common import N_ERAS, era_of
from .runinfo import banner, environment_info, format_duration, setup_logging
from .sae import TopKSAE

logger = logging.getLogger("emi_analysis")


# ------------------------------------------------------------------ data selection

def era_balanced_rows(congress, counts, heldout, total_tokens: int, seed: int) -> tuple[np.ndarray, dict[int, int]]:
    """Training rows giving each era (with training speeches) an equal share of total_tokens, shuffled together.

    Within an era, speeches are taken in a seeded random order until the era's share is reached; an era with fewer
    tokens contributes all of them.
    """
    rng = np.random.default_rng(seed)
    era = era_of(congress)
    counts = np.asarray(counts, dtype=np.int64)
    eligible = ~np.asarray(heldout, dtype=bool) & (era >= 0) & (counts > 0)
    present = [e for e in range(N_ERAS) if (eligible & (era == e)).any()]
    if not present:
        raise ValueError("No training speeches with tokens in any era.")
    budget = total_tokens / len(present)
    chosen, per_era = [], {}
    for e in present:
        rows = rng.permutation(np.flatnonzero(eligible & (era == e)))
        cumulative = np.cumsum(counts[rows])
        stop = min(len(rows), int(np.searchsorted(cumulative, budget)) + 1)
        chosen.append(rows[:stop])
        per_era[e] = int(cumulative[stop - 1])
    return rng.permutation(np.concatenate(chosen)), per_era


def heldout_flags(act: Path, heldout_path: str) -> np.ndarray:
    ids = (act / "speech_ids.txt").read_text(encoding="utf-8").split()
    with open(heldout_path, encoding="utf-8") as handle:
        held = {str(json.loads(line)["speech_id"]) for line in handle if line.strip()}
    return np.fromiter((i in held for i in ids), dtype=bool, count=len(ids))


# ------------------------------------------------------------------ model plumbing

def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def find_layer_module(base, batch: dict, layer: int):
    """The module whose output is hidden_states[layer] (the last such module to finish, so replacing its output
    changes everything downstream)."""
    recorded = []
    handles = [m.register_forward_hook(lambda mod, inp, out, name=name: recorded.append((name, mod, _first_tensor(out))))
               for name, m in base.named_modules() if name]
    try:
        with torch.no_grad():
            target = base(**{k: v for k, v in batch.items() if not k.startswith("_")}, output_hidden_states=True).hidden_states[layer]
    finally:
        for handle in handles:
            handle.remove()
    matches = [mod for _, mod, out in recorded
               if out is not None and out.shape == target.shape and (out is target or torch.equal(out, target))]
    if not matches:
        raise RuntimeError(f"No module outputs hidden_states[{layer}]; cannot splice the SAE in.")
    return matches[-1]


def stream_token_states(base, tokenizer, config, speeches, rows: np.ndarray, layer: int, device, context,
                        tokens_per_batch: int, block_speeches: int, num_proc: int | None, limit: int | None = None):
    """Yield [n_tokens, hidden] float32 tensors of real, non-special token states at the layer, speeches in row order."""
    from .collect_activations import batches_for_block, position_speech_index, tokenize_speeches
    from .packing import build_packed_batch, model_inputs
    from .train import collate

    wrap = tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
    window = config.max_length - (2 if wrap else 0)
    special = torch.as_tensor(tokenizer.all_special_ids)
    produced = 0
    for start in range(0, len(rows), block_speeches):
        block = speeches.select(rows[start : start + block_speeches].tolist())
        chunks = tokenize_speeches(block, tokenizer, window, num_proc)
        cols = chunks.with_format(None)[:]
        local = np.asarray(cols["speech"])
        for units, unit_speech in batches_for_block(cols["input_ids"], local, config, tokens_per_batch):
            batch = build_packed_batch(units, tokenizer.pad_token_id) if config.packing else collate(units, tokenizer.pad_token_id)
            keep = (position_speech_index(batch, unit_speech, config.packing, special) >= 0).reshape(-1).to(device)
            with torch.no_grad(), context():
                states = base(**model_inputs(batch, base, config.objective, device, labels=False),
                              output_hidden_states=True).hidden_states
            tokens = states[layer].reshape(-1, states[layer].shape[-1])[keep].float()
            produced += len(tokens)
            yield tokens
            if limit is not None and produced >= limit:
                return


def lm_loss(model, batches, objective: str, device, context, splice=None) -> float:
    """train.evaluate's held-out loss, optionally with splice=(module, sae, special_ids) replacing real-token states."""
    from .packing import model_inputs

    handle, state = None, {}
    if splice is not None:
        module, sae, special = splice

        def hook(mod, inp, out):
            hidden = _first_tensor(out)
            mask = state["mask"].to(hidden.device)
            replaced = hidden.clone()
            replaced[mask] = sae(hidden[mask].float())[0].to(hidden.dtype)
            return replaced if isinstance(out, torch.Tensor) else (replaced, *out[1:])

        handle = module.register_forward_hook(hook)
    total, count = 0.0, 0
    model.eval()
    try:
        for batch in batches:
            if splice is not None:
                real = batch["_real"] if "_real" in batch else batch["attention_mask"].bool()
                state["mask"] = real & ~torch.isin(batch["input_ids"], splice[2])
            labels = batch["labels"].to(device)
            with torch.no_grad(), context():
                logits = model(**model_inputs(batch, model, objective, device, labels=False)).logits
            if objective == "clm":
                logits, labels = logits[:, :-1], labels[:, 1:]
            total += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                                     ignore_index=-100, reduction="sum").item()
            count += int((labels != -100).sum())
    finally:
        if handle is not None:
            handle.remove()
    nll = total / max(1, count)
    return math.exp(nll) if objective == "clm" else nll


# ------------------------------------------------------------------ training

class Trainer:
    def __init__(self, sae: TopKSAE, lr: float, dead_after: int, device):
        self.sae, self.device, self.dead_after = sae, device, dead_after
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        self.since_fired = torch.zeros(sae.n_features, dtype=torch.long, device=device)
        self.steps, self.seen = 0, 0
        self.history: list[dict] = []

    def step(self, x: torch.Tensor) -> dict:
        dead = self.since_fired > self.dead_after
        loss, parts = self.sae.loss(x, dead)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.sae.normalize_decoder()
        self.since_fired += len(x)
        self.since_fired[parts["fired"]] = 0
        self.steps += 1
        self.seen += len(x)
        self.last = {"step": self.steps, "seen": self.seen, "fvu": parts["fvu"], "aux": parts["aux"],
                "dead_fraction": float(dead.float().mean())}
        return self.last

    def finish(self) -> None:
        if self.steps and (not self.history or self.history[-1]["step"] != self.steps):
            self.history.append(self.last)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TopK SAE on token or speech activations of one layer.")
    parser.add_argument("--model", required=True, choices=["bert", "gpt2", "modernbert", "qwen3"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--kind", required=True, choices=["token", "speech"])
    parser.add_argument("--activations", required=True, help="Speech activation collection (checkpoint, row order, counts).")
    layer = parser.add_mutually_exclusive_group(required=True)
    layer.add_argument("--layer", type=int)
    layer.add_argument("--layer-from", help="emi_score meta.json: use its primary_layer.")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=200_000_000, help="Training tokens (token kind).")
    parser.add_argument("--samples", type=int, default=50_000_000, help="Training examples drawn (speech kind).")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--buffer-tokens", type=int, default=2_000_000, help="Shuffling buffer (token kind).")
    parser.add_argument("--eval-tokens", type=int, default=200_000)
    parser.add_argument("--eval-chunks", type=int, default=256, help="Held-out chunks for the spliced LM loss.")
    parser.add_argument("--precision", choices=["auto", "bf16", "tf32", "fp32"], default="auto")
    parser.add_argument("--tokens-per-batch", type=int, default=65_536)
    parser.add_argument("--block-speeches", type=int, default=5_000)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output-root", default="sae")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def _write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run(args: argparse.Namespace) -> dict:
    import datasets

    from .keyword_activations import precision_context, resolve_precision
    from .train import build_eval_batches, load_jsonl_dataset, load_model_and_tokenizer, parse_config

    datasets.disable_progress_bars()
    started = time.time()
    torch.manual_seed(args.seed)
    config = parse_config(args.config or f"configs/{args.model}_continued_pretraining.json")
    act = Path(args.activations)
    act_meta = json.loads((act / "meta.json").read_text(encoding="utf-8"))
    checkpoint = act_meta["checkpoint"]
    layer = args.layer if args.layer is not None else int(json.loads(Path(args.layer_from).read_text())["primary_layer"])
    name = f"layer{layer:02d}_k{args.k}_x{args.expansion}_seed{args.seed}"
    out = Path(args.output_root) / args.model / checkpoint["run"] / f"step{checkpoint['step']}" / args.kind / name
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "train_sae.log", mode="a")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")
    num_proc = args.num_proc or config.num_proc

    congress = np.load(act / "congress.npy")
    counts = np.load(act / "token_counts.npy")
    heldout = heldout_flags(act, config.heldout_data_path)
    hidden = int(act_meta["hidden_size"])
    sae = TopKSAE(hidden, hidden * args.expansion, args.k).to(device)
    metrics: dict = {"model": args.model, "kind": args.kind, "layer": layer, "k": args.k, "expansion": args.expansion,
                     "n_features": sae.n_features, "seed": args.seed, "checkpoint": checkpoint, "activations": str(act)}
    banner(logger, f"SAE TRAINING: {args.model} layer {layer} ({args.kind})", {
        "checkpoint": checkpoint["path"], "activations": act,
        "sae": f"TopK k={args.k}, {sae.n_features:,} features ({args.expansion}x {hidden})",
        "budget": f"{args.tokens:,} tokens" if args.kind == "token" else f"{args.samples:,} speech vectors",
        "held-out speeches excluded": f"{int(heldout.sum()):,}", "output": out, **environment_info(),
    })

    if args.kind == "token":
        precision = resolve_precision(args.precision, args.model)
        context = precision_context(config, device, precision)
        model, tokenizer = load_model_and_tokenizer(config, checkpoint["path"], device)
        model.eval()
        base = model.base_model
        speeches = datasets.concatenate_datasets([load_jsonl_dataset(p, config.datasets_cache_dir) for p in act_meta["inputs"]])
        rows, per_era = era_balanced_rows(congress, counts, heldout, args.tokens, args.seed)
        metrics.update(precision=precision, tokens_per_era={str(e): per_era.get(e, 0) for e in range(N_ERAS)},
                       training_speeches=int(len(rows)))
        logger.info(f"{len(rows):,} training speeches; tokens per era: "
                    + ", ".join(f"{e}: {per_era.get(e, 0):,}" for e in range(N_ERAS)))
        trainer = Trainer(sae, args.lr, dead_after=min(10_000_000, max(args.batch_size, args.tokens // 20)), device=device)
        # Blocks are small: tokenizing one in-process avoids starting worker processes hundreds of times.
        stream = stream_token_states(base, tokenizer, config, speeches, rows, layer, device, context,
                                     args.tokens_per_batch, args.block_speeches, None)
        buffer, filled, normalized = [], 0, False

        def train_buffer(tokens: torch.Tensor) -> None:
            nonlocal normalized
            if not normalized:
                sae.set_normalization(tokens.mean(dim=0), tokens[: 200_000])
                normalized = True
                share = torch.sort(tokens.var(dim=0), descending=True).values
                metrics["top5_dim_variance_share"] = float(share[:5].sum() / share.sum())
            order = torch.randperm(len(tokens), device=tokens.device)
            for s in range(0, len(tokens) - args.batch_size + 1, args.batch_size):
                record = trainer.step(tokens[order[s : s + args.batch_size]])
                if trainer.steps % args.log_every == 0:
                    trainer.history.append(record)
                    logger.info(f"step {record['step']:,} | tokens {record['seen']:,}/{args.tokens:,} | FVU {record['fvu']:.4f} "
                                f"| aux {record['aux']:.4f} | dead {record['dead_fraction']:.2%} | "
                                f"elapsed {format_duration(time.time() - started)}")
                if trainer.seen >= args.tokens:
                    return

        for tokens in stream:
            buffer.append(tokens.to(device))
            filled += len(tokens)
            if filled >= args.buffer_tokens:
                train_buffer(torch.cat(buffer))
                buffer, filled = [], 0
            if trainer.seen >= args.tokens:
                break
        if buffer and trainer.seen < args.tokens:
            train_buffer(torch.cat(buffer))
        if trainer.seen < args.tokens:
            logger.warning(f"Data ran out after {trainer.seen:,} of {args.tokens:,} tokens")
        metrics["tokens_trained"] = trainer.seen

        held_rows = np.flatnonzero(heldout & (counts > 0))
        eval_tokens = torch.cat(list(stream_token_states(base, tokenizer, config, speeches, held_rows, layer, device, context,
                                                         args.tokens_per_batch, args.block_speeches, num_proc,
                                                         limit=args.eval_tokens)))[: args.eval_tokens]
        sae.eval()
        metrics["heldout_fvu"] = sae.fvu(eval_tokens)
        with torch.no_grad():
            fired = torch.zeros(sae.n_features, dtype=torch.bool, device=device)
            for s in range(0, len(eval_tokens), 8192):
                fired |= (sae.encode(eval_tokens[s : s + 8192]) > 0).any(dim=0)
        metrics["heldout_dead_fraction"] = float((~fired).float().mean())
        metrics["train_dead_fraction"] = float((trainer.since_fired > trainer.dead_after).float().mean())

        from .collect_activations import tokenize_speeches

        held_ds = speeches.select(held_rows[: max(1, args.eval_chunks)].tolist())
        wrap = tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
        chunks = tokenize_speeches(held_ds, tokenizer, config.max_length - (2 if wrap else 0), num_proc)
        chunks = chunks.select(range(min(len(chunks), args.eval_chunks)))
        batches, _ = build_eval_batches(chunks, config.objective, tokenizer, config, config.seed)
        from .packing import model_inputs

        module = find_layer_module(base, model_inputs(batches[0], base, config.objective, device, labels=False), layer)
        special = torch.as_tensor(tokenizer.all_special_ids)
        metrics["loss_metric"] = "perplexity" if config.objective == "clm" else "mlm_loss"
        metrics["original_loss"] = lm_loss(model, batches, config.objective, device, context)
        metrics["spliced_loss"] = lm_loss(model, batches, config.objective, device, context, splice=(module, sae, special))
        metrics["spliced_module"] = type(module).__name__
    else:
        layer_file = np.load(act / f"layer_{layer:02d}.npy", mmap_mode="r")
        era = era_of(congress)
        train_rows = np.flatnonzero(~heldout & (era >= 0) & (counts > 0))
        logger.info(f"Loading {len(train_rows):,} training speech vectors of layer {layer} into memory")
        data = np.asarray(layer_file[train_rows])  # stored dtype (float16) to halve memory
        train_era = era[train_rows]
        present = [e for e in range(N_ERAS) if (train_era == e).any()]
        by_era = [np.flatnonzero(train_era == e) for e in present]
        rng = np.random.default_rng(args.seed)

        def draw(n: int) -> torch.Tensor:
            eras = rng.integers(0, len(present), size=n)
            idx = np.concatenate([by_era[e][rng.integers(0, len(by_era[e]), size=int((eras == e).sum()))]
                                  for e in range(len(present))])
            return torch.from_numpy(data[idx].astype(np.float32)).to(device)

        sample = draw(min(200_000, max(args.batch_size, len(train_rows))))
        sae.set_normalization(sample.mean(dim=0), sample)
        share = torch.sort(sample.var(dim=0), descending=True).values
        metrics["top5_dim_variance_share"] = float(share[:5].sum() / share.sum())
        metrics["training_speeches"] = int(len(train_rows))
        metrics["speeches_per_era"] = {str(e): int((train_era == e).sum()) for e in range(N_ERAS)}
        trainer = Trainer(sae, args.lr, dead_after=min(1_000_000, max(args.batch_size, args.samples // 10)), device=device)
        while trainer.seen < args.samples:
            record = trainer.step(draw(args.batch_size))
            if trainer.steps % args.log_every == 0:
                trainer.history.append(record)
                logger.info(f"step {record['step']:,} | samples {record['seen']:,}/{args.samples:,} | FVU {record['fvu']:.4f} "
                            f"| dead {record['dead_fraction']:.2%} | elapsed {format_duration(time.time() - started)}")
        metrics["samples_trained"] = trainer.seen
        held_rows = np.flatnonzero(heldout & (counts > 0))
        held = torch.from_numpy(np.asarray(layer_file[held_rows], dtype=np.float32)).to(device)
        sae.eval()
        metrics["heldout_fvu"] = sae.fvu(held)
        with torch.no_grad():
            fired = (sae.encode(held) > 0).any(dim=0) if len(held) else torch.zeros(sae.n_features, dtype=torch.bool)
        metrics["heldout_dead_fraction"] = float((~fired).float().mean())
        metrics["train_dead_fraction"] = float((trainer.since_fired > trainer.dead_after).float().mean())

    sae.save(out / "sae.pt", {"model": args.model, "kind": args.kind, "layer": layer, "k": args.k,
                              "expansion": args.expansion, "seed": args.seed, "checkpoint": checkpoint,
                              "activations": str(act)})
    metrics.update(elapsed_seconds=round(time.time() - started, 1), finished_at=datetime.now().isoformat(timespec="seconds"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    trainer.finish()
    _write_history(out / "history.csv", trainer.history)
    banner(logger, f"SAE TRAINING COMPLETE: {args.model} layer {layer} ({args.kind})", {
        "held-out FVU": f"{metrics['heldout_fvu']:.4f}",
        "dead features (held-out / training)": f"{metrics['heldout_dead_fraction']:.2%} / {metrics['train_dead_fraction']:.2%}",
        **({"held-out LM loss (original -> spliced)": f"{metrics['original_loss']:.4f} -> {metrics['spliced_loss']:.4f}"}
           if "spliced_loss" in metrics else {}),
        "top-5 dimensions' share of variance": f"{metrics['top5_dim_variance_share']:.1%}",
        "output": out, "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "metrics": metrics}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run(args)
    except Exception:
        logger.exception("SAE training failed")
        raise


if __name__ == "__main__":
    main()
