"""Sample the dictionary keywords as they occur in the speeches: one shared sample for every model.

    python -m emi_analysis.keyword_occurrences --config configs/bert_continued_pretraining.json

Rows follow the activation collections' order (the input files concatenated; default train then held-out).
Only training speeches contribute occurrences, so the held-out speeches stay uncontaminated.

Per keyword and era the sample is ranked (bottom-k of a seeded hash), so the first n of a larger sample are
exactly a smaller sample: rank < 100 is the main sample and ranks < 25/50/200 serve the sample-size ablation.
A separate ranking per keyword over all eras gives the unbalanced-sampling ablation. Each speech contributes at
most one occurrence of each keyword (a seeded choice among its matches). Matches are exact, case-insensitive,
whole words on character spans, so each model later maps the same spans to its own tokens.

Outputs in emi/occurrences/<tag>/:
    occurrences.npz   row, keyword, pole (0 evidence / 1 intuition), era, start, end, rank_balanced, rank_unbalanced
    speech_stats.npz  word_count, top100_frac, heldout, congress (one entry per collection row)
    speech_ids.txt    the rows' speech IDs (lets a collection of a subset, e.g. only held-out speeches, be aligned)
    keywords.json     the keywords in index order with their poles
    meta.json         counts per keyword and era, shortfalls, rare keywords
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .emi_common import N_ERAS, era_label, era_of, hash_key, hash_key_int, load_lexicons, top100_words, word_stats
from .runinfo import banner, environment_info, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

RARE_TOTAL = 20


def keyword_pattern(words: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![\w'])(?:{alternatives})(?![\w'])", re.IGNORECASE)


def bottom_k_ranks(groups: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """Rank of each item within its group by ascending key; -1 beyond the first k."""
    ranks = np.full(len(groups), -1, dtype=np.int32)
    if len(groups) == 0:
        return ranks
    order = np.lexsort((keys, groups))
    sorted_groups = groups[order]
    is_start = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    start_pos = np.maximum.accumulate(np.where(is_start, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - start_pos
    keep = rank < k
    ranks[order[keep]] = rank[keep]
    return ranks


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample era-balanced keyword occurrences for contextual EMI poles.")
    parser.add_argument("--config", default="configs/bert_continued_pretraining.json",
                        help="Any training config (for data paths, lexicons and the datasets cache).")
    parser.add_argument("--input", action="append", default=None,
                        help="Speech JSONL files in collection order (default: train then held-out).")
    parser.add_argument("--heldout", action="append", default=None,
                        help="Inputs whose speeches never contribute occurrences (default: the held-out file).")
    parser.add_argument("--per-era", type=int, default=200, help="Ranked occurrences kept per keyword and era.")
    parser.add_argument("--unbalanced", type=int, default=800, help="Occurrences per keyword for the unbalanced ablation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="emi/occurrences")
    parser.add_argument("--tag", default="corpus")
    parser.add_argument("--num-proc", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> dict:
    import datasets

    from .train import load_jsonl_dataset, parse_config

    datasets.disable_progress_bars()
    started = time.time()
    config = parse_config(args.config)
    inputs = [Path(p) for p in (args.input or [config.train_data_path, config.heldout_data_path])]
    heldout_inputs = {str(Path(p)) for p in (args.heldout or [config.heldout_data_path])}
    out = Path(args.output_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "occurrences.log", mode="a")
    num_proc = args.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    evidence, intuition = load_lexicons(config)
    words = evidence + intuition
    poles = np.asarray([0] * len(evidence) + [1] * len(intuition), dtype=np.int8)
    index_of = {w: i for i, w in enumerate(words)}
    pattern = keyword_pattern(words)
    top100 = top100_words()

    parts = [load_jsonl_dataset(p, config.datasets_cache_dir) for p in inputs]
    speeches = datasets.concatenate_datasets(parts)
    n = len(speeches)
    heldout = np.concatenate([np.full(len(d), str(p) in heldout_inputs) for p, d in zip(inputs, parts)])
    congress = np.asarray(speeches["congress"], dtype=np.int16)
    banner(logger, "KEYWORD OCCURRENCES", {
        "inputs": ", ".join(f"{p} ({len(d):,}{', held out' if str(p) in heldout_inputs else ''})" for p, d in zip(inputs, parts)),
        "keywords": f"{len(evidence)} evidence + {len(intuition)} intuition",
        "sample": f"{args.per_era} ranked per keyword and era (eras of 9 Congresses), {args.unbalanced} per keyword unbalanced",
        "seed": args.seed,
        "output": out,
        **environment_info(),
    })
    proc = num_proc if num_proc > 1 else None

    def find(batch, indices):
        rows, kws, starts, ends = [], [], [], []
        for row, text in zip(indices, batch["text"]):
            found: dict[int, list[tuple[int, int]]] = {}
            for match in pattern.finditer(text):
                found.setdefault(index_of[match.group(0).lower()], []).append(match.span())
            for kw, spans in found.items():
                start, end = spans[hash_key_int(args.seed, row, kw) % len(spans)]
                rows.append(row), kws.append(kw), starts.append(start), ends.append(end)
        return {"row": rows, "keyword": kws, "start": starts, "end": ends}

    def stats(batch):
        counted = [word_stats(text, top100) for text in batch["text"]]
        return {"word_count": [c for c, _ in counted], "top100_frac": [f for _, f in counted]}

    t0 = time.time()
    matches = speeches.map(find, batched=True, with_indices=True, num_proc=proc, remove_columns=speeches.column_names,
                           features=datasets.Features({k: datasets.Value("int64") for k in ("row", "keyword", "start", "end")}))
    per_speech = speeches.map(stats, batched=True, num_proc=proc, remove_columns=speeches.column_names,
                              features=datasets.Features({"word_count": datasets.Value("int32"),
                                                          "top100_frac": datasets.Value("float32")}))
    found = matches.with_format("numpy")[:]
    stat_cols = per_speech.with_format("numpy")[:]
    logger.info(f"Scanned {n:,} speeches in {format_duration(time.time() - t0)}: "
                f"{len(found['row']):,} (speech, keyword) pairs with at least one match")

    row, keyword = found["row"].astype(np.int64), found["keyword"].astype(np.int64)
    era = era_of(congress[row]) if len(row) else np.zeros(0, dtype=np.int64)
    eligible = ~heldout[row] & (era >= 0)
    rank_balanced = np.full(len(row), -1, dtype=np.int32)
    rank_unbalanced = np.full(len(row), -1, dtype=np.int32)
    idx = np.flatnonzero(eligible)
    rank_balanced[idx] = bottom_k_ranks(keyword[idx] * N_ERAS + era[idx], hash_key(args.seed, 1, row[idx], keyword[idx]),
                                        args.per_era)
    rank_unbalanced[idx] = bottom_k_ranks(keyword[idx], hash_key(args.seed, 2, row[idx], keyword[idx]), args.unbalanced)
    keep = (rank_balanced >= 0) | (rank_unbalanced >= 0)
    order = np.flatnonzero(keep)[np.argsort(row[keep], kind="stable")]
    np.savez(out / "occurrences.npz", row=row[order], keyword=keyword[order].astype(np.int32),
             pole=poles[keyword[order]], era=era[order].astype(np.int8),
             start=found["start"][order].astype(np.int64), end=found["end"][order].astype(np.int64),
             rank_balanced=rank_balanced[order], rank_unbalanced=rank_unbalanced[order])
    np.savez(out / "speech_stats.npz", word_count=stat_cols["word_count"].astype(np.int32),
             top100_frac=stat_cols["top100_frac"].astype(np.float32), heldout=heldout, congress=congress)
    (out / "speech_ids.txt").write_text("\n".join(str(s) for s in speeches["speech_id"]) + "\n", encoding="utf-8")
    (out / "keywords.json").write_text(json.dumps({"words": words, "poles": poles.tolist(),
                                                   "evidence": evidence, "intuition": intuition}, indent=2) + "\n")

    available = np.zeros((len(words), N_ERAS), dtype=np.int64)
    np.add.at(available, (keyword[idx], era[idx]), 1)
    sampled = np.minimum(available, args.per_era)
    shortfalls = [{"word": words[k], "era": e, "era_congresses": era_label(e), "available": int(available[k, e])}
                  for k in range(len(words)) for e in range(N_ERAS) if available[k, e] < args.per_era]
    rare = [words[k] for k in range(len(words)) if sampled[k].sum() < RARE_TOTAL]
    meta = {
        "inputs": [str(p) for p in inputs], "heldout_inputs": sorted(heldout_inputs), "n_speeches": n,
        "n_heldout": int(heldout.sum()), "seed": args.seed, "per_era": args.per_era, "unbalanced": args.unbalanced,
        "eras": [era_label(e) for e in range(N_ERAS)], "n_evidence": len(evidence), "n_intuition": len(intuition),
        "n_occurrences": int(keep.sum()),
        "sampled_per_keyword_era": {words[k]: sampled[k].tolist() for k in range(len(words))},
        "available_per_keyword_era": {words[k]: available[k].tolist() for k in range(len(words))},
        "shortfalls": shortfalls, "flagged_rare": rare,
        "unscorable_speeches": int((~((stat_cols["word_count"] > 10) & (stat_cols["top100_frac"] >= 0.05))).sum()),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if len(evidence) != 49 or len(intuition) != 35:
        logger.warning(f"Lexicons have {len(evidence)} evidence and {len(intuition)} intuition keywords "
                       f"(the paper describes 49 and 35); using the CSVs as given.")
    for word in rare:
        logger.warning(f"Keyword '{word}' has fewer than {RARE_TOTAL} sampled occurrences in total")
    banner(logger, "KEYWORD OCCURRENCES COMPLETE", {
        "occurrences kept": f"{int(keep.sum()):,}",
        "keyword-era cells short of the target": f"{len(shortfalls):,} of {len(words) * N_ERAS:,}",
        "rare keywords": ", ".join(rare) or "none",
        "unscorable speeches (paper filters)": f"{meta['unscorable_speeches']:,}",
        "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "n_occurrences": int(keep.sum())}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run(args)
    except Exception:
        logger.exception("Keyword occurrence sampling failed")
        raise


if __name__ == "__main__":
    main()
