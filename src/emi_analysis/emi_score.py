"""Contextual EMI per speech from stored activations, for one model and one representation.

    python -m emi_analysis.emi_score --model bert --representation raw \\
        --activations activations/bert/run-job4757959/step350847/corpus \\
        --keywords emi/keywords/bert/run-job4757959/step350847/corpus --occurrences emi/occurrences/corpus

Representations (docs/superpowers/specs/2026-11-04-contextual-emi-design.md):
    raw         the stored hidden states, every layer (or --layers)
    speech_sae  stored speech vectors encoded by an SAE trained on speech vectors (--sae); its layer only
    token_sae   per-speech means of token-level SAE features (--sae-features, a collection made with
                collect_activations --sae); occurrence features from keyword_activations --sae

Per layer: mu = token-weighted corpus mean (it equals the mean over all tokens, so it centers single-token poles
and speech means alike); keyword vector = mean over eras of the mean of that era's sampled occurrences; pole = mean
of its keyword vectors, minus mu; evidence/intuition score = cosine of (speech - mu) with each pole; then the paper's
length-bin adjustment, z-scores and EMI = z_ev - z_int. Speeches the paper would drop (<= 10 words, < 5% top-100
words) get NaN. The primary layer is the one whose poles best separate held-out keywords' occurrences
(leave-one-keyword-out AUC); it uses only the dictionary, never a validation target.

Output in results/emi/<representation>[/<sae name>]/<model>/<run>/step<N>/:
    layer_XX_{ev,int,emi}.npy, mu_layer_XX.npy, poles_layer_XX.npz, params_layer_XX.json
    series.csv (layer, congress, n, mean, lower, upper, ci_method), layer_selection.csv, meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .emi_common import N_ERAS, scorable
from .runinfo import banner, environment_info, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

BLOCK = 200_000
EXACT_BOOTSTRAP_MAX = 20_000


# ------------------------------------------------------------------ library

def _blocks(n_rows: int, block: int):
    for start in range(0, n_rows, block):
        yield start, min(start + block, n_rows)


def _read(source, rows: np.ndarray | None, start: int, stop: int, encode=None) -> np.ndarray:
    block = np.asarray(source[start:stop] if rows is None else source[rows[start:stop]], dtype=np.float32)
    return encode(block) if encode is not None else block


def token_weighted_mean(source, counts, rows: np.ndarray | None = None, encode=None, block: int = BLOCK) -> np.ndarray:
    """sum_i count_i * x_i / sum_i count_i over the rows (all rows by default), in float64."""
    counts = np.asarray(counts, dtype=np.float64)
    n = len(source) if rows is None else len(rows)
    total, weight = None, 0.0
    for start, stop in _blocks(n, block):
        x = _read(source, rows, start, stop, encode).astype(np.float64)
        c = counts[start:stop] if rows is None else counts[rows[start:stop]]
        part = c @ x
        total = part if total is None else total + part
        weight += c.sum()
    return total / weight


def keyword_vectors(occ: np.ndarray, keyword: np.ndarray, era: np.ndarray, n_keywords: int, use: np.ndarray,
                    balanced: bool = True) -> tuple[np.ndarray, list[int]]:
    """Per-keyword vector: mean over eras of the era's mean occurrence (balanced) or the plain mean.

    Occurrences with non-finite vectors (spans without tokens) are skipped. Keywords left with none are NaN and
    returned in the second value.
    """
    occ = np.asarray(occ, dtype=np.float64)
    use = np.asarray(use, dtype=bool) & np.isfinite(occ).all(axis=1)
    dim = occ.shape[1]
    out = np.full((n_keywords, dim), np.nan)
    missing = []
    for k in range(n_keywords):
        mine = use & (keyword == k)
        if not mine.any():
            missing.append(k)
            continue
        if balanced:
            era_means = [occ[mine & (era == e)].mean(axis=0) for e in range(N_ERAS) if (mine & (era == e)).any()]
            out[k] = np.mean(era_means, axis=0)
        else:
            out[k] = occ[mine].mean(axis=0)
    return out, missing


def poles(kw_vectors: np.ndarray, kw_pole: np.ndarray, mu: np.ndarray, keep: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Centered evidence and intuition poles: DDR mean of the available keyword vectors, minus mu."""
    ok = np.isfinite(kw_vectors).all(axis=1) & (np.ones(len(kw_vectors), dtype=bool) if keep is None else keep)
    p_ev = kw_vectors[ok & (kw_pole == 0)].mean(axis=0) - mu
    p_int = kw_vectors[ok & (kw_pole == 1)].mean(axis=0) - mu
    return p_ev, p_int


def _cos(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(p)
    return np.where(norms > 0, (x @ p) / np.where(norms > 0, norms, 1.0), np.nan)


def cosines(source, rows: np.ndarray | None, mu: np.ndarray, p_ev: np.ndarray, p_int: np.ndarray, encode=None,
            center: bool = True, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """cos(x - mu, P_ev) and cos(x - mu, P_int) per speech (no centering of x with center=False)."""
    n = len(source) if rows is None else len(rows)
    ev, intu = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
    for start, stop in _blocks(n, block):
        x = _read(source, rows, start, stop, encode).astype(np.float64)
        if center:
            x -= mu
        ev[start:stop], intu[start:stop] = _cos(x, p_ev), _cos(x, p_int)
    return ev, intu


def projection(source, rows: np.ndarray | None, mu: np.ndarray, p_ev: np.ndarray, p_int: np.ndarray, encode=None,
               center: bool = True, block: int = BLOCK) -> np.ndarray:
    """cos(x - mu, P_ev - P_int): one score on the evidence-minus-intuition direction."""
    direction = p_ev - p_int
    n = len(source) if rows is None else len(rows)
    out = np.empty(n, dtype=np.float32)
    for start, stop in _blocks(n, block):
        x = _read(source, rows, start, stop, encode).astype(np.float64)
        out[start:stop] = _cos(x - mu if center else x, direction)
    return out


def _length_bin(word_count) -> np.ndarray:
    """The paper's 10-word bins (10, 20], (20, 30], ... as integers."""
    return (np.asarray(word_count, dtype=np.int64) - 11) // 10


def standardize(score, word_count, ok, adjust: bool = True) -> tuple[np.ndarray, dict]:
    """Length-bin adjustment then z-score (population SD) over the scored speeches; NaN elsewhere."""
    score = np.asarray(score, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(score)
    bins = _length_bin(word_count)
    adjusted = np.full(len(score), np.nan)
    bin_means: dict[int, float] = {}
    if adjust and ok.any():
        uniq, inverse = np.unique(bins[ok], return_inverse=True)
        sums = np.bincount(inverse, weights=score[ok])
        means = sums / np.bincount(inverse)
        bin_means = {int(b): float(m) for b, m in zip(uniq, means)}
        adjusted[ok] = score[ok] - means[inverse]
    else:
        adjusted[ok] = score[ok]
    mean = float(adjusted[ok].mean()) if ok.any() else float("nan")
    sd = float(adjusted[ok].std()) if ok.any() else float("nan")
    z = np.full(len(score), np.nan)
    z[ok] = (adjusted[ok] - mean) / sd if sd > 0 else 0.0
    return z, {"adjust": adjust, "bin_means": bin_means, "mean": mean, "sd": sd}


def apply_standardize(score, word_count, params: dict) -> np.ndarray:
    """standardize() with parameters fitted on the corpus (for new texts, e.g. rated excerpts)."""
    score = np.asarray(score, dtype=np.float64)
    adjusted = score.copy()
    if params["adjust"]:
        keys = np.asarray(sorted(int(k) for k in params["bin_means"]))
        values = np.asarray([params["bin_means"][k] if k in params["bin_means"] else params["bin_means"][str(k)] for k in keys])
        bins = np.clip(_length_bin(word_count), keys.min(), keys.max())
        adjusted = score - values[np.clip(np.searchsorted(keys, bins), 0, len(keys) - 1)]
    return (adjusted - params["mean"]) / params["sd"]


def emi_from_cosines(ev, intu, word_count, ok, adjust: bool = True) -> tuple[np.ndarray, dict]:
    z_ev, p_ev = standardize(ev, word_count, ok, adjust)
    z_int, p_int = standardize(intu, word_count, ok, adjust)
    return (z_ev - z_int).astype(np.float32), {"ev": p_ev, "int": p_int}


def congress_series(emi, congress, weights=None, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Mean EMI per Congress with 95% bootstrap CIs over speeches.

    Congresses with more than 20,000 scored speeches use the normal approximation to the bootstrap of the mean
    (indistinguishable at that size); smaller ones are resampled exactly.
    """
    emi, congress = np.asarray(emi, dtype=np.float64), np.asarray(congress)
    weights = np.ones(len(emi)) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows = []
    for c in np.unique(congress):
        mine = (congress == c) & np.isfinite(emi) & (weights > 0)
        x, w = emi[mine], weights[mine]
        row = {"congress": int(c), "n": int(mine.sum()), "mean": float("nan"), "lower": float("nan"),
               "upper": float("nan"), "ci_method": ""}
        if len(x):
            mean = float(np.average(x, weights=w))
            row["mean"] = mean
            if len(x) > EXACT_BOOTSTRAP_MAX:
                w_n = w / w.sum()
                se = float(np.sqrt(np.sum(w_n ** 2 * (x - mean) ** 2)))
                row.update(lower=mean - 1.96 * se, upper=mean + 1.96 * se, ci_method="normal")
            elif len(x) > 1:
                picks = rng.integers(0, len(x), size=(n_boot, len(x)))
                boots = (x[picks] * w[picks]).sum(axis=1) / w[picks].sum(axis=1)
                row.update(lower=float(np.percentile(boots, 2.5)), upper=float(np.percentile(boots, 97.5)), ci_method="bootstrap")
        rows.append(row)
    return rows


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    from scipy.stats import rankdata

    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def layer_auc(occ: np.ndarray, keyword: np.ndarray, pole: np.ndarray, era: np.ndarray, mu: np.ndarray, use: np.ndarray,
              n_keywords: int) -> float:
    """Leave one keyword out: poles from the other keywords score its occurrences; AUC of evidence vs intuition."""
    occ = np.asarray(occ, dtype=np.float64)
    kw_pole = np.zeros(n_keywords, dtype=np.int8)
    for k in range(n_keywords):
        if (keyword == k).any():
            kw_pole[k] = pole[keyword == k][0]
    kw, _ = keyword_vectors(occ, keyword, era, n_keywords, use)
    ok = np.isfinite(kw).all(axis=1)
    scores, labels = [], []
    for k in np.flatnonzero(ok):
        keep = ok.copy()
        keep[k] = False
        if not (keep & (kw_pole == 0)).any() or not (keep & (kw_pole == 1)).any():
            continue
        p_ev, p_int = poles(kw, kw_pole, mu, keep)
        mine = use & (keyword == k) & np.isfinite(occ).all(axis=1)
        x = occ[mine] - mu
        scores.append(_cos(x, p_ev) - _cos(x, p_int))
        labels.append(np.full(int(mine.sum()), kw_pole[k] == 0))
    if not scores:
        return float("nan")
    return auc(np.concatenate(scores), np.concatenate(labels))


def aligned_stats(act: Path, occ_dir: Path) -> dict[str, np.ndarray]:
    """speech_stats.npz rows for each row of an activation collection (the corpus itself, or any subset of it)."""
    stats = dict(np.load(occ_dir / "speech_stats.npz"))
    act_ids_path, occ_ids_path = act / "speech_ids.txt", occ_dir / "speech_ids.txt"
    if act_ids_path.exists() and occ_ids_path.exists():
        act_ids = act_ids_path.read_text(encoding="utf-8").split()
        occ_ids = occ_ids_path.read_text(encoding="utf-8").split()
        if act_ids == occ_ids:
            return stats
        position = {s: i for i, s in enumerate(occ_ids)}
        missing = [s for s in act_ids if s not in position]
        if missing:
            raise RuntimeError(f"{len(missing):,} speeches of {act} are not in {occ_dir} (e.g. {missing[:3]}).")
        index = np.asarray([position[s] for s in act_ids], dtype=np.int64)
        return {k: v[index] for k, v in stats.items()}
    congress = np.load(act / "congress.npy")
    if len(congress) != len(stats["congress"]) or not np.array_equal(congress, stats["congress"]):
        raise RuntimeError(f"Row order of {act} does not match {occ_dir}: they must come from the same input files.")
    return stats


# ------------------------------------------------------------------ representations

class Representation:
    """Where speech vectors and occurrence vectors come from for one layer."""

    def __init__(self, name: str, source, counts: np.ndarray, occ: np.ndarray, encode=None, label: str = ""):
        self.name, self.source, self.counts, self.occ, self.encode, self.label = name, source, counts, occ, encode, label


def numpy_encoder(sae, device):
    import torch

    def encode(block: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return sae.encode(torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(device)).cpu().numpy()

    return encode


def load_representation(args, layer: int) -> Representation:
    act = Path(args.activations)
    kw = Path(args.keywords)
    if args.representation == "raw":
        return Representation("raw", np.load(act / f"layer_{layer:02d}.npy", mmap_mode="r"), np.load(act / "token_counts.npy"),
                              np.load(kw / f"occ_layer_{layer:02d}.npy", mmap_mode="r"))
    import torch

    from .sae import load_sae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, sae_meta = load_sae(args.sae, device)
    if args.representation == "speech_sae":
        encode = numpy_encoder(sae, device)
        occ = np.load(kw / f"occ_layer_{layer:02d}.npy", mmap_mode="r")
        occ_encoded = np.concatenate([encode(np.asarray(occ[s:e])) for s, e in _blocks(len(occ), 50_000)]) if len(occ) else occ
        occ_encoded[~np.isfinite(np.asarray(occ)).all(axis=1)] = np.nan
        return Representation("speech_sae", np.load(act / f"layer_{layer:02d}.npy", mmap_mode="r"),
                              np.load(act / "token_counts.npy"), occ_encoded, encode, sae_meta.get("name", ""))
    features = Path(args.sae_features)
    name = sae_name(args.sae)
    return Representation("token_sae", np.load(features / f"layer_{layer:02d}.npy", mmap_mode="r"),
                          np.load(features / "token_counts.npy"), np.load(kw / f"occ_sae_{name}.npy", mmap_mode="r"),
                          label=name)


def sae_name(path) -> str:
    path = Path(path)
    return path.parent.name if path.name == "sae.pt" else path.stem


# ------------------------------------------------------------------ CLI

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contextual EMI per speech for one model and representation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--activations", required=True, help="Speech activation collection (raw; also row order and counts).")
    parser.add_argument("--keywords", required=True, help="keyword_activations output for the same checkpoint.")
    parser.add_argument("--occurrences", required=True, help="keyword_occurrences output.")
    parser.add_argument("--representation", choices=["raw", "speech_sae", "token_sae"], default="raw")
    parser.add_argument("--sae", default=None, help="SAE checkpoint (speech_sae, token_sae).")
    parser.add_argument("--sae-features", default=None, help="Token-SAE speech feature collection (token_sae).")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layers (SAE representations: the SAE's layer).")
    parser.add_argument("--n-per-era", type=int, default=100, help="Occurrences per keyword and era used for the poles.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output-root", default="results/emi")
    return parser


def output_dir(args, act_meta: dict) -> Path:
    checkpoint = act_meta["checkpoint"]
    rep = args.representation if args.representation == "raw" else f"{args.representation}/{sae_name(args.sae)}"
    return Path(args.output_root) / rep / args.model / checkpoint["run"] / f"step{checkpoint['step']}"


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    act = Path(args.activations)
    act_meta = json.loads((act / "meta.json").read_text(encoding="utf-8"))
    occ_dir = Path(args.occurrences)
    occ = np.load(occ_dir / "occurrences.npz")
    stats = aligned_stats(act, occ_dir)
    keywords = json.loads((occ_dir / "keywords.json").read_text(encoding="utf-8"))
    n_keywords = len(keywords["words"])
    kw_pole = np.asarray(keywords["poles"], dtype=np.int8)
    if args.representation != "raw":
        from .sae import load_sae

        _, sae_meta = load_sae(args.sae, "cpu")
        layers = [int(sae_meta["layer"])]
    elif args.layers == "all":
        layers = list(range(int(act_meta["n_layers"])))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    out = output_dir(args, act_meta)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "emi_score.log", mode="a")

    congress = np.load(act / "congress.npy")
    ok = scorable(stats["word_count"], stats["top100_frac"])
    use = (occ["rank_balanced"] >= 0) & (occ["rank_balanced"] < args.n_per_era)
    banner(logger, f"CONTEXTUAL EMI: {args.model} ({args.representation})", {
        "activations": act, "keywords": args.keywords, "occurrences": f"{occ_dir} ({int(use.sum()):,} used for poles)",
        **({"sae": args.sae} if args.sae else {}), **({"sae features": args.sae_features} if args.sae_features else {}),
        "layers": ", ".join(map(str, layers)), "speeches": f"{len(congress):,} ({int(ok.sum()):,} scorable)",
        "output": out, **environment_info(),
    })

    series_rows, selection = [], []
    for layer in layers:
        t0 = time.time()
        rep = load_representation(args, layer)
        mu = token_weighted_mean(rep.source, rep.counts, encode=rep.encode)
        kw_vecs, missing = keyword_vectors(rep.occ, occ["keyword"], occ["era"], n_keywords, use)
        p_ev, p_int = poles(kw_vecs, kw_pole, mu)
        layer_score = layer_auc(rep.occ, occ["keyword"], occ["pole"], occ["era"], mu, use, n_keywords)
        ev, intu = cosines(rep.source, None, mu, p_ev, p_int, encode=rep.encode)
        emi, params = emi_from_cosines(ev, intu, stats["word_count"], ok)
        np.save(out / f"layer_{layer:02d}_ev.npy", ev)
        np.save(out / f"layer_{layer:02d}_int.npy", intu)
        np.save(out / f"layer_{layer:02d}_emi.npy", emi)
        np.save(out / f"mu_layer_{layer:02d}.npy", mu)
        np.savez(out / f"poles_layer_{layer:02d}.npz", p_ev=p_ev, p_int=p_int, keyword_vectors=kw_vecs)
        (out / f"params_layer_{layer:02d}.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        for row in congress_series(emi, congress, n_boot=args.n_boot):
            series_rows.append({"layer": layer, **row})
        pole_cos = float(p_ev @ p_int / (np.linalg.norm(p_ev) * np.linalg.norm(p_int)))
        selection.append({"layer": layer, "loo_keyword_auc": layer_score, "pole_cosine": pole_cos,
                          "missing_keywords": ";".join(keywords["words"][k] for k in missing)})
        logger.info(f"layer {layer:2d}: leave-one-keyword-out AUC {layer_score:.3f} | cos(P_ev, P_int) {pole_cos:+.3f} | "
                    f"{len(missing)} keywords without occurrences | {format_duration(time.time() - t0)}")

    with (out / "series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "congress", "n", "mean", "lower", "upper", "ci_method"])
        writer.writeheader()
        writer.writerows(series_rows)
    with (out / "layer_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "loo_keyword_auc", "pole_cosine", "missing_keywords"])
        writer.writeheader()
        writer.writerows(selection)
    valid = [s for s in selection if np.isfinite(s["loo_keyword_auc"])]
    primary = max(valid, key=lambda s: (s["loo_keyword_auc"], s["layer"]))["layer"] if valid else layers[-1]
    meta = {
        "model": args.model, "representation": args.representation, "sae": args.sae, "sae_features": args.sae_features,
        "activations": str(act), "keywords": args.keywords, "occurrences": str(occ_dir), "checkpoint": act_meta["checkpoint"],
        "layers": layers, "primary_layer": int(primary), "primary_rule": "highest leave-one-keyword-out AUC (ties: deeper)",
        "n_per_era": args.n_per_era, "n_speeches": int(len(congress)), "n_scored": int(ok.sum()),
        "elapsed_seconds": round(time.time() - started, 1), "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    banner(logger, f"CONTEXTUAL EMI COMPLETE: {args.model} ({args.representation})", {
        "primary layer": f"{primary} (AUC {next(s['loo_keyword_auc'] for s in selection if s['layer'] == primary):.3f})",
        "scored speeches": f"{int(ok.sum()):,} of {len(congress):,}",
        "output": out, "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "primary_layer": int(primary)}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run(args)
    except Exception:
        logger.exception("Contextual EMI scoring failed")
        raise


if __name__ == "__main__":
    main()
