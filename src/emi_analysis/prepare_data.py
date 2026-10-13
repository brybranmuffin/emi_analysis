"""Build the training file and the fixed held-out sets from the raw congressional speech corpus.

Order follows the handoff: hold out speeches first, then deduplicate only the training side.

    python -m emi_analysis.prepare_data --input <corpus.csv.gzip> [--out-root data]

Writes, under ``--out-root``:
    train.jsonl                    training speeches (speech_id, congress, text), deduplicated
    heldout/speech_ids.txt         fixed held-out speech IDs (reused on later runs unless --regenerate-holdout)
    heldout/congressional.jsonl    held-out speeches
    heldout/general.jsonl          general-domain sample for the forgetting check
    processed/dedup_report.csv     duplicates removed per Congress
    processed/prep_summary.json    every count printed in the log
    processed/prepare_data.log     copy of the log
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_pipeline import (
    DEFAULT_GENERAL_HOLDOUT_COUNT,
    DEFAULT_HOLDOUT_SPEECH_COUNT,
    congress_from_year,
    dedup_key,
    stratified_holdout_ids,
)
from .runinfo import banner, environment_info, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

WIKITEXT_MIN_WORDS = 30


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train.jsonl and the fixed held-out sets from the raw speech corpus.")
    parser.add_argument("--input", required=True, help="Speech corpus (.csv, .csv.gz/.csv.gzip, .tsv, .jsonl or .parquet).")
    parser.add_argument("--out-root", default="data", help="Output root. Use a separate root (e.g. data/val10k) for subset runs.")
    parser.add_argument("--id-column", default="speech_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--congress-column", default="congress", help="Used if present; otherwise Congress is derived from --year-column.")
    parser.add_argument("--year-column", default="year")
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_SPEECH_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="Use a random subset of N speeches (e.g. 10000 for pipeline validation).")
    parser.add_argument("--regenerate-holdout", action="store_true", help="Overwrite an existing heldout/speech_ids.txt.")
    parser.add_argument(
        "--general-source",
        default="wikitext",
        help="'wikitext' (WikiText-103 test split via Hugging Face), a .txt/.jsonl file path, or 'none'.",
    )
    parser.add_argument("--general-count", type=int, default=DEFAULT_GENERAL_HOLDOUT_COUNT)
    parser.add_argument("--regenerate-general", action="store_true", help="Overwrite an existing heldout/general.jsonl.")
    parser.add_argument("--cache-dir", default=".cache/huggingface", help="Hugging Face datasets cache for --general-source wikitext.")
    return parser


def read_corpus(path: Path, id_column: str, text_column: str, congress_column: str, year_column: str) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz", ".json")):
        df = pd.read_json(path, lines=True, dtype=False)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        sep = "\t" if ".tsv" in name else ","
        compression = "gzip" if name.endswith(".gzip") else "infer"
        header = list(pd.read_csv(path, sep=sep, compression=compression, nrows=0).columns)
        _check_columns(header, id_column, text_column, congress_column, year_column)
        wanted = [c for c in (id_column, text_column, congress_column, year_column) if c in header]
        return pd.read_csv(path, sep=sep, compression=compression, usecols=wanted, dtype={id_column: str})
    _check_columns(list(df.columns), id_column, text_column, congress_column, year_column)
    return df


def _check_columns(columns: list[str], id_column: str, text_column: str, congress_column: str, year_column: str) -> None:
    missing = [c for c in (id_column, text_column) if c not in columns]
    if congress_column not in columns and year_column not in columns:
        missing.append(f"{congress_column} or {year_column}")
    if missing:
        raise ValueError(
            f"Corpus is missing required column(s) {missing}; found {columns}. "
            "Point --id-column/--text-column/--congress-column/--year-column at the right names."
        )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write rows via a temp file so an interrupted run never leaves a partial file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def speech_rows(df: pd.DataFrame) -> Iterable[dict]:
    for speech_id, congress, text in zip(df["speech_id"], df["congress"], df["text"]):
        yield {"speech_id": speech_id, "congress": int(congress), "text": text}


def load_general_texts(source: str, cache_dir: str) -> tuple[str, list[str]]:
    if source == "wikitext":
        import datasets

        datasets.disable_progress_bars()
        paragraphs: list[str] = []
        for split in ("test", "validation"):
            dataset = datasets.load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=str(Path(cache_dir) / "datasets")
            )
            paragraphs.extend(line.strip() for line in dataset["text"])
        texts = [
            p.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
            for p in paragraphs
            if p and not p.startswith("=") and len(p.split()) >= WIKITEXT_MIN_WORDS
        ]
        return "wikitext-103-test+validation", texts

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"--general-source {source!r} is neither 'wikitext', 'none', nor an existing file.")
    if path.suffix == ".jsonl":
        texts = [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return path.name, texts


def dedup_training_speeches(train_df: pd.DataFrame, heldout_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return masks over ``train_df``: near-duplicates within training, and copies of held-out speeches."""
    train_keys = train_df["text"].map(dedup_key)
    heldout_keys = set(heldout_df["text"].map(dedup_key))
    matches_heldout = train_keys.isin(heldout_keys)
    within_train = train_keys.duplicated(keep="first") & ~matches_heldout
    return within_train, matches_heldout


def dedup_report(train_df: pd.DataFrame, within_train: pd.Series, matches_heldout: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "congress": train_df["congress"].to_numpy(),
            "within_train": within_train.to_numpy(),
            "matches_heldout": matches_heldout.to_numpy(),
        }
    )
    report = frame.groupby("congress").agg(
        speeches_before=("within_train", "size"),
        removed_within_train=("within_train", "sum"),
        removed_matching_heldout=("matches_heldout", "sum"),
    )
    report["removed_total"] = report["removed_within_train"] + report["removed_matching_heldout"]
    report["removed_pct"] = (100 * report["removed_total"] / report["speeches_before"]).round(2)
    report.insert(0, "start_year", 1789 + 2 * (report.index - 1))
    return report.reset_index()


def prepare(args: argparse.Namespace) -> dict:
    started = time.time()
    out_root = Path(args.out_root)
    heldout_dir = out_root / "heldout"
    processed_dir = out_root / "processed"
    input_path = Path(args.input)

    banner(logger, "EMI data preparation", {"input": input_path, "out_root": out_root, **environment_info()})

    logger.info(f"Reading corpus {input_path} ({input_path.stat().st_size / 1e9:.2f} GB on disk)")
    df = read_corpus(input_path, args.id_column, args.text_column, args.congress_column, args.year_column)
    input_rows = len(df)
    logger.info(f"Read {input_rows:,} rows with columns {list(df.columns)} in {format_duration(time.time() - started)}")

    df = df.rename(columns={args.id_column: "speech_id", args.text_column: "text"})
    df["speech_id"] = df["speech_id"].astype(str)
    empty = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
    dropped_empty = int(empty.sum())
    df = df.loc[~empty].copy()
    df["text"] = df["text"].astype(str)
    duplicate_ids = df["speech_id"].duplicated(keep="first")
    dropped_duplicate_ids = int(duplicate_ids.sum())
    df = df.loc[~duplicate_ids]
    logger.info(f"Dropped {dropped_empty:,} rows with empty text and {dropped_duplicate_ids:,} rows with a repeated speech_id")

    if args.congress_column in df.columns:
        df["congress"] = df[args.congress_column].astype(int)
    else:
        df["congress"] = df[args.year_column].map(congress_from_year)

    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed).sort_index()
        logger.info(f"Sampled {len(df):,} speeches (--sample {args.sample}, seed {args.seed})")
    df["n_words"] = df["text"].str.count(r"\S+")
    logger.info(
        f"Usable speeches: {len(df):,} across Congresses {df['congress'].min()}-{df['congress'].max()} "
        f"({df['n_words'].sum():,} whitespace words)"
    )

    # Step 1: fixed held-out split, before any other processing.
    ids_path = heldout_dir / "speech_ids.txt"
    reused = ids_path.exists() and not args.regenerate_holdout
    if reused:
        heldout_ids = ids_path.read_text(encoding="utf-8").split()
        logger.info(f"Reusing fixed held-out IDs from {ids_path} ({len(heldout_ids):,} IDs); pass --regenerate-holdout to rebuild")
    else:
        heldout_ids = stratified_holdout_ids(df["speech_id"].tolist(), df["congress"].tolist(), args.holdout_count, seed=args.seed)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(heldout_ids) + "\n", encoding="utf-8")
        logger.info(f"Wrote {len(heldout_ids):,} stratified held-out IDs to {ids_path}")
    is_heldout = df["speech_id"].isin(set(heldout_ids))
    missing_heldout = len(set(heldout_ids)) - int(is_heldout.sum())
    if missing_heldout:
        logger.warning(f"{missing_heldout:,} held-out IDs from {ids_path} are not in this corpus")
    heldout_df = df.loc[is_heldout]
    train_df = df.loc[~is_heldout]
    logger.info(
        f"Held out {len(heldout_df):,} speeches spanning {heldout_df['congress'].nunique()} Congresses; "
        f"{len(train_df):,} remain for training"
    )

    # Step 2: deduplicate the training side only; also drop training copies of held-out speeches.
    dedup_started = time.time()
    within_train, matches_heldout = dedup_training_speeches(train_df, heldout_df)
    removed = within_train | matches_heldout
    report = dedup_report(train_df, within_train, matches_heldout)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(processed_dir / "dedup_report.csv", index=False)
    logger.info(
        f"Dedup removed {int(removed.sum()):,} training speeches ({int(within_train.sum()):,} near-duplicates within training, "
        f"{int(matches_heldout.sum()):,} copies of held-out speeches) in {format_duration(time.time() - dedup_started)}"
    )
    era = report.assign(era=(report["start_year"] // 20) * 20).groupby("era")[["speeches_before", "removed_total"]].sum()
    for era_start, row in era.iterrows():
        pct = 100 * row["removed_total"] / max(1, row["speeches_before"])
        logger.info(f"  dedup {era_start}-{era_start + 19}: removed {int(row['removed_total']):,} of {int(row['speeches_before']):,} ({pct:.2f}%)")
    final_train = train_df.loc[~removed]

    train_path = out_root / "train.jsonl"
    heldout_path = heldout_dir / "congressional.jsonl"
    write_jsonl(train_path, speech_rows(final_train))
    write_jsonl(heldout_path, speech_rows(heldout_df))
    logger.info(f"Wrote {len(final_train):,} training speeches to {train_path} and {len(heldout_df):,} held-out speeches to {heldout_path}")

    general_path = heldout_dir / "general.jsonl"
    if general_path.exists() and not args.regenerate_general:
        general_count = sum(1 for line in general_path.open(encoding="utf-8") if line.strip())
        general_source = "existing file"
        logger.info(f"Reusing general-domain sample {general_path} ({general_count:,} sequences); pass --regenerate-general to rebuild")
    elif args.general_source == "none":
        general_count, general_source = 0, "none"
        logger.warning("No general-domain sample written (--general-source none); training will refuse to start without one")
    else:
        general_source, texts = load_general_texts(args.general_source, args.cache_dir)
        rng = np.random.default_rng(args.seed)
        picks = sorted(rng.choice(len(texts), size=min(args.general_count, len(texts)), replace=False))
        if len(picks) < args.general_count:
            logger.warning(f"General-domain source {general_source} has only {len(texts):,} usable texts; wanted {args.general_count:,}")
        general_count = write_jsonl(
            general_path, ({"id": f"{general_source}-{i}", "source": general_source, "text": texts[i]} for i in picks)
        )
        logger.info(f"Wrote {general_count:,} general-domain sequences from {general_source} to {general_path}")

    summary = {
        "input_path": str(input_path),
        "input_rows": input_rows,
        "dropped_empty_text": dropped_empty,
        "dropped_duplicate_ids": dropped_duplicate_ids,
        "sample": args.sample,
        "usable_speeches": len(df),
        "heldout_speeches": len(heldout_df),
        "heldout_ids_reused": reused,
        "heldout_ids_missing_from_corpus": missing_heldout,
        "train_speeches_before_dedup": len(train_df),
        "dedup_removed_within_train": int(within_train.sum()),
        "dedup_removed_matching_heldout": int(matches_heldout.sum()),
        "dedup_removed_total": int(removed.sum()),
        "train_speeches": len(final_train),
        "words_usable": int(df["n_words"].sum()),
        "words_heldout": int(heldout_df["n_words"].sum()),
        "words_removed_by_dedup": int(train_df.loc[removed, "n_words"].sum()),
        "words_train": int(final_train["n_words"].sum()),
        "general_source": general_source,
        "general_sequences": general_count,
        "seed": args.seed,
        "outputs": {
            "train": str(train_path),
            "heldout_ids": str(ids_path),
            "heldout_congressional": str(heldout_path),
            "general": str(general_path),
            "dedup_report": str(processed_dir / "dedup_report.csv"),
        },
        **environment_info(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (processed_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    banner(
        logger,
        "DATA PREPARATION SUMMARY (whitespace words; model token counts are logged by train.py)",
        {
            "input rows": f"{input_rows:,}",
            "dropped (empty text / repeated id)": f"{dropped_empty:,} / {dropped_duplicate_ids:,}",
            "usable speeches": f"{len(df):,} ({summary['words_usable']:,} words)",
            "held-out speeches": f"{len(heldout_df):,} ({summary['words_heldout']:,} words)",
            "removed by dedup": f"{summary['dedup_removed_total']:,} ({summary['words_removed_by_dedup']:,} words)",
            "training speeches": f"{len(final_train):,} ({summary['words_train']:,} words)",
            "words check": "usable = held-out + dedup + train: "
            + ("OK" if summary["words_usable"] == summary["words_heldout"] + summary["words_removed_by_dedup"] + summary["words_train"] else "MISMATCH"),
            "general-domain sequences": f"{general_count:,} ({general_source})",
            "summary file": processed_dir / "prep_summary.json",
            "elapsed": format_duration(time.time() - started),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    setup_logging(Path(args.out_root) / "processed" / "prepare_data.log", mode="a")
    try:
        return prepare(args)
    except Exception:
        logger.exception("Data preparation failed")
        raise


if __name__ == "__main__":
    main()
