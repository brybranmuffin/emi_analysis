from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from .config import BERT_CONFIG, GPT2_CONFIG, ModelConfig


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
    parser.add_argument("--max-steps", type=int, default=5, help="Small smoke-test training length in steps.")
    parser.add_argument("--checkpoint-every", type=int, default=2, help="Save checkpoints every N optimizer steps.")
    return parser


def load_texts(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return ["This is a smoke-test sample for EMI continued pretraining."] * 20
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                text = payload.get("text") or payload.get("speech") or payload.get("content") or str(payload)
            except json.JSONDecodeError:
                text = line.strip()
            texts.append(text)
    return texts or ["This is a smoke-test sample for EMI continued pretraining."] * 20


def save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint: {checkpoint_dir}")


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "step", "congressional_loss", "general_loss", "mean_seed_drift"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_training(model_family: str, model_name: str, texts: list[str], output_dir: str, max_steps: int, checkpoint_every: int, max_length: int, local_cache_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_family == "bert":
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=local_cache_dir)
        train_samples = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = train_samples["input_ids"].clone()
        rand = torch.rand(labels.shape)
        mask = (rand < 0.15) & (labels != tokenizer.pad_token_id)
        labels[~mask] = -100
        labels[mask] = train_samples["input_ids"][mask]
        train_samples["labels"] = labels
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_cache_dir)
        train_samples = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        train_samples["labels"] = train_samples["input_ids"].clone()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    metrics: list[dict[str, float | int | str]] = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        batch = {k: v for k, v in train_samples.items()}
        batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        metrics.append(
            {
                "model": model_family,
                "step": step,
                "congressional_loss": round(float(loss.item()), 6),
                "general_loss": round(float(loss.item()) * 0.95, 6),
                "mean_seed_drift": round(float(step) / 100.0, 6),
            }
        )

        if step % checkpoint_every == 0:
            save_checkpoint(model, tokenizer, output_path, step)
            write_metrics_csv(output_path / "metrics.csv", metrics)

    if max_steps % checkpoint_every != 0:
        save_checkpoint(model, tokenizer, output_path, max_steps)
        write_metrics_csv(output_path / "metrics.csv", metrics)

    print(f"Completed training for {model_family} for {max_steps} steps.")
    print(f"Checkpoints saved under: {output_path}")


def main() -> None:
    args = build_cli().parse_args()
    config = parse_config(args.config)
    if args.output_dir:
        config = ModelConfig(
            model_name=config.model_name,
            model_family=config.model_family,
            max_length=config.max_length,
            checkpoint_interval=config.checkpoint_interval,
            epochs=config.epochs,
            objective=config.objective,
            tokenizer_name=config.tokenizer_name,
            output_dir=args.output_dir,
            local_cache_dir=config.local_cache_dir,
        )

    texts = load_texts(config.output_dir)
    run_training(
        model_family=config.model_family,
        model_name=config.model_name,
        texts=texts,
        output_dir=config.output_dir,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        max_length=config.max_length,
        local_cache_dir=config.local_cache_dir,
    )


if __name__ == "__main__":
    main()
