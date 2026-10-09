from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from .config import DATASET_URL, DATA_DIR, RAW_DATA_DIR


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file_if_missing(url: str, destination: str | Path) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Found existing file: {dest}")
        return dest
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    return dest


def extract_archive_if_needed(archive_path: str | Path, extract_dir: str | Path) -> Path:
    archive = Path(archive_path)
    target = Path(extract_dir)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        print(f"Archive already extracted to {target}")
        return target

    print(f"Extracting {archive} into {target}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")

    return target


def find_dataset_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    candidates = list(root.rglob("*.jsonl")) + list(root.rglob("*.csv")) + list(root.rglob("*.parquet")) + list(root.rglob("*.tsv"))
    return sorted(set(candidates))


def ensure_dataset_download(dataset_url: str = DATASET_URL, raw_dir: str | Path = RAW_DATA_DIR, data_dir: str | Path = DATA_DIR) -> list[Path]:
    raw_dir = ensure_directory(raw_dir)
    data_dir = ensure_directory(data_dir)
    dataset_files = find_dataset_files(data_dir)
    if dataset_files:
        print(f"Dataset is already present: {dataset_files[:3]}")
        return dataset_files

    archive_name = "dataset_archive.zip"
    archive_path = raw_dir / archive_name
    download_file_if_missing(dataset_url, archive_path)
    extracted = extract_archive_if_needed(archive_path, raw_dir / "extracted")
    dataset_files = find_dataset_files(extracted)
    if not dataset_files:
        raise FileNotFoundError(f"No speech or JSON files were found in the extracted archive: {extracted}")

    print(f"Discovered dataset files: {[str(p) for p in dataset_files[:5]]}")
    return dataset_files


def ensure_local_model(model_name: str, cache_dir: str | Path = ".cache/huggingface") -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring local Hugging Face model cache for {model_name} under {cache_path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path))
    return cache_path / "models--" / model_name.replace('/', '--')


def main() -> None:
    dataset_files = ensure_dataset_download()
    print(f"Downloaded dataset assets: {[str(p) for p in dataset_files[:3]]}")


if __name__ == "__main__":
    main()
