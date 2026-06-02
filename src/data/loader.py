"""
Universal dataset loader.
Supports: CSV, JSON, JSONL, Excel, Parquet, HuggingFace Hub, local folders, streaming.
"""
from __future__ import annotations

import os
import csv
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.preprocessing.preprocessor import preprocess_dataset
from src.utils.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
}


def _detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _EXT_MAP.get(suffix, "unknown")


def _is_local(name: str) -> bool:
    return os.path.exists(name)


def _is_hf_dataset(name: str) -> bool:
    """Heuristic: no file extension → likely a HuggingFace dataset name."""
    return "/" in name or "." not in name


def _looks_like_data_token(value: Any) -> bool:
    """Heuristic check to identify values that look like row data, not header names."""
    s = str(value).strip()
    if not s:
        return False
    if s.isdigit():
        return True
    if any(ch in s for ch in [",", " ", "@", "://"]):
        return True
    return False


def _first_row_looks_like_header(path: str, sep: str) -> bool:
    """Fallback heuristic when csv.Sniffer mis-detects headers."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            first_row = next(reader, None)
    except Exception:
        return False

    if not first_row:
        return False

    normalized = [str(col).strip().lower() for col in first_row]
    if not all(normalized):
        return False

    header_token = re.compile(r"^[a-z_][a-z0-9_]*$")
    known_header_words = {
        "text", "label", "labels", "target", "source", "input", "output",
        "instruction", "response", "question", "answer", "tokens", "score",
        "summary", "translation",
    }

    all_headerish = all(
        header_token.match(col) is not None and not _looks_like_data_token(col)
        for col in normalized
    )
    known_hits = sum(col in known_header_words for col in normalized)
    unique_cols = len(set(normalized)) == len(normalized)

    return all_headerish and (known_hits >= 1 or unique_cols)


def _read_csv_with_header_fallback(path: str, sep: str):
    """Read CSV/TSV while handling headerless files safely."""
    import pandas as pd

    has_header = True
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        has_header = csv.Sniffer().has_header(sample)
    except Exception:
        # Keep default behavior if sniffer fails.
        has_header = True

    if not has_header and _first_row_looks_like_header(path, sep):
        logger.info("Header fallback heuristic detected a valid header row")
        has_header = True

    if has_header:
        df = pd.read_csv(path, sep=sep)
        # Fallback: if all column names look like row data, reload as headerless.
        if len(df.columns) > 0 and all(_looks_like_data_token(c) for c in df.columns):
            logger.warning("CSV header looks invalid; reloading as headerless with generic column names")
            df = pd.read_csv(path, sep=sep, header=None)
            df.columns = [f"column_{i}" for i in range(len(df.columns))]
    else:
        logger.info("Detected headerless CSV/TSV; assigning generic column names")
        df = pd.read_csv(path, sep=sep, header=None)
        df.columns = [f"column_{i}" for i in range(len(df.columns))]

    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(cfg: DictConfig):
    """
    Load a dataset from any supported source.

    Returns a HuggingFace DatasetDict with at minimum a 'train' split.
    """
    from datasets import load_dataset as hf_load, DatasetDict, Dataset
    import pandas as pd

    name: str = cfg.dataset.name
    cache_dir: str = cfg.dataset.cache_dir
    max_samples: int | None = cfg.dataset.max_samples
    seed: int = cfg.dataset.seed
    streaming: bool = cfg.dataset.streaming

    logger.info(f"Loading dataset: {name}")

    # ------------------------------------------------------------------ #
    # 1. Local file                                                        #
    # ------------------------------------------------------------------ #
    if _is_local(name):
        fmt = _detect_format(name)
        logger.info(f"Detected local file format: {fmt}")

        if fmt == "csv":
            sep = "\t" if name.endswith(".tsv") else ","
            df = _read_csv_with_header_fallback(name, sep=sep)
        elif fmt == "json":
            df = pd.read_json(name, lines=name.endswith(".jsonl"))
        elif fmt == "parquet":
            df = pd.read_parquet(name)
        elif fmt == "excel":
            df = pd.read_excel(name)
        elif fmt == "unknown" and Path(name).is_dir():
            # Local folder → try HuggingFace load_dataset
            ds = hf_load(name, cache_dir=cache_dir, streaming=streaming)
            return _finalize(ds, cfg)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

        ds = Dataset.from_pandas(df)
        dataset = DatasetDict({"train": ds})
        return _finalize(dataset, cfg)

    # ------------------------------------------------------------------ #
    # 2. HuggingFace Hub dataset                                          #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading from HuggingFace Hub: {name}")
    try:
        ds = hf_load(name, cache_dir=cache_dir, streaming=streaming)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' from HuggingFace Hub: {e}")

    return _finalize(ds, cfg)


# ---------------------------------------------------------------------------
# Post-processing: splits + sampling
# ---------------------------------------------------------------------------

def _safe_train_test_split(dataset, test_size, seed, description: str | None = None):
    if test_size is None or test_size <= 0:
        return None

    if len(dataset) < 2:
        logger.warning(
            "Skipping %s split: only %d row(s) available, not enough data for test_size=%s",
            description or "dataset",
            len(dataset),
            test_size,
        )
        return None

    try:
        return dataset.train_test_split(test_size=test_size, seed=seed)
    except ValueError as exc:
        logger.warning(
            "Unable to split %s with n=%d and test_size=%s: %s",
            description or "dataset",
            len(dataset),
            test_size,
            exc,
        )
        return None


def _finalize(ds, cfg: DictConfig):
    """Ensure train/val/test splits exist; apply max_samples."""
    from datasets import DatasetDict, Dataset

    max_samples: int | None = cfg.dataset.max_samples
    test_size: float = cfg.dataset.test_size
    val_size: float = cfg.dataset.val_size
    seed: int = cfg.dataset.seed

    # Normalize to DatasetDict
    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    cleaning_report = None
    quality_report = None
    row_level_audit = None
    issue_summary = None
    warnings = None
    recommendations = None

    res = preprocess_dataset(ds, cfg)
    if isinstance(res, tuple):
        ds = res[0]
        cleaning_report = res[1]
        quality_report = res[2]
        row_level_audit = res[3]
        issue_summary = res[4]
        warnings = res[5]
        recommendations = res[6]
    else:
        ds = res

    # Create missing splits
    if "train" not in ds:
        # Use first available split as train
        first_split = list(ds.keys())[0]
        logger.warning(f"No 'train' split found; using '{first_split}' as train")
        ds = DatasetDict({"train": ds[first_split]})

    # Apply max_samples to train split BEFORE creating val/test splits
    if max_samples is not None and "train" in ds:
        if len(ds["train"]) > max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples (from {len(ds['train'])})")
            ds["train"] = ds["train"].select(range(max_samples))

    if "test" not in ds and "validation" not in ds:
        logger.info("Creating train/val/test splits automatically")
        split = _safe_train_test_split(ds["train"], test_size=test_size + val_size, seed=seed, description="train")
        if split is not None:
            val_ratio = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.0
            if val_ratio == 0.0:
                ds = DatasetDict({
                    "train": split["train"],
                    "test": split["test"],
                })
            else:
                val_test = _safe_train_test_split(
                    split["test"],
                    test_size=1 - val_ratio,
                    seed=seed,
                    description="validation/test",
                )
                if val_test is not None:
                    ds = DatasetDict({
                        "train": split["train"],
                        "validation": val_test["train"],
                        "test": val_test["test"],
                    })
                else:
                    if len(split["test"]) > 0:
                        logger.warning(
                            "Unable to split held-out set into validation and test. Using held-out set as validation only."
                        )
                        ds = DatasetDict({
                            "train": split["train"],
                            "validation": split["test"],
                        })
                    else:
                        ds = DatasetDict({"train": split["train"]})
        else:
            logger.warning("Skipping automatic train/val/test split because dataset is too small.")
    elif "validation" not in ds and "test" in ds:
        logger.info("Creating validation split from train")
        split = _safe_train_test_split(ds["train"], test_size=val_size, seed=seed, description="train/validation")
        if split is not None:
            ds = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
                "test": ds["test"],
            })
        else:
            logger.warning("Skipping automatic validation split because train split is too small.")
    elif "test" not in ds and "validation" in ds:
        logger.info("Creating test split from validation to avoid val/test leakage")
        val_test = _safe_train_test_split(ds["validation"], test_size=0.5, seed=seed, description="validation/test")
        if val_test is not None:
            ds = DatasetDict({
                "train": ds["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            })
        else:
            logger.warning("Skipping automatic test split because validation split is too small.")


    for split, d in ds.items():
        logger.info(f"  {split}: {len(d):,} samples")


    if cleaning_report is not None:
        ds.cleaning_report = cleaning_report
        ds.quality_report = quality_report
        ds.row_level_audit = row_level_audit
        ds.issue_summary = issue_summary
        ds.warnings = warnings
        ds.recommendations = recommendations

    return ds
