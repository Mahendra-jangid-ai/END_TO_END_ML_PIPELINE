"""
Universal dataset loader.
Supports: CSV, JSON, JSONL, Excel, Parquet, HuggingFace Hub, local folders, streaming.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

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
            df = pd.read_csv(name, sep=sep)
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

    # Create missing splits
    if "train" not in ds:
        # Use first available split as train
        first_split = list(ds.keys())[0]
        logger.warning(f"No 'train' split found; using '{first_split}' as train")
        ds = DatasetDict({"train": ds[first_split]})

    if "test" not in ds and "validation" not in ds:
        logger.info("Creating train/val/test splits automatically")
        split = ds["train"].train_test_split(test_size=test_size + val_size, seed=seed)
        val_ratio = val_size / (test_size + val_size)
        val_test = split["test"].train_test_split(test_size=1 - val_ratio, seed=seed)
        ds = DatasetDict({
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
    elif "validation" not in ds and "test" in ds:
        logger.info("Creating validation split from train")
        split = ds["train"].train_test_split(test_size=val_size, seed=seed)
        ds = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        })
    elif "test" not in ds and "validation" in ds:
        ds = DatasetDict({
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["validation"],  # reuse val as test
        })

    # Apply max_samples
    if max_samples is not None:
        for split in ds:
            if len(ds[split]) > max_samples:
                ds[split] = ds[split].select(range(max_samples))

    for split, d in ds.items():
        logger.info(f"  {split}: {len(d):,} samples")

    return ds
