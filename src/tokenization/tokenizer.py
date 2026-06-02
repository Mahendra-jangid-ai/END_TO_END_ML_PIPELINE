#!/usr/bin/env python3
"""
Production-grade dataset tokenizer

Supports:
- .csv
- .json / .jsonl
- .txt
- .parquet
- .xlsx

Features:
- Auto-detects text columns
- Combines multiple text fields
- Tokenizes with any Hugging Face tokenizer
- Saves tokenized dataset and summary
- Robust logging and error handling

Usage:
python tokenize_dataset.py \
    --input_file data.csv \
    --model_name bert-base-uncased \
    --output_dir आउटपुट/

Optional:
--text_columns col1,col2
--max_length 512
--truncation true
--padding false
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None


# -----------------------------
# Logging
# -----------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
SUPPORTED_TEXT_EXTS = {".csv", ".json", ".jsonl", ".txt", ".parquet", ".xlsx", ".xls"}


def detect_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_TEXT_EXTS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_TEXT_EXTS))}"
        )
    return ext


def safe_str(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_text_columns(df: pd.DataFrame, threshold: float = 0.55) -> List[str]:
    """
    Infer text-like columns:
    - object/string columns
    - mixed columns that mostly contain strings
    """
    text_cols = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            sample = series.dropna().astype(str).head(200)
            if len(sample) == 0:
                continue
            avg_len = sample.map(len).mean()
            text_ratio = sample.map(lambda x: isinstance(x, str)).mean()
            if text_ratio >= threshold and avg_len >= 1:
                text_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # numeric columns are generally not primary text columns
            continue
    return text_cols


def combine_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Create one canonical text field from selected columns.
    """
    if not text_columns:
        raise ValueError("No text columns found to tokenize.")

    def row_to_text(row) -> str:
        parts = []
        for col in text_columns:
            val = safe_str(row.get(col, ""))
            if val:
                parts.append(val)
        return " [SEP] ".join(parts).strip()

    out = df.copy()
    out["__text__"] = out.apply(row_to_text, axis=1)
    out = out[out["__text__"].astype(str).str.len() > 0].reset_index(drop=True)
    return out


def load_input_file(file_path: str) -> pd.DataFrame:
    ext = detect_file_type(file_path)

    logger.info("Loading file: %s", file_path)

    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext in {".json", ".jsonl"}:
        return pd.read_json(file_path, lines=(ext == ".jsonl"))
    if ext == ".parquet":
        return pd.read_parquet(file_path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": lines})

    raise ValueError(f"Unsupported file type: {ext}")


def build_tokenizer(model_name: str):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Some tokenizers have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token

    return tokenizer


def tokenize_batch(
    examples,
    tokenizer,
    max_length: int,
    truncation: bool,
    padding: bool,
):
    return tokenizer(
        examples["__text__"],
        max_length=max_length,
        truncation=truncation,
        padding="max_length" if padding else False,
    )


def save_summary(
    output_dir: str,
    file_path: str,
    model_name: str,
    text_columns: List[str],
    total_rows: int,
    kept_rows: int,
    max_length: int,
    truncation: bool,
    padding: bool,
) -> None:
    summary = {
        "input_file": file_path,
        "model_name": model_name,
        "text_columns": text_columns,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
        "max_length": max_length,
        "truncation": truncation,
        "padding": padding,
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(output_dir) / "tokenization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Summary saved to %s", summary_path)


def maybe_split_dataset(df: pd.DataFrame, test_size: float, seed: int):
    if test_size <= 0 or test_size >= 1:
        return df, None

    if train_test_split is None:
        raise RuntimeError("scikit-learn is required for dataset split.")

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def save_dataset(ds: Dataset, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    ds.save_to_disk(output_path)
    logger.info("Tokenized dataset saved to %s", output_path)


def infer_text_columns_from_dataset(dataset) -> List[str]:
    candidates = [
        "text",
        "input",
        "prompt",
        "sentence",
        "question",
        "message",
        "instruction",
        "source",
    ]
    available = [c for c in dataset.column_names]
    for candidate in candidates:
        if candidate in available:
            return [candidate]
    fuzzy = [c for c in available if any(c.lower().endswith(suffix) for suffix in ("text", "input", "prompt", "question", "message"))]
    if fuzzy:
        return [fuzzy[0]]
    raise ValueError(
        "Could not infer a text column from dataset splits. "
        "Pass text_columns explicitly to the tokenizer."
    )


def prepare_text_batch(batch: dict, text_columns: List[str]) -> dict:
    if not text_columns:
        raise ValueError("text_columns must be provided for Dataset tokenization")
    row_count = len(batch[text_columns[0]])
    text_values = []

    for i in range(row_count):
        parts: list[str] = []
        for col in text_columns:
            values = batch.get(col)
            if values is None:
                continue
            value = safe_str(values[i])
            if value:
                parts.append(value)
        text_values.append(" [SEP] ".join(parts).strip())

    return {"__text__": text_values}


def tokenize_dataset_dict(
    dataset,
    model_name: str,
    text_columns: Optional[List[str]] = None,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool = True,
    output_dir: Optional[str] = None,
):
    from datasets import DatasetDict

    if text_columns is not None and len(text_columns) == 0:
        text_columns = None

    if isinstance(dataset, DatasetDict):
        ds_dict = dataset
    else:
        raise TypeError("tokenize_dataset_dict expects a DatasetDict")

    if text_columns is None:
        text_columns = infer_text_columns_from_dataset(ds_dict["train"])

    tokenizer = build_tokenizer(model_name)

    tokenized_splits = {}
    for split_name, split_ds in ds_dict.items():
        if "__text__" not in split_ds.column_names:
            split_ds = split_ds.map(
                lambda batch: prepare_text_batch(batch, text_columns),
                batched=True,
                desc=f"Preparing text for {split_name}",
            )

        tokenized = split_ds.map(
            lambda batch: tokenize_batch(
                batch,
                tokenizer=tokenizer,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
            ),
            batched=True,
            desc=f"Tokenizing {split_name}",
        )

        if "__text__" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["__text__"])

        tokenized_splits[split_name] = tokenized

    tokenized_dataset = DatasetDict(tokenized_splits)

    if output_dir is not None:
        save_dataset(tokenized_dataset, output_dir)

    return tokenized_dataset


# -----------------------------
# Main pipeline
# -----------------------------
@dataclass
class Config:
    input_file: str
    model_name: str
    output_dir: str
    text_columns: Optional[List[str]]
    max_length: int
    truncation: bool
    padding: bool
    test_size: float
    seed: int
    infer_text: bool


def tokenize_dataset(config: Config) -> None:
    df = load_input_file(config.input_file)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    total_rows = len(df)
    logger.info("Rows loaded: %d", total_rows)
    logger.info("Columns: %s", list(df.columns))

    if config.text_columns:
        missing = [c for c in config.text_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Text columns not found in file: {missing}")
        text_columns = config.text_columns
    else:
        text_columns = infer_text_columns(df)
        if not text_columns:
            if "text" in df.columns:
                text_columns = ["text"]
            else:
                raise ValueError(
                    "Could not infer text columns. "
                    "Pass --text_columns manually."
                )

    logger.info("Using text columns: %s", text_columns)

    df = combine_text_columns(df, text_columns)
    kept_rows = len(df)

    if kept_rows == 0:
        raise ValueError("No valid text rows found after cleaning.")

    logger.info("Valid text rows: %d", kept_rows)

    tokenizer = build_tokenizer(config.model_name)

    # Train/val split if requested
    train_df, val_df = maybe_split_dataset(df, config.test_size, config.seed)

    def tokenize_df(dataframe: pd.DataFrame) -> Dataset:
        ds = dataframe_to_dataset(dataframe)
        tokenized = ds.map(
            lambda x: tokenize_batch(
                x,
                tokenizer=tokenizer,
                max_length=config.max_length,
                truncation=config.truncation,
                padding=config.padding,
            ),
            batched=True,
            desc="Tokenizing",
        )
        # Keep original text if useful; remove if you want pure tensor-like output
        return tokenized

    os.makedirs(config.output_dir, exist_ok=True)

    if val_df is not None:
        train_tok = tokenize_df(train_df)
        val_tok = tokenize_df(val_df)

        dd = DatasetDict({"train": train_tok, "validation": val_tok})
        dd.save_to_disk(config.output_dir)
        logger.info("DatasetDict saved to %s", config.output_dir)
    else:
        tokenized = tokenize_df(df)
        save_dataset(tokenized, config.output_dir)

    save_summary(
        output_dir=config.output_dir,
        file_path=config.input_file,
        model_name=config.model_name,
        text_columns=text_columns,
        total_rows=total_rows,
        kept_rows=kept_rows,
        max_length=config.max_length,
        truncation=config.truncation,
        padding=config.padding,
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Production-grade dataset tokenizer")
    parser.add_argument("--input_file", required=True, help="Path to dataset file")
    parser.add_argument("--model_name", required=True, help="Hugging Face model/tokenizer name or local path")
    parser.add_argument("--output_dir", required=True, help="Directory to save tokenized output")
    parser.add_argument("--text_columns", default=None, help="Comma-separated text columns")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--truncation", type=str, default="true")
    parser.add_argument("--padding", type=str, default="false")
    parser.add_argument("--test_size", type=float, default=0.0, help="Optional validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_text", type=str, default="true")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    text_columns = None
    if args.text_columns and args.text_columns.strip():
        text_columns = [c.strip() for c in args.text_columns.split(",") if c.strip()]

    return Config(
        input_file=args.input_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        text_columns=text_columns,
        max_length=args.max_length,
        truncation=args.truncation.lower() in {"true", "1", "yes", "y"},
        padding=args.padding.lower() in {"true", "1", "yes", "y"},
        test_size=args.test_size,
        seed=args.seed,
        infer_text=args.infer_text.lower() in {"true", "1", "yes", "y"},
    )


def main() -> int:
    try:
        config = parse_args()
        tokenize_dataset(config)
        logger.info("Done.")
        return 0
    except Exception as e:
        logger.exception("Tokenization failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())