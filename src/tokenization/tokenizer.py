# from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)





def safe_str(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_text_columns(df: pd.DataFrame, threshold: float = 0.55) -> List[str]:
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
            continue
    return text_cols








def build_tokenizer(model_name: str):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token

    return tokenizer


def tokenize_batch(
    examples,
    tokenizer,
    max_length: int,
    truncation: bool,
    padding: bool | str,
):
    if isinstance(padding, bool):
        pad_arg = "max_length" if padding else False
    else:
        pad_arg = padding

    return tokenizer(
        examples["__text__"],
        max_length=max_length,
        truncation=truncation,
        padding=pad_arg,
    )





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
        
    try:
        df_sample = dataset.select(range(min(len(dataset), 50))).to_pandas()
        inferred = infer_text_columns(df_sample)
        if inferred:
            return inferred
    except Exception:
        pass
        
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
    padding: bool | str = True,
    output_dir: Optional[str] = None,
):
    from datasets import Dataset, DatasetDict

    if text_columns is not None and len(text_columns) == 0:
        text_columns = None

    is_single_dataset = False
    if isinstance(dataset, DatasetDict):
        ds_dict = dataset
    elif isinstance(dataset, Dataset):
        ds_dict = DatasetDict({"train": dataset})
        is_single_dataset = True
    elif isinstance(dataset, pd.DataFrame):
        ds_dict = DatasetDict({"train": Dataset.from_pandas(dataset, preserve_index=False)})
        is_single_dataset = True
    elif isinstance(dataset, (list, dict)):
        try:
            df = pd.DataFrame(dataset)
            ds_dict = DatasetDict({"train": Dataset.from_pandas(df, preserve_index=False)})
            is_single_dataset = True
        except Exception as e:
            raise TypeError(f"Could not convert input {type(dataset)} to Dataset: {e}")
    else:
        raise TypeError("tokenize_dataset_dict expects a Dataset, DatasetDict, DataFrame, list, or dict")

    split_key = "train" if "train" in ds_dict else next(iter(ds_dict.keys()))
    if text_columns is None:
        text_columns = infer_text_columns_from_dataset(ds_dict[split_key])

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

    if is_single_dataset:
        return tokenized_dataset["train"]
    return tokenized_dataset


