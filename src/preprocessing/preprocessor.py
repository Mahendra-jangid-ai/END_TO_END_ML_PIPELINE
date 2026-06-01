"""
NLP Dataset Preprocessor
Supports:
- Hugging Face Dataset / DatasetDict
- Unicode normalization
- HTML unescape
- Null normalization
- Text column detection
- Label column detection
- Empty row removal
- Empty text removal
- Duplicate text removal
- Label normalization
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from omegaconf import DictConfig

from src.utils.common import get_logger

logger = get_logger(__name__)

NULL_VALUES = {
    "", "na", "n/a", "null", "none", "nan", "missing"
}


def _normalize_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    if value.lower() in NULL_VALUES:
        return None

    return value


def _sample_values(dataset, column: str, limit: int = 200):
    return [dataset[i].get(column) for i in range(min(limit, len(dataset)))]


def _infer_text_columns(columns, dataset):
    text_columns = set()

    for column in columns:
        values = [v for v in _sample_values(dataset, column) if v is not None]

        if not values:
            continue

        str_values = [v for v in values if isinstance(v, str)]

        if not str_values:
            continue

        str_ratio = len(str_values) / len(values)
        unique_ratio = len(set(str_values)) / max(len(str_values), 1)

        # Smart check: if 80% or more are strings
        if str_ratio >= 0.8:
            # High uniqueness OR long average string length (e.g. > 15 chars)
            avg_length = sum(len(v) for v in str_values) / len(str_values)
            if unique_ratio >= 0.05 or avg_length > 15:
                text_columns.add(column)

    return text_columns


def _infer_label_columns(columns, dataset, max_classes: int = 200):
    label_columns = set()

    for column in columns:
        values = [v for v in _sample_values(dataset, column) if v is not None]

        if not values:
            continue

        unique_count = len(
            set(str(v).strip().lower() for v in values)
        )

        if 2 <= unique_count <= min(max_classes, len(values)):
            label_columns.add(column)

    return label_columns


def _row_has_signal(example):
    for value in example.values():
        if value is None:
            continue

        if isinstance(value, str):
            if value.strip():
                return True

        elif isinstance(value, list):
            if len(value):
                return True

        else:
            return True

    return False


def preprocess_dataset(dataset, cfg: DictConfig | None = None):
    from datasets import Dataset, DatasetDict

    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: preprocess_dataset(ds, cfg)
            for split, ds in dataset.items()
        })

    if not isinstance(dataset, Dataset):
        return dataset

    if len(dataset) == 0:
        return dataset

    columns = list(dataset.column_names)

    text_columns = set()
    label_columns = set()

    if cfg is not None and hasattr(cfg, "dataset"):
        text_column = getattr(cfg.dataset, "text_column", None)
        label_column = getattr(cfg.dataset, "label_column", None)

        if text_column:
            text_columns.add(text_column)

        if label_column:
            label_columns.add(label_column)

    text_columns.update(
        _infer_text_columns(columns, dataset)
    )

    max_label_classes = 200
    if cfg is not None and hasattr(cfg, "dataset"):
        max_label_classes = getattr(cfg.dataset, "max_label_classes", 200)

    label_columns.update(
        _infer_label_columns(columns, dataset, max_classes=max_label_classes)
    )

    logger.info(
        f"Preprocessing NLP dataset "
        f"(text={sorted(text_columns)}, "
        f"labels={sorted(label_columns)})"
    )

    def _clean_batch(batch):

        cleaned = {}

        for column, values in batch.items():
            output = []

            for value in values:

                if isinstance(value, list):
                    cleaned_list = [
                        _normalize_string(v) if isinstance(v, str) else v
                        for v in value
                    ]
                    # Note: We do not drop None/empty items to preserve sequence length/alignment (e.g. NER)
                    output.append(cleaned_list)
                    continue

                value = _normalize_string(value)

                if (
                    column in label_columns
                    and isinstance(value, str)
                ):
                    value = value.strip().lower()

                output.append(value)

            cleaned[column] = output

        return cleaned

    dataset = dataset.map(
        _clean_batch,
        batched=True,
        desc="Cleaning NLP dataset"
    )

    dataset = dataset.filter(
        _row_has_signal,
        desc="Removing empty rows"
    )

    if text_columns:

        def _valid_text(example):

            for col in text_columns:

                text = example.get(col)

                if (
                    isinstance(text, str)
                    and text.strip()
                ):
                    return True

            return False

        dataset = dataset.filter(
            _valid_text,
            desc="Removing empty text rows"
        )

    try:
        import pandas as pd

        before = len(dataset)

        # Memory-efficient duplicate removal: only load text columns into pandas
        # to identify unique row indices, keeping the rest of the dataset memory-mapped.
        if text_columns:
            subset = list(text_columns)
            # Remove all other columns to minimize memory usage
            cols_to_remove = [c for c in dataset.column_names if c not in subset]
            small_ds = dataset.remove_columns(cols_to_remove)
            df_small = small_ds.to_pandas()
            df_unique = df_small.drop_duplicates(subset=subset)
            unique_indices = df_unique.index.tolist()
        else:
            df_all = dataset.to_pandas()
            df_unique = df_all.drop_duplicates()
            unique_indices = df_unique.index.tolist()

        removed = before - len(unique_indices)

        if removed:
            logger.info(
                f"Removed {removed} duplicate rows"
            )
            dataset = dataset.select(unique_indices)

    except Exception as e:
        logger.warning(
            f"Duplicate removal skipped: {e}"
        )

    logger.info(
        f"Preprocessing completed. Rows={len(dataset)}"
    )

    return dataset
