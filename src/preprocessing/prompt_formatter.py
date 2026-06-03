from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from src.utils.common import get_logger

logger = get_logger(__name__)


def _build_classification_prompt(
    text: str,
    task_info: Dict[str, Any],
) -> str:
    id2label = task_info.get("id2label") or {}

    if id2label:
        classes = sorted({str(v) for v in id2label.values()})
        classes_str = ", ".join(classes)
    else:
        classes_str = "categories"

    return (
        f"Classify the following text into one of these categories: "
        f"{classes_str}.\n\n"
        f"Text:\n{text}"
    )


def _build_regression_prompt(text: str) -> str:
    return (
        "Predict the numeric score for the following text.\n\n"
        f"Text:\n{text}"
    )


def _get_input_text(
    batch: Dict[str, List],
    text_columns: List[str],
    row_idx: int,
) -> str:
    parts = []

    for col in text_columns:
        value = batch[col][row_idx]

        if value is None:
            continue

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            parts.append(value)

    return "\n".join(parts)


def _get_target(
    batch: Dict[str, List],
    label_column: str,
    row_idx: int,
):
    value = batch[label_column][row_idx]

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return value


def _format_batch(
    batch: Dict[str, List],
    format_type: str,
    text_columns: List[str],
    label_column: str,
    task_info: Dict[str, Any],
    custom_instruction: Optional[str] = None,
) -> Dict[str, List]:

    n_rows = len(next(iter(batch.values())))

    records = []

    for i in range(n_rows):

        input_text = _get_input_text(
            batch=batch,
            text_columns=text_columns,
            row_idx=i,
        )

        target = _get_target(
            batch=batch,
            label_column=label_column,
            row_idx=i,
        )

        target_text = str(target).strip()

        if format_type == "classification":

            records.append(
                {
                    "prompt": _build_classification_prompt(
                        input_text,
                        task_info,
                    ),
                    "response": target_text,
                }
            )

        elif format_type == "regression":

            records.append(
                {
                    "prompt": _build_regression_prompt(
                        input_text,
                    ),
                    "response": target,
                }
            )

        elif format_type == "chat":

            records.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": input_text,
                        },
                        {
                            "role": "assistant",
                            "content": target_text,
                        },
                    ]
                }
            )

        elif format_type == "instruction":

            instruction = custom_instruction

            if not instruction:
                task_name = task_info.get("task", "classification")

                if task_name == "classification":
                    instruction = (
                        "Classify the given text."
                    )

                elif task_name == "regression":
                    instruction = (
                        "Predict the numeric score for the given text."
                    )

                else:
                    instruction = (
                        "Perform the task on the given input."
                    )

            records.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": target_text,
                }
            )

        else:
            raise ValueError(
                f"Unsupported format_type: {format_type}"
            )

    result = {}

    for key in records[0].keys():
        result[key] = [row[key] for row in records]

    return result


def _format_split(
    dataset: Dataset,
    format_type: str,
    text_columns: List[str],
    label_column: str,
    task_info: Dict[str, Any],
    custom_instruction: Optional[str],
) -> Dataset:

    return dataset.map(
        lambda batch: _format_batch(
            batch=batch,
            format_type=format_type,
            text_columns=text_columns,
            label_column=label_column,
            task_info=task_info,
            custom_instruction=custom_instruction,
        ),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting ({format_type})",
    )


def format_dataset(
    dataset: Dataset | DatasetDict,
    format_type: str,
    task_info: Dict[str, Any],
    custom_instruction: Optional[str] = None,
):

    logger.info(
        f"Formatting dataset using '{format_type}' format"
    )

    text_columns = task_info.get("text_columns", [])
    label_column = task_info.get("label_column")

    if not text_columns:
        raise ValueError(
            "text_columns not found in task_info"
        )

    if not label_column:
        raise ValueError(
            "label_column not found in task_info"
        )

    if isinstance(dataset, DatasetDict):

        formatted_splits = {}

        for split_name, split_dataset in dataset.items():

            if len(split_dataset) == 0:
                formatted_splits[split_name] = split_dataset
                continue

            logger.info(
                f"Formatting split: {split_name}"
            )

            formatted_splits[split_name] = _format_split(
                dataset=split_dataset,
                format_type=format_type,
                text_columns=text_columns,
                label_column=label_column,
                task_info=task_info,
                custom_instruction=custom_instruction,
            )

        return DatasetDict(formatted_splits)

    if len(dataset) == 0:
        return dataset

    return _format_split(
        dataset=dataset,
        format_type=format_type,
        text_columns=text_columns,
        label_column=label_column,
        task_info=task_info,
        custom_instruction=custom_instruction,
    )