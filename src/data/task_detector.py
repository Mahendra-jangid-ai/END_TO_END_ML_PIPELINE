"""
Automatic NLP task detection from dataset schema and content.

Detects:
  - classification (binary / multi-class)
  - regression
  - token_classification (NER, POS)
  - seq2seq (summarization, translation, QA)
  - causal_lm (instruction tuning, chatbot, text generation)
"""
from __future__ import annotations

import re
from numbers import Real
from typing import Any

from omegaconf import DictConfig

from src.utils.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known column name patterns
# ---------------------------------------------------------------------------

_TEXT_PATTERNS = re.compile(
    r"^(text|sentence|content|body|review|comment|document|passage|"
    r"question|input|source|premise|hypothesis|tweet|post)s?$",
    re.IGNORECASE,
)
_LABEL_PATTERNS = re.compile(
    r"^(label|labels|target|class|category|sentiment|tag|tags|"
    r"output|answer|response|intent|y)s?$",
    re.IGNORECASE,
)
_SEQ2SEQ_SRC = re.compile(r"^(source|src|input|question|article|document|premise)$", re.IGNORECASE)
_SEQ2SEQ_TGT = re.compile(r"^(target|tgt|output|answer|summary|translation|hypothesis|response)$", re.IGNORECASE)
_INSTRUCTION_PATTERNS = re.compile(r"^(instruction|prompt|system|user_input)s?$", re.IGNORECASE)
_TOKEN_LABEL_PATTERNS = re.compile(r"^(ner_tags|pos_tags|chunk_tags|labels|tags)$", re.IGNORECASE)
_CONV_PATTERNS = re.compile(r"^(conversation|messages|chat|dialog|dialogue)s?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def _get_columns(dataset) -> list[str]:
    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        return dataset["train"].column_names
    return dataset.column_names


def _get_sample(dataset, n: int = 50) -> list[dict]:
    from datasets import DatasetDict
    split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    return [split[i] for i in range(min(n, len(split)))]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _find_text_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if _TEXT_PATTERNS.match(c)]


def _find_label_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if _LABEL_PATTERNS.match(c)]


def _find_seq2seq_columns(columns: list[str]) -> tuple[str | None, str | None]:
    src = next((c for c in columns if _SEQ2SEQ_SRC.match(c)), None)
    tgt = next((c for c in columns if _SEQ2SEQ_TGT.match(c)), None)
    return src, tgt


def _infer_text_and_label_columns(columns: list[str], samples: list[dict]) -> tuple[str | None, str | None]:
    """Infer likely text/label columns using value patterns when names are unhelpful."""
    if not samples:
        return None, None

    stats: dict[str, dict[str, Any]] = {}
    for c in columns:
        values = [s.get(c) for s in samples if s.get(c) is not None]
        if not values:
            continue

        str_values = [v for v in values if isinstance(v, str)]
        str_ratio = len(str_values) / len(values)
        avg_len = sum(len(v.strip()) for v in str_values) / len(str_values) if str_values else 0.0
        uniq = len(set(str(v).strip().lower() for v in values))

        stats[c] = {
            "str_ratio": str_ratio,
            "avg_len": avg_len,
            "uniq": uniq,
            "total": len(values),
        }

    if not stats:
        return None, None

    # Text column tends to be string-heavy and longer content.
    text_candidates = [
        c for c, st in stats.items()
        if st["str_ratio"] >= 0.8 and st["avg_len"] >= 20
    ]
    text_col = max(text_candidates, key=lambda c: stats[c]["avg_len"]) if text_candidates else None

    # Label column tends to be low-cardinality and relatively short values.
    label_candidates = [
        c for c, st in stats.items()
        if st["uniq"] >= 2 and st["uniq"] <= min(20, st["total"]) and st["avg_len"] <= 30
    ]

    # Avoid selecting the same column for both roles.
    if text_col in label_candidates:
        label_candidates = [c for c in label_candidates if c != text_col]

    label_col = min(label_candidates, key=lambda c: stats[c]["uniq"]) if label_candidates else None
    return text_col, label_col


def _is_token_classification(columns: list[str], samples: list[dict]) -> bool:
    """Check if labels are lists (token-level)."""
    label_cols = _find_label_columns(columns)
    if not label_cols:
        return False
    label_col = label_cols[0]
    for s in samples:
        if isinstance(s.get(label_col), list):
            return True
    return False


def _is_regression(label_col: str | None, samples: list[dict]) -> bool:
    if label_col is None:
        return False
    values = [s.get(label_col) for s in samples if s.get(label_col) is not None]
    if not values:
        return False

    parsed: list[float] = []
    for val in values:
        if isinstance(val, bool):
            return False
        if isinstance(val, Real):
            parsed.append(float(val))
            continue
        if isinstance(val, str) and _is_float(val):
            parsed.append(float(val))
            continue
        return False

    if not parsed:
        return False

    # Any clear decimal signal strongly indicates regression.
    for x in parsed:
        if abs(x - round(x)) > 1e-9:
            return True

    n = len(parsed)
    unique_count = len(set(parsed))
    unique_ratio = unique_count / max(n, 1)

    # Integer labels with low cardinality are usually classification.
    return unique_count > max(20, int(0.35 * n)) or unique_ratio > 0.7


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _count_unique_labels(dataset, label_col: str) -> int:
    from datasets import DatasetDict
    split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    return len(set(split[label_col]))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class TaskDetector:
    """
    Inspects a HuggingFace dataset and returns a TaskInfo dict:
      {
        task: str,
        text_column: str,
        label_column: str | None,
        input_column: str | None,
        output_column: str | None,
        num_labels: int | None,
        label2id: dict | None,
        id2label: dict | None,
        problem_type: str | None,
      }
    """

    def detect(self, dataset, cfg: DictConfig) -> dict[str, Any]:
        columns = _get_columns(dataset)
        samples = _get_sample(dataset)

        logger.info(f"Dataset columns: {columns}")

        # Override from config if user specified
        text_col: str | None = cfg.dataset.text_column
        label_col: str | None = cfg.dataset.label_column

        task_info: dict[str, Any] = {
            "task": None,
            "text_column": text_col,
            "label_column": label_col,
            "input_column": cfg.dataset.input_column,
            "output_column": cfg.dataset.output_column,
            "instruction_column": cfg.dataset.instruction_column,
            "num_labels": None,
            "label2id": None,
            "id2label": None,
            "problem_type": None,
        }

        # ------------------------------------------------- #
        # 1. Conversational / instruction format             #
        # ------------------------------------------------- #
        conv_cols = [c for c in columns if _CONV_PATTERNS.match(c)]
        inst_cols = [c for c in columns if _INSTRUCTION_PATTERNS.match(c)]
        if conv_cols or inst_cols:
            logger.info("Detected: causal_lm (conversational/instruction format)")
            task_info["task"] = "causal_lm"
            task_info["text_column"] = (inst_cols or conv_cols)[0]
            return task_info

        # ------------------------------------------------- #
        # 2. Seq2seq (two text columns: src + tgt)          #
        # ------------------------------------------------- #
        src_col, tgt_col = _find_seq2seq_columns(columns)
        if src_col and tgt_col:
            logger.info(f"Detected: seq2seq  (src='{src_col}', tgt='{tgt_col}')")
            task_info["task"] = "seq2seq"
            task_info["input_column"] = task_info["input_column"] or src_col
            task_info["output_column"] = task_info["output_column"] or tgt_col
            return task_info

        # ------------------------------------------------- #
        # 3. Token classification                           #
        # ------------------------------------------------- #
        if _is_token_classification(columns, samples):
            label_cols = _find_label_columns(columns)
            text_cols = _find_text_columns(columns) or ["tokens"]
            logger.info("Detected: token_classification (NER/POS)")
            task_info["task"] = "token_classification"
            task_info["text_column"] = task_info["text_column"] or (text_cols[0] if text_cols else columns[0])
            task_info["label_column"] = task_info["label_column"] or (label_cols[0] if label_cols else None)
            self._enrich_classification_info(dataset, task_info)
            task_info["problem_type"] = "token_classification"
            return task_info

        # ------------------------------------------------- #
        # 4. Find text + label columns                      #
        # ------------------------------------------------- #
        text_cols = _find_text_columns(columns)
        label_cols = _find_label_columns(columns)

        # Fallback for anonymous schemas like column_0, column_1, ...
        if not text_cols or not label_cols:
            inferred_text, inferred_label = _infer_text_and_label_columns(columns, samples)
            if inferred_text and inferred_text not in text_cols:
                text_cols.append(inferred_text)
            if inferred_label and inferred_label not in label_cols:
                label_cols.append(inferred_label)

        if not text_col:
            task_info["text_column"] = text_cols[0] if text_cols else columns[0]
        if not label_col:
            task_info["label_column"] = label_cols[0] if label_cols else None

        # ------------------------------------------------- #
        # 5. Regression                                     #
        # ------------------------------------------------- #
        if _is_regression(task_info["label_column"], samples):
            logger.info("Detected: regression")
            task_info["task"] = "regression"
            task_info["num_labels"] = 1
            task_info["problem_type"] = "regression"
            return task_info

        # ------------------------------------------------- #
        # 6. Classification                                 #
        # ------------------------------------------------- #
        if task_info["label_column"]:
            task_info["task"] = "classification"
            self._enrich_classification_info(dataset, task_info)
            n = task_info["num_labels"]
            ptype = "single_label_classification"
            task_info["problem_type"] = ptype
            logger.info(
                f"Detected: classification  (num_labels={n}, "
                f"text_col='{task_info['text_column']}', "
                f"label_col='{task_info['label_column']}')"
            )
            return task_info

        # ------------------------------------------------- #
        # 7. Fallback: causal LM (pure text)               #
        # ------------------------------------------------- #
        logger.info("No labels found → defaulting to causal_lm")
        task_info["task"] = "causal_lm"
        task_info["text_column"] = task_info["text_column"] or columns[0]
        return task_info

    # ------------------------------------------------------------------

    def _enrich_classification_info(self, dataset, task_info: dict) -> None:
        """Populate num_labels, label2id, id2label."""
        from datasets import DatasetDict, ClassLabel

        label_col = task_info["label_column"]
        if label_col is None:
            return

        split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset

        # Check if it's a ClassLabel feature
        features = split.features
        if label_col in features:
            feat = features[label_col]
            if hasattr(feat, "names"):  # ClassLabel
                names = feat.names
                task_info["num_labels"] = len(names)
                task_info["label2id"] = {n: i for i, n in enumerate(names)}
                task_info["id2label"] = {i: n for i, n in enumerate(names)}
                return

            # Sequence(ClassLabel) for token classification datasets
            if hasattr(feat, "feature") and isinstance(feat.feature, ClassLabel):
                names = feat.feature.names
                task_info["num_labels"] = len(names)
                task_info["label2id"] = {n: i for i, n in enumerate(names)}
                task_info["id2label"] = {i: n for i, n in enumerate(names)}
                return

        # Token-level labels as list[int]/list[str]
        sample_value = split[0][label_col] if len(split) > 0 else None
        if isinstance(sample_value, list):
            token_values: set[str] = set()
            for seq in split[label_col]:
                if isinstance(seq, list):
                    for item in seq:
                        token_values.add(str(item))

            labels = sorted(token_values)
            task_info["num_labels"] = len(labels)
            task_info["label2id"] = {l: i for i, l in enumerate(labels)}
            task_info["id2label"] = {i: l for i, l in enumerate(labels)}
            return

        # Infer from data
        labels = list(set(str(x) for x in split[label_col]))
        labels.sort()
        task_info["num_labels"] = len(labels)
        task_info["label2id"] = {l: i for i, l in enumerate(labels)}
        task_info["id2label"] = {i: l for i, l in enumerate(labels)}
