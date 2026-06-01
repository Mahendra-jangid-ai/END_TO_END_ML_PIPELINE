"""
Automatic NLP task detection from dataset schema and content.

Detects:
  - classification       (binary / multi-class)
  - regression
  - token_classification (NER, POS, chunking)
  - seq2seq              (summarization, translation, QA)
  - causal_lm            (instruction tuning, chatbot, text generation)
  - ner                  (explicit NER alias)
  - qa                   (question answering)
  - summarization        (article → summary)
  - translation          (src_lang → tgt_lang)
  - chatbot              (multi-turn dialog)
  - instruction_tuning   (instruction + response)
  - rag                  (retrieval-augmented generation)
"""
from __future__ import annotations

import re
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column name patterns
# ---------------------------------------------------------------------------

_TEXT_PATTERNS = re.compile(
    r"^(text|sentence|content|body|review|comment|document|passage|"
    r"question|input|source|premise|hypothesis|tweet|post|context)s?$",
    re.IGNORECASE,
)
_LABEL_PATTERNS = re.compile(
    r"^(label|labels|target|class|category|sentiment|tag|tags|"
    r"output|answer|response|intent|y)s?$",
    re.IGNORECASE,
)
_TABULAR_LABEL_HINTS = re.compile(
    r"(label|target|class|category|sentiment|intent|output|answer|response|"
    r"purchase|purchased|buy|bought|churn|clicked|converted|conversion|"
    r"fraud|spam|default|approved|accept|pass|survived|survive|won|positive|negative)",
    re.IGNORECASE,
)
_IDENTIFIER_PATTERNS = re.compile(r"(^id$|_id$|^id_|user id|uuid|index)", re.IGNORECASE)
_SEQ2SEQ_SRC = re.compile(
    r"^(source|src|input|question|article|document|premise|context|text)$",
    re.IGNORECASE,
)
_SEQ2SEQ_TGT = re.compile(
    r"^(target|tgt|output|answer|summary|translation|hypothesis|response|highlights)$",
    re.IGNORECASE,
)
_INSTRUCTION_PATTERNS = re.compile(r"^(instruction|prompt|system|user_input)s?$", re.IGNORECASE)
_TOKEN_LABEL_PATTERNS  = re.compile(r"^(ner_tags|pos_tags|chunk_tags|labels|tags)$", re.IGNORECASE)
_CONV_PATTERNS         = re.compile(r"^(conversation|messages|chat|dialog|dialogue)s?$", re.IGNORECASE)
_SUMMARY_SRC           = re.compile(r"^(article|document|body|text|content|passage)s?$", re.IGNORECASE)
_SUMMARY_TGT           = re.compile(r"^(summary|summaries|highlights|abstract)s?$", re.IGNORECASE)
_TRANS_SRC             = re.compile(r"^(translation\.|en_|src_|source_)", re.IGNORECASE)
_QA_Q                  = re.compile(r"^question$", re.IGNORECASE)
_QA_A                  = re.compile(r"^(answer|answers|response)$", re.IGNORECASE)
_QA_CONTEXT            = re.compile(r"^(context|passage|document)$", re.IGNORECASE)
_RAG_PATTERNS          = re.compile(r"^(retrieved|retrieval|chunks|documents|contexts|evidence)s?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_columns(dataset) -> list[str]:
    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        return dataset["train"].column_names
    return dataset.column_names


def _get_sample(dataset, n: int = 10) -> list[dict]:
    from datasets import DatasetDict
    split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    return [split[i] for i in range(min(n, len(split)))]


def _find_text_columns(columns):   return [c for c in columns if _TEXT_PATTERNS.match(c)]
def _find_label_columns(columns):  return [c for c in columns if _LABEL_PATTERNS.match(c)]


def _looks_like_identifier(column: str) -> bool:
    return bool(_IDENTIFIER_PATTERNS.search(column))


def _infer_tabular_label_column(columns: list[str], samples: list[dict]) -> str | None:
    """Infer a low-cardinality target column from tabular samples."""
    best_column: str | None = None
    best_score = float("-inf")

    for index, column in enumerate(columns):
        if _looks_like_identifier(column):
            continue

        values = [sample.get(column) for sample in samples if sample.get(column) is not None]
        if len(values) < 2:
            continue

        unique_values = {str(value).strip().lower() for value in values}
        unique_count = len(unique_values)
        if unique_count < 2 or unique_count > min(20, len(values)):
            continue

        if any(isinstance(value, str) and len(value.strip()) >= 40 for value in values):
            continue

        score = float(unique_count)
        if _LABEL_PATTERNS.match(column) or _TABULAR_LABEL_HINTS.search(column):
            score += 20.0
        if unique_count <= 4:
            score += 5.0
        if all(not isinstance(value, str) or len(value.strip()) <= 12 for value in values):
            score += 1.0

        # Prefer later columns when everything else is tied; tabular targets are often last.
        score += index / 1000.0

        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def _infer_tabular_text_column(columns: list[str], label_column: str | None) -> str | None:
    for column in columns:
        if column == label_column:
            continue
        if _looks_like_identifier(column):
            continue
        return column
    return None


def _find_seq2seq_columns(columns):
    src = next((c for c in columns if _SEQ2SEQ_SRC.match(c)), None)
    tgt = next((c for c in columns if _SEQ2SEQ_TGT.match(c)), None)
    return src, tgt


def _is_token_classification(columns, samples):
    label_cols = _find_label_columns(columns)
    if not label_cols:
        return False
    label_col = label_cols[0]
    return any(isinstance(s.get(label_col), list) for s in samples)


def _is_regression(label_col, samples):
    if not label_col:
        return False
    for s in samples:
        val = s.get(label_col)
        if isinstance(val, float):
            return True
        if isinstance(val, str):
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                pass
    return False


def _count_unique_labels(dataset, label_col):
    from datasets import DatasetDict
    split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    return len(set(split[label_col]))


# ---------------------------------------------------------------------------
# Sub-task detectors
# ---------------------------------------------------------------------------

def _detect_summarization(columns):
    """Detect summarization: long article → short summary."""
    src = next((c for c in columns if _SUMMARY_SRC.match(c)), None)
    tgt = next((c for c in columns if _SUMMARY_TGT.match(c)), None)
    return src, tgt


def _detect_translation(columns):
    """Detect translation: may have nested dict like {'en': ..., 'fr': ...}."""
    # Look for 'translation' column that holds a dict
    if "translation" in columns:
        return True, "translation"
    src_langs = [c for c in columns if _TRANS_SRC.match(c)]
    return (len(src_langs) >= 2), None


def _detect_qa(columns):
    """Detect QA: question + (context) + answer."""
    has_q = any(_QA_Q.match(c) for c in columns)
    has_a = any(_QA_A.match(c) for c in columns)
    has_ctx = any(_QA_CONTEXT.match(c) for c in columns)
    if has_q and has_a:
        q_col = next(c for c in columns if _QA_Q.match(c))
        a_col = next(c for c in columns if _QA_A.match(c))
        ctx_col = next((c for c in columns if _QA_CONTEXT.match(c)), None)
        return True, q_col, a_col, ctx_col
    return False, None, None, None


def _detect_rag(columns):
    """Detect RAG: retrieved documents + question + answer."""
    has_rag  = any(_RAG_PATTERNS.match(c) for c in columns)
    has_q    = any(_QA_Q.match(c) for c in columns)
    return has_rag and has_q


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class TaskDetector:
    """
    Inspects a HuggingFace dataset and returns a TaskInfo dict.

    Returns
    -------
    dict with keys:
        task, text_column, label_column, input_column, output_column,
        instruction_column, context_column, num_labels, label2id, id2label,
        problem_type, sub_task, detected_columns
    """

    def detect(self, dataset, cfg: DictConfig) -> dict[str, Any]:
        columns = _get_columns(dataset)
        samples = _get_sample(dataset)

        logger.info(f"Dataset columns: {columns}")

        # Config overrides
        text_col:  str | None = cfg.dataset.text_column
        label_col: str | None = cfg.dataset.label_column

        task_info: dict[str, Any] = {
            "task":                None,
            "sub_task":            None,         # finer label (ner, qa, summarization…)
            "text_column":         text_col,
            "label_column":        label_col,
            "input_column":        cfg.dataset.input_column,
            "output_column":       cfg.dataset.output_column,
            "instruction_column":  cfg.dataset.instruction_column,
            "context_column":      None,
            "num_labels":          None,
            "label2id":            None,
            "id2label":            None,
            "problem_type":        None,
            "detected_columns": {
                "text":        _find_text_columns(columns),
                "label":       _find_label_columns(columns),
                "all":         columns,
            },
        }

        # ── 0. User forced task via config ──────────────────────────────── #
        forced_task = getattr(cfg.dataset, "task", None) or getattr(getattr(cfg, "task", None), "name", None)
        if forced_task and forced_task != "auto":
            logger.info(f"Task forced by config: {forced_task}")
            task_info["task"] = forced_task
            task_info["sub_task"] = forced_task
            self._fill_columns(task_info, columns, samples, dataset)
            return task_info

        # ── 1. RAG ──────────────────────────────────────────────────────── #
        if _detect_rag(columns):
            logger.info("Detected: RAG (retrieved context + QA)")
            task_info["task"]    = "seq2seq"
            task_info["sub_task"] = "rag"
            qa_ok, q_col, a_col, ctx_col = _detect_qa(columns)
            task_info["input_column"]   = task_info["input_column"] or q_col
            task_info["output_column"]  = task_info["output_column"] or a_col
            task_info["context_column"] = ctx_col or next((c for c in columns if _RAG_PATTERNS.match(c)), None)
            return task_info

        # ── 2. Conversational / chatbot ──────────────────────────────────── #
        conv_cols = [c for c in columns if _CONV_PATTERNS.match(c)]
        if conv_cols:
            logger.info("Detected: chatbot (conversational format)")
            task_info["task"]    = "causal_lm"
            task_info["sub_task"] = "chatbot"
            task_info["text_column"] = conv_cols[0]
            return task_info

        # ── 3. Instruction tuning ────────────────────────────────────────── #
        inst_cols = [c for c in columns if _INSTRUCTION_PATTERNS.match(c)]
        if inst_cols:
            logger.info("Detected: instruction_tuning")
            task_info["task"]    = "causal_lm"
            task_info["sub_task"] = "instruction_tuning"
            task_info["instruction_column"] = task_info["instruction_column"] or inst_cols[0]
            out_col = task_info["output_column"] or next(
                (c for c in columns if _SEQ2SEQ_TGT.match(c)), None
            )
            task_info["output_column"] = out_col
            return task_info

        # ── 4. QA ────────────────────────────────────────────────────────── #
        qa_ok, q_col, a_col, ctx_col = _detect_qa(columns)
        if qa_ok:
            logger.info(f"Detected: QA  (q='{q_col}', a='{a_col}', ctx='{ctx_col}')")
            task_info["task"]    = "seq2seq"
            task_info["sub_task"] = "qa"
            task_info["input_column"]   = task_info["input_column"]  or q_col
            task_info["output_column"]  = task_info["output_column"] or a_col
            task_info["context_column"] = ctx_col
            return task_info

        # ── 5. Summarization ─────────────────────────────────────────────── #
        s_src, s_tgt = _detect_summarization(columns)
        if s_src and s_tgt:
            logger.info(f"Detected: summarization  (src='{s_src}', tgt='{s_tgt}')")
            task_info["task"]    = "seq2seq"
            task_info["sub_task"] = "summarization"
            task_info["input_column"]  = task_info["input_column"]  or s_src
            task_info["output_column"] = task_info["output_column"] or s_tgt
            return task_info

        # ── 6. Translation ───────────────────────────────────────────────── #
        is_trans, trans_col = _detect_translation(columns)
        if is_trans:
            logger.info("Detected: translation")
            task_info["task"]    = "seq2seq"
            task_info["sub_task"] = "translation"
            if trans_col:
                task_info["input_column"]  = task_info["input_column"]  or trans_col
                task_info["output_column"] = task_info["output_column"] or trans_col
            return task_info

        # ── 7. Generic seq2seq (two text columns: src + tgt) ─────────────── #
        src_col, tgt_col = _find_seq2seq_columns(columns)
        if src_col and tgt_col:
            logger.info(f"Detected: seq2seq  (src='{src_col}', tgt='{tgt_col}')")
            task_info["task"]   = "seq2seq"
            task_info["sub_task"] = "seq2seq"
            task_info["input_column"]  = task_info["input_column"]  or src_col
            task_info["output_column"] = task_info["output_column"] or tgt_col
            return task_info

        # ── 8. Token classification / NER ────────────────────────────────── #
        if _is_token_classification(columns, samples):
            label_cols = _find_label_columns(columns)
            text_cols  = _find_text_columns(columns) or ["tokens"]
            # Distinguish NER from POS by column name
            sub = "ner" if any("ner" in c.lower() for c in columns) else "token_classification"
            logger.info(f"Detected: {sub} (token-level labels)")
            task_info["task"]    = "token_classification"
            task_info["sub_task"] = sub
            task_info["text_column"]  = task_info["text_column"]  or (text_cols[0]  if text_cols  else columns[0])
            task_info["label_column"] = task_info["label_column"] or (label_cols[0] if label_cols else None)
            self._enrich_classification_info(dataset, task_info)
            return task_info

        # ── 9. Find text + label columns ─────────────────────────────────── #
        text_cols  = _find_text_columns(columns)
        label_cols = _find_label_columns(columns)

        if not task_info["text_column"]:
            task_info["text_column"] = text_cols[0] if text_cols else _infer_tabular_text_column(columns, task_info["label_column"])
        if not task_info["label_column"]:
            task_info["label_column"] = label_cols[0] if label_cols else _infer_tabular_label_column(columns, samples)

        # ── 10. Regression ───────────────────────────────────────────────── #
        if _is_regression(task_info["label_column"], samples):
            logger.info("Detected: regression")
            task_info["task"]    = "regression"
            task_info["sub_task"] = "regression"
            task_info["num_labels"]    = 1
            task_info["problem_type"]  = "regression"
            return task_info

        # ── 11. Classification ───────────────────────────────────────────── #
        if task_info["label_column"]:
            self._enrich_classification_info(dataset, task_info)
            n     = task_info["num_labels"]
            ptype = "single_label_classification" if n and n > 1 else "regression"
            task_info["task"]    = "classification"
            task_info["sub_task"] = "binary_classification" if n == 2 else "multi_class_classification"
            task_info["problem_type"] = ptype
            logger.info(
                f"Detected: classification  (num_labels={n}, "
                f"text_col='{task_info['text_column']}', "
                f"label_col='{task_info['label_column']}')"
            )
            return task_info

        # ── 12. Fallback: causal LM ──────────────────────────────────────── #
        logger.info("No labels / structure found → defaulting to causal_lm")
        task_info["task"]    = "causal_lm"
        task_info["sub_task"] = "text_generation"
        task_info["text_column"] = task_info["text_column"] or columns[0]
        return task_info

    # ------------------------------------------------------------------

    def _fill_columns(self, task_info, columns, samples, dataset):
        """Populate column fields when task is forced via config."""
        text_cols  = _find_text_columns(columns)
        label_cols = _find_label_columns(columns)
        if not task_info["text_column"]:
            task_info["text_column"] = text_cols[0] if text_cols else (columns[0] if columns else None)
        if not task_info["label_column"]:
            task_info["label_column"] = label_cols[0] if label_cols else None
        if task_info["task"] in ("classification", "token_classification"):
            self._enrich_classification_info(dataset, task_info)

    def _enrich_classification_info(self, dataset, task_info):
        """Populate num_labels, label2id, id2label."""
        from datasets import DatasetDict
        label_col = task_info["label_column"]
        if not label_col:
            return
        split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
        features = split.features
        if label_col in features:
            feat = features[label_col]
            if hasattr(feat, "names"):
                names = feat.names
                task_info["num_labels"] = len(names)
                task_info["label2id"]   = {n: i for i, n in enumerate(names)}
                task_info["id2label"]   = {i: n for i, n in enumerate(names)}
                return
        labels = sorted(set(str(x) for x in split[label_col]))
        task_info["num_labels"] = len(labels)
        task_info["label2id"]   = {l: i for i, l in enumerate(labels)}
        task_info["id2label"]   = {i: l for i, l in enumerate(labels)}
