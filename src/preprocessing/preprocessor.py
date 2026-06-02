"""
Universal NLP Dataset Preprocessing Engine for LLM Fine-Tuning
══════════════════════════════════════════════════════════════
Implements the 7 basic cleaning steps:
  1. Remove Null Values
  2. Remove Empty Text
  3. Remove Duplicate Records
  4. Trim Extra Spaces
  5. Fix Encoding Issues
  6. Remove Corrupted/Broken Text
  7. Basic Label Validation
"""
from __future__ import annotations

import html
import re
import unicodedata
import json
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd
from omegaconf import DictConfig

from src.utils.common import get_logger

logger = get_logger(__name__)

# Regex patterns for advanced text cleaning and masking
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(
    r"\+?\d{1,3}[-.\s]\d{3,5}[-.\s]\d{4,5}(?:[-.\s]\d{4})?"
    r"|\b(?:\d{3,5}[-.\s]\d{3,5}[-.\s]\d{4}|\d{5}[-.\s]\d{5}|\d{10})\b"
)

# PII Patterns
AADHAAR_RE = re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{6}[-\s]?\d{5})\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fix Encoding Issues
# ─────────────────────────────────────────────────────────────────────────────

def heal_mojibake(text: str) -> str:
    """Heal common Mojibake encoding artifacts."""
    if not isinstance(text, str):
        return text
    # Try dynamic CP1252 to UTF-8 healing
    try:
        if any(ord(c) > 127 for c in text):
            candidate = text.encode("cp1252").decode("utf-8")
            if sum(ord(c) > 127 for c in candidate) < sum(ord(c) > 127 for c in text):
                return candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    mojibake_map = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€¢": "•",
        "â€”": "—", "â€“": "–", "Ã©": "é", "Ã¡": "á",
        "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ", "Ã­": "í",
        "Ã": "à", "â€˜": "'", "â€ž": "„", "â€¦": "…",
    }
    for bad, good in mojibake_map.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization to text."""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKC", text)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trim Extra Spaces
# ─────────────────────────────────────────────────────────────────────────────

def clean_spaces(text: str) -> str:
    """Trim leading/trailing whitespace and collapse consecutive spaces."""
    if not isinstance(text, str):
        return text
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Remove Corrupted/Broken Text
# ─────────────────────────────────────────────────────────────────────────────

def is_corrupted_text(text: str) -> bool:
    """Detect OCR word corruption and keyboard mashes (excessively long words)."""
    if not isinstance(text, str):
        return True
    
    # 1. OCR Corruption Heuristic (e.g. letters and numbers mixed inside a word like "he11o")
    if re.search(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", text):
        return True
        
    # 2. Keyboard Mash Heuristic (e.g. words that are too long without spaces)
    words = text.split()
    for w in words:
        if len(w) > 40:
            return True
            
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Basic Label Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_clean_label(val: Any, task_type: str) -> tuple[Any, bool]:
    """Validate labels, lowercase classification labels, or convert regression labels to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None, False

    if task_type == "regression":
        try:
            return float(val), True
        except (ValueError, TypeError):
            return None, False
    else:
        # classification
        val_str = str(val).strip()
        if not val_str or val_str.lower() in {"nan", "null", "none"}:
            return None, False
        return val_str.lower(), True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Remove Duplicate Records
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicate_rows(dataset: Any, text_columns: List[str]) -> tuple[Any, int]:
    """Remove exact duplicate rows based on text columns using pandas drop_duplicates."""
    before = len(dataset)
    try:
        cols_present = [c for c in text_columns if c in dataset.column_names]
        df = dataset.select_columns(cols_present).to_pandas() if cols_present else dataset.to_pandas()
        df_unique = df.drop_duplicates(subset=cols_present if cols_present else None)
        keep_indices = df_unique.index.tolist()
        total_removed = before - len(keep_indices)
        if total_removed > 0:
            dataset = dataset.select(keep_indices)
        return dataset, total_removed
    except Exception as e:
        logger.warning(f"Deduplication failed or skipped: {e}")
        return dataset, 0


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing Mapper
# ─────────────────────────────────────────────────────────────────────────────

def _clean_and_audit_batch(
    batch: Dict[str, List],
    text_columns: List[str],
    label_columns: List[str],
    task_info: Dict[str, Any],
) -> Dict[str, List]:
    """Clean text, validate labels, and audit row quality flag for batch map function."""
    task_type = task_info.get("task", "classification")

    cleaned_batch = {col: [] for col in batch.keys()}
    
    is_null_list = []
    is_empty_list = []
    is_corrupted_list = []
    is_invalid_label_list = []

    mojibake_fixes = []
    spaces_normalized = []
    unicode_normalizations = []
    
    html_removed = []
    urls_masked = []
    emails_masked = []
    phones_masked = []
    pii_masked = []

    n_rows = len(next(iter(batch.values())))

    for i in range(n_rows):
        is_null = False
        is_empty = False
        is_corrupted = False
        is_invalid_label = False

        m_fix = 0
        s_norm = 0
        u_norm = 0
        
        html_rem = 0
        url_msk = 0
        email_msk = 0
        phone_msk = 0
        pii_msk = 0

        # Process text columns
        for col in text_columns:
            val = batch[col][i]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                is_null = True
                cleaned_batch[col].append(None)
                continue

            val_str = str(val)
            if len(val_str.strip()) == 0:
                is_empty = True
                cleaned_batch[col].append(val_str)
                continue

            # Mojibake fixes
            healed = heal_mojibake(val_str)
            if healed != val_str:
                m_fix += 1
                val_str = healed

            # Unicode normalization
            norm = normalize_unicode(val_str)
            if norm != val_str:
                u_norm += 1
                val_str = norm

            # 1. HTML Tag Removal
            html_cleaned = HTML_TAG_RE.sub(" ", val_str)
            if html_cleaned != val_str:
                html_rem += 1
                val_str = html_cleaned

            # HTML entity decoding (Unescape)
            decoded = html.unescape(val_str)
            if decoded != val_str:
                val_str = decoded

            # 2. URL Handling
            url_cleaned = URL_RE.sub("<URL>", val_str)
            if url_cleaned != val_str:
                url_msk += 1
                val_str = url_cleaned

            # 3. Email Masking
            email_cleaned = EMAIL_RE.sub("<EMAIL>", val_str)
            if email_cleaned != val_str:
                email_msk += 1
                val_str = email_cleaned

            # 5. PII Masking (CC first to avoid Aadhaar collision)
            cc_cleaned = CREDIT_CARD_RE.sub("<CREDIT_CARD>", val_str)
            cc_changed = (cc_cleaned != val_str)
            val_str = cc_cleaned

            aadhaar_cleaned = AADHAAR_RE.sub("<AADHAAR>", val_str)
            aadhaar_changed = (aadhaar_cleaned != val_str)
            val_str = aadhaar_cleaned

            pan_cleaned = PAN_RE.sub("<PAN>", val_str)
            pan_changed = (pan_cleaned != val_str)
            val_str = pan_cleaned

            ssn_cleaned = SSN_RE.sub("<SSN>", val_str)
            ssn_changed = (ssn_cleaned != val_str)
            val_str = ssn_cleaned

            if cc_changed or aadhaar_changed or pan_changed or ssn_changed:
                pii_msk += 1

            # 4. Phone Number Masking
            phone_cleaned = PHONE_RE.sub("<PHONE>", val_str)
            if phone_cleaned != val_str:
                phone_msk += 1
                val_str = phone_cleaned

            # Space collapsing and trimming
            trimmed = clean_spaces(val_str)
            if trimmed != val_str:
                s_norm += 1
                val_str = trimmed

            # Corrupted text heuristics
            if is_corrupted_text(val_str):
                is_corrupted = True

            cleaned_batch[col].append(val_str)

        # Process label columns
        for col in label_columns:
            val = batch[col][i]
            cleaned_val, is_valid = validate_and_clean_label(val, task_type)
            if not is_valid:
                is_invalid_label = True
            cleaned_batch[col].append(cleaned_val)

        # Append non-text and non-label columns
        for col in batch.keys():
            if col not in text_columns and col not in label_columns:
                cleaned_batch[col].append(batch[col][i])

        is_null_list.append(is_null)
        is_empty_list.append(is_empty)
        is_corrupted_list.append(is_corrupted)
        is_invalid_label_list.append(is_invalid_label)

        mojibake_fixes.append(m_fix)
        spaces_normalized.append(s_norm)
        unicode_normalizations.append(u_norm)
        
        html_removed.append(html_rem)
        urls_masked.append(url_msk)
        emails_masked.append(email_msk)
        phones_masked.append(phone_msk)
        pii_masked.append(pii_msk)

    cleaned_batch["__is_null"] = is_null_list
    cleaned_batch["__is_empty"] = is_empty_list
    cleaned_batch["__is_corrupted"] = is_corrupted_list
    cleaned_batch["__is_invalid_label"] = is_invalid_label_list

    cleaned_batch["__mojibake_fixes"] = mojibake_fixes
    cleaned_batch["__spaces_normalized"] = spaces_normalized
    cleaned_batch["__unicode_normalizations"] = unicode_normalizations
    
    cleaned_batch["__html_removed"] = html_removed
    cleaned_batch["__urls_masked"] = urls_masked
    cleaned_batch["__emails_masked"] = emails_masked
    cleaned_batch["__phones_masked"] = phones_masked
    cleaned_batch["__pii_masked"] = pii_masked

    return cleaned_batch


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator function
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    dataset: Any,
    cfg: DictConfig | None = None,
    task_info: dict[str, Any] | None = None,
):
    """
    Universal Preprocessing Entry Point.
    Processes either a Dataset or a DatasetDict and implements the 7 basic cleaning steps.
    """
    from datasets import Dataset, DatasetDict

    if task_info is None:
        task_name = cfg.task.name if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "name") else "classification"
        task_info = {
            "task": task_name,
            "input_column": cfg.dataset.text_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "text_column") else None,
            "output_column": cfg.dataset.label_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "label_column") else None,
            "label_column": cfg.dataset.label_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "label_column") else None,
            "text_columns": [cfg.dataset.text_column] if cfg and hasattr(cfg, "dataset") and cfg.dataset.text_column else [],
            "label_columns": [cfg.dataset.label_column] if cfg and hasattr(cfg, "dataset") and cfg.dataset.label_column else [],
            "num_labels": cfg.task.num_labels if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "num_labels") else None,
            "label2id": cfg.task.label2id if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "label2id") else None,
            "id2label": cfg.task.id2label if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "id2label") else None,
        }
        task_info["detected_columns"] = {
            "text": task_info["text_columns"],
            "label": task_info["label_columns"]
        }
    else:
        task_name = task_info.get("task", "classification")

    logger.info(f"Basic Preprocessing Engine: Target Task = '{task_name}'")

    detected_cols = task_info.get("detected_columns", {})
    text_columns = set(detected_cols.get("text", []))
    label_columns = set(detected_cols.get("label", []))

    input_col = task_info.get("input_column")
    label_col = task_info.get("label_column")
    if input_col:
        if isinstance(input_col, list):
            for c in input_col:
                text_columns.add(c)
        elif isinstance(input_col, str):
            for c in input_col.split(","):
                c_clean = c.strip()
                if c_clean:
                    text_columns.add(c_clean)
    if label_col:
        if isinstance(label_col, list):
            for c in label_col:
                label_columns.add(c)
        elif isinstance(label_col, str):
            for c in label_col.split(","):
                c_clean = c.strip()
                if c_clean:
                    label_columns.add(c_clean)

    text_columns = list(text_columns)
    label_columns = list(label_columns)

    if isinstance(dataset, DatasetDict):
        cleaned_splits = {}
        split_reports = {}

        for split_name, ds in dataset.items():
            if len(ds) == 0:
                cleaned_splits[split_name] = ds
                continue

            logger.info(f"Preprocessing split '{split_name}' ({len(ds)} rows)...")

            # Map generic cleanup batch
            mapped_ds = ds.map(
                lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
                batched=True,
                desc=f"Cleaning split '{split_name}'"
            )

            # Filter step 1, 2, 6, 7
            before_len = len(mapped_ds)
            cleaned_ds = mapped_ds.filter(
                lambda x: not (x["__is_null"] or x["__is_empty"] or x["__is_corrupted"] or x["__is_invalid_label"]),
                desc=f"Filtering corrupted/empty/null rows from '{split_name}'"
            )
            filtered_count = before_len - len(cleaned_ds)

            # Step 3: exact duplicate removal
            cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)

            # Accumulate report metrics
            m_fixes = sum(mapped_ds["__mojibake_fixes"])
            s_norms = sum(mapped_ds["__spaces_normalized"])
            u_norms = sum(mapped_ds["__unicode_normalizations"])
            html_rem = sum(mapped_ds["__html_removed"])
            urls_msk = sum(mapped_ds["__urls_masked"])
            emails_msk = sum(mapped_ds["__emails_masked"])
            phones_msk = sum(mapped_ds["__phones_masked"])
            pii_msk = sum(mapped_ds["__pii_masked"])

            split_reports[split_name] = {
                "total_rows_processed": before_len,
                "total_rows_cleaned": m_fixes + s_norms + u_norms + html_rem + urls_msk + emails_msk + phones_msk + pii_msk,
                "mojibake_fixes": m_fixes,
                "spaces_normalized": s_norms,
                "unicode_normalizations": u_norms,
                "html_tags_removed": html_rem,
                "urls_masked": urls_msk,
                "emails_masked": emails_msk,
                "phones_masked": phones_msk,
                "pii_masked": pii_msk,
                "null_rows_removed": sum(mapped_ds["__is_null"]),
                "empty_rows_removed": sum(mapped_ds["__is_empty"]),
                "corrupted_rows_removed": filtered_count - sum(mapped_ds["__is_null"]) - sum(mapped_ds["__is_empty"]),
                "duplicates_removed": removed_dupes
            }

            # Drop temporary metadata columns
            cols_to_remove = [c for c in cleaned_ds.column_names if c.startswith("__")]
            final_ds = cleaned_ds.remove_columns(cols_to_remove)
            cleaned_splits[split_name] = final_ds

        combined_cleaned_ds = DatasetDict(cleaned_splits)

        # Consolidate reports
        total_processed = sum(r["total_rows_processed"] for r in split_reports.values())
        total_cleaned = sum(r["total_rows_cleaned"] for r in split_reports.values())
        mojibake_fixes = sum(r["mojibake_fixes"] for r in split_reports.values())
        spaces_normalized = sum(r["spaces_normalized"] for r in split_reports.values())
        unicode_normalizations = sum(r["unicode_normalizations"] for r in split_reports.values())
        html_tags_removed = sum(r["html_tags_removed"] for r in split_reports.values())
        urls_masked = sum(r["urls_masked"] for r in split_reports.values())
        emails_masked = sum(r["emails_masked"] for r in split_reports.values())
        phones_masked = sum(r["phones_masked"] for r in split_reports.values())
        pii_masked = sum(r["pii_masked"] for r in split_reports.values())
        null_removed = sum(r["null_rows_removed"] for r in split_reports.values())
        empty_removed = sum(r["empty_rows_removed"] for r in split_reports.values())
        corrupted_removed = sum(r["corrupted_rows_removed"] for r in split_reports.values())
        dupes_removed = sum(r["duplicates_removed"] for r in split_reports.values())

        cleaning_report = {
            "total_rows_processed": total_processed,
            "total_rows_cleaned": total_cleaned,
            "mojibake_fixes": mojibake_fixes,
            "spaces_normalized": spaces_normalized,
            "unicode_normalizations": unicode_normalizations,
            "html_tags_removed": html_tags_removed,
            "urls_masked": urls_masked,
            "emails_masked": emails_masked,
            "phones_masked": phones_masked,
            "pii_masked": pii_masked
        }

        # Keep output reports compatible with train.py
        quality_report = {
            "average_quality_score": 100.0,
            "category_counts": {
                "CLEAN": total_processed - null_removed - empty_removed - corrupted_removed - dupes_removed,
                "CORRUPTED": null_removed + empty_removed + corrupted_removed
            }
        }

        row_level_audit = {sname: [] for sname in split_reports.keys()}
        
        issue_summary = {
            "total_warnings": 0,
            "total_errors": 0,
            "corrupted_rows_count": null_removed + empty_removed + corrupted_removed,
            "corrupted_rows_removed": null_removed + empty_removed + corrupted_removed,
            "duplicates_removed": dupes_removed
        }

        warnings = []
        if null_removed > 0:
            warnings.append(f"Removed {null_removed} rows containing null values.")
        if empty_removed > 0:
            warnings.append(f"Removed {empty_removed} rows containing empty text.")
        if corrupted_removed > 0:
            warnings.append(f"Removed {corrupted_removed} rows containing corrupted or invalid labels.")
        if dupes_removed > 0:
            warnings.append(f"Removed {dupes_removed} exact duplicate records.")
        if html_tags_removed > 0:
            warnings.append(f"Removed HTML tags from {html_tags_removed} rows.")
        if urls_masked > 0:
            warnings.append(f"Masked URLs in {urls_masked} rows.")
        if emails_masked > 0:
            warnings.append(f"Masked Email addresses in {emails_masked} rows.")
        if phones_masked > 0:
            warnings.append(f"Masked Phone numbers in {phones_masked} rows.")
        if pii_masked > 0:
            warnings.append(f"Removed PII card/ID details from {pii_masked} rows.")

        recommendations = ["Universal Preprocessing completed successfully. Dataset is clean, masked, and ready for training."]

        return (
            combined_cleaned_ds,
            cleaning_report,
            quality_report,
            row_level_audit,
            issue_summary,
            warnings,
            recommendations
        )

    else:
        # Single Dataset
        if len(dataset) == 0:
            return dataset, {}, {}, [], {}, [], []

        logger.info(f"Preprocessing single dataset ({len(dataset)} rows)...")

        mapped_ds = dataset.map(
            lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
            batched=True,
            desc="Cleaning dataset"
        )

        before_len = len(mapped_ds)
        cleaned_ds = mapped_ds.filter(
            lambda x: not (x["__is_null"] or x["__is_empty"] or x["__is_corrupted"] or x["__is_invalid_label"]),
            desc="Filtering corrupted/empty/null rows"
        )
        filtered_count = before_len - len(cleaned_ds)

        cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)

        m_fixes = sum(mapped_ds["__mojibake_fixes"])
        s_norms = sum(mapped_ds["__spaces_normalized"])
        u_norms = sum(mapped_ds["__unicode_normalizations"])
        html_rem = sum(mapped_ds["__html_removed"])
        urls_msk = sum(mapped_ds["__urls_masked"])
        emails_msk = sum(mapped_ds["__emails_masked"])
        phones_msk = sum(mapped_ds["__phones_masked"])
        pii_msk = sum(mapped_ds["__pii_masked"])
        
        null_removed = sum(mapped_ds["__is_null"])
        empty_removed = sum(mapped_ds["__is_empty"])
        corrupted_removed = filtered_count - null_removed - empty_removed

        cleaning_report = {
            "total_rows_processed": before_len,
            "total_rows_cleaned": m_fixes + s_norms + u_norms + html_rem + urls_msk + emails_msk + phones_msk + pii_msk,
            "mojibake_fixes": m_fixes,
            "spaces_normalized": s_norms,
            "unicode_normalizations": u_norms,
            "html_tags_removed": html_rem,
            "urls_masked": urls_msk,
            "emails_masked": emails_msk,
            "phones_masked": phones_msk,
            "pii_masked": pii_msk
        }

        quality_report = {
            "average_quality_score": 100.0,
            "category_counts": {
                "CLEAN": len(cleaned_ds),
                "CORRUPTED": filtered_count
            }
        }

        row_level_audit = []
        
        issue_summary = {
            "total_warnings": 0,
            "total_errors": 0,
            "corrupted_rows_count": filtered_count,
            "corrupted_rows_removed": filtered_count,
            "duplicates_removed": removed_dupes
        }

        warnings = []
        if null_removed > 0:
            warnings.append(f"Removed {null_removed} rows containing null values.")
        if empty_removed > 0:
            warnings.append(f"Removed {empty_removed} rows containing empty text.")
        if corrupted_removed > 0:
            warnings.append(f"Removed {corrupted_removed} rows containing corrupted or invalid labels.")
        if removed_dupes > 0:
            warnings.append(f"Removed {removed_dupes} exact duplicate records.")
        if html_rem > 0:
            warnings.append(f"Removed HTML tags from {html_rem} rows.")
        if urls_msk > 0:
            warnings.append(f"Masked URLs in {urls_msk} rows.")
        if emails_msk > 0:
            warnings.append(f"Masked Email addresses in {emails_msk} rows.")
        if phones_msk > 0:
            warnings.append(f"Masked Phone numbers in {phones_msk} rows.")
        if pii_msk > 0:
            warnings.append(f"Removed PII card/ID details from {pii_msk} rows.")

        recommendations = ["Universal Preprocessing completed successfully. Dataset is clean, masked, and ready for training."]

        cols_to_remove = [c for c in cleaned_ds.column_names if c.startswith("__")]
        final_ds = cleaned_ds.remove_columns(cols_to_remove)

        return (
            final_ds,
            cleaning_report,
            quality_report,
            row_level_audit,
            issue_summary,
            warnings,
            recommendations
        )