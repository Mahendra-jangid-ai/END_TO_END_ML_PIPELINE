# """
# Universal NLP Dataset Preprocessing Engine
# Automatically detects the NLP task type, profiles the schema, applies generic cleaning,
# audits row-level quality (0-100), performs task-specific sanitization and validation,
# and executes advanced checks for target leakage and split leakage.
# """
# from __future__ import annotations

# import html
# import re
# import unicodedata
# import json
# import difflib
# from typing import Any, Dict, List, Optional, Tuple, Set
# from omegaconf import DictConfig

# from src.utils.common import get_logger

# logger = get_logger(__name__)

# # Constants
# NULL_VALUES = {
#     "", "na", "n/a", "null", "none", "nan", "missing"
# }

# # Regex Patterns for Noise Detection
# URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
# PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
# HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# MARKDOWN_PATTERN = re.compile(r"\*\*|__|^#+\s|\[.*?\]\(.*?\)", re.MULTILINE)
# RT_PATTERN = re.compile(r"\bRT\b")
# MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
# HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
# INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200d\uFEFF\u00AD]")
# OCR_CORRUPTION_PATTERN = re.compile(r"\b[a-zA-Z]+[0-9]+[a-zA-Z]+\b")
# REPEATED_PUNCT_PATTERN = re.compile(r"[.,;:!?\-+=_*]{3,}")
# REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-Z])\1{3,}")


# def heal_mojibake(text: str) -> str:
#     """Heal common Mojibake encoding artifacts."""
#     if not isinstance(text, str):
#         return text
#     # Try dynamic healing first
#     try:
#         if any(c in text for c in "âãåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"):
#             re_encoded = text.encode('cp1252').decode('utf-8')
#             return re_encoded
#     except (UnicodeEncodeError, UnicodeDecodeError):
#         pass
        
#     # Static mappings for remaining issues
#     mojibake_map = {
#         "â€™": "'",
#         "â€œ": '"',
#         "â€": '"',
#         "â€¢": "•",
#         "â€”": "—",
#         "â€“": "–",
#         "â€": "",
#         "Ã©": "é",
#         "Ã¡": "á",
#         "Ã³": "ó",
#         "Ãº": "ú",
#         "Ã±": "ñ",
#         "Ã­": "í",
#         "Ã": "à",
#     }
#     for bad, good in mojibake_map.items():
#         text = text.replace(bad, good)
#     return text


# def clean_text_generic(text: Any) -> tuple[Any, dict[str, int]]:
#     """
#     Applies generic text sanitization.
#     Returns: (cleaned_text, change_counts)
#     """
#     if not isinstance(text, str):
#         return text, {}
        
#     counts = {
#         "mojibake_fixes": 0,
#         "html_escapes": 0,
#         "spaces_normalized": 0,
#         "unicode_normalizations": 0
#     }
    
#     orig = text
    
#     # 1. HTML entity decoding
#     unescaped = html.unescape(text)
#     if unescaped != text:
#         counts["html_escapes"] += 1
#         text = unescaped
        
#     # 2. Heal mojibake
#     healed = heal_mojibake(text)
#     if healed != text:
#         counts["mojibake_fixes"] += 1
#         text = healed
        
#     # 3. Unicode normalization
#     normalized = unicodedata.normalize("NFKC", text)
#     if normalized != text:
#         counts["unicode_normalizations"] += 1
#         text = normalized
        
#     # 4. Remove control characters (except tab and newline)
#     cleaned_chars = []
#     for ch in text:
#         if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"):
#             cleaned_chars.append(ch)
#     text = "".join(cleaned_chars)
    
#     # 5. Normalize smart quotes and dashes to standard ASCII equivalents
#     quote_map = {
#         '“': '"', '”': '"', '‟': '"', '″': '"', '′': "'", '‘': "'", '’': "'"
#     }
#     for q_src, q_tgt in quote_map.items():
#         text = text.replace(q_src, q_tgt)
        
#     dash_map = {
#         '–': '-', '—': '-', '―': '-', '−': '-'
#     }
#     for d_src, d_tgt in dash_map.items():
#         text = text.replace(d_src, d_tgt)
        
#     # 6. Normalize tabs and spacing
#     text = text.replace("\t", " ")
    
#     # Replace multiple spaces with single space
#     space_norm = re.sub(r"[ \t]+", " ", text)
#     if space_norm != text:
#         counts["spaces_normalized"] += 1
#         text = space_norm
        
#     # 7. Normalize newlines
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     text = re.sub(r"\n{3,}", "\n\n", text)
    
#     # Trim whitespace
#     text = text.strip()
    
#     return text, counts


# def analyze_noise(text: str) -> dict[str, float]:
#     """Detect noise sources and anomalies in text."""
#     if not isinstance(text, str):
#         return {}
        
#     urls = URL_PATTERN.findall(text)
#     emails = EMAIL_PATTERN.findall(text)
#     phones = PHONE_PATTERN.findall(text)
#     html_tags = HTML_TAG_PATTERN.findall(text)
#     markdown = MARKDOWN_PATTERN.findall(text)
#     rt = RT_PATTERN.findall(text)
#     mentions = MENTION_PATTERN.findall(text)
#     hashtags = HASHTAG_PATTERN.findall(text)
#     invisible = INVISIBLE_UNICODE_PATTERN.findall(text)
#     ocr = OCR_CORRUPTION_PATTERN.findall(text)
    
#     # Keyboard mash heuristic: consonant ratio and repetition check
#     words = text.split()
#     mash_count = 0
#     for w in words:
#         if len(w) >= 5:
#             vowels = set("aeiouAEIOU")
#             consonants = sum(1 for c in w if c.isalpha() and c not in vowels)
#             letters = sum(1 for c in w if c.isalpha())
#             if letters > 0 and consonants / letters >= 0.80:
#                 mash_count += 1
#             elif any(w.count(w[i:i+3]) >= 3 for i in range(len(w)-3)):
#                 mash_count += 1
                
#     # Broken tokenization (e.g. hello.world)
#     broken_tok = re.findall(r"\b[a-zA-Z]+[.,;:!?][a-zA-Z]+\b", text)
    
#     # Repeated punctuation/characters
#     rep_punct = REPEATED_PUNCT_PATTERN.findall(text)
#     rep_chars = REPEATED_CHAR_PATTERN.findall(text)
    
#     # Low information text
#     is_low_info = len(text.strip()) < 5 or len(words) < 2
    
#     return {
#         "urls": len(urls),
#         "emails": len(emails),
#         "phones": len(phones),
#         "html_tags": len(html_tags),
#         "markdown": len(markdown),
#         "rt": len(rt),
#         "mentions": len(mentions),
#         "hashtags": len(hashtags),
#         "invisible": len(invisible),
#         "ocr": len(ocr),
#         "keyboard_mash": mash_count,
#         "broken_tokenization": len(broken_tok),
#         "repeated_punctuation": len(rep_punct),
#         "repeated_chars": len(rep_chars),
#         "low_information": 1.0 if is_low_info else 0.0
#     }


# def compute_row_quality(row: dict[str, Any], text_cols: list[str], label_cols: list[str], task_info: dict[str, Any]) -> tuple[float, str]:
#     """Calculate quality score (0-100) and category for a single row."""
#     input_col = task_info.get("input_column")
#     label_col = task_info.get("label_column")
#     task = task_info.get("task")
    
#     if not input_col or row.get(input_col) is None:
#         return 0.0, "CORRUPTED"
        
#     score = 100.0
    
#     # Check text noise
#     for col in text_cols:
#         val = row.get(col)
#         if not isinstance(val, str):
#             continue
            
#         noise = analyze_noise(val)
#         if not noise:
#             continue
            
#         score -= noise.get("html_tags", 0) * 15
#         score -= noise.get("ocr", 0) * 20
#         score -= noise.get("keyboard_mash", 0) * 30
#         score -= noise.get("broken_tokenization", 0) * 10
#         score -= noise.get("repeated_punctuation", 0) * 5
#         score -= noise.get("repeated_chars", 0) * 5
#         score -= noise.get("invisible", 0) * 5
#         score -= noise.get("urls", 0) * 10
#         score -= noise.get("emails", 0) * 10
#         score -= noise.get("phones", 0) * 10
        
#         if noise.get("low_information", 0.0) > 0.0:
#             score -= 40
            
#     score = max(0.0, min(100.0, score))
    
#     # Determine base category
#     if score >= 90:
#         category = "CLEAN"
#     elif score >= 70:
#         category = "FIXABLE"
#     elif score >= 40:
#         category = "LOW_QUALITY"
#     else:
#         category = "CORRUPTED"
        
#     # Check if target label is missing for label-dependent tasks
#     if label_col and row.get(label_col) is None:
#         if task in ("classification", "regression"):
#             category = "CORRUPTED"
#             score = min(score, 30.0)
            
#     return score, category





# def clean_row_task_specific(row: dict[str, Any], task_info: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
#     """Apply cleaning rules and validations tailored to the NLP task."""
#     task = task_info.get("task")
#     input_col = task_info.get("input_column")
#     label_col = task_info.get("label_column")
    
#     cleaned = row.copy()
#     warnings = []
#     errors = []
    
#     # Classification / Single Label
#     if task in ("classification", "sentiment_analysis"):
#         if label_col in cleaned and cleaned[label_col] is not None:
#             val = cleaned[label_col]
#             if isinstance(val, str):
#                 cleaned[label_col] = val.strip().lower()
#             elif isinstance(val, bool):
#                 cleaned[label_col] = str(val).lower()
                
#     # Regression
#     elif task == "regression":
#         if label_col in cleaned and cleaned[label_col] is not None:
#             val = cleaned[label_col]
#             try:
#                 cleaned[label_col] = float(val)
#             except (ValueError, TypeError):
#                 errors.append(f"Invalid numeric value '{val}' in regression target")
#                 cleaned[label_col] = None

#     return cleaned, warnings, errors


# def check_target_leakage(dataset, text_cols: list[str], label_col: str | None, task_name: str) -> list[str]:
#     """Scan sample of dataset for target label leakage in input text fields."""
#     warnings = []
#     if not label_col or not text_cols:
#         return warnings
        
#     sample_size = min(len(dataset), 200)
#     leakage_count = 0
    
#     for i in range(sample_size):
#         row = dataset[i]
#         label_val = str(row.get(label_col) or "").strip().lower()
#         if not label_val or len(label_val) < 3:
#             continue
            
#         for text_col in text_cols:
#             text_val = str(row.get(text_col) or "").strip().lower()
#             if label_val in text_val and task_name == "classification":
#                 leakage_count += 1
#                 break
                
#     if sample_size > 0 and (leakage_count / sample_size) > 0.15:
#         warnings.append(f"Verbatim target leakage detected: label column '{label_col}' is present inside input texts in {leakage_count}/{sample_size} checked rows.")
        
#     return warnings


# def check_train_test_leakage(dataset_dict) -> list[str]:
#     """Scan across dataset splits to check for identical input text leakage."""
#     warnings = []
#     if not isinstance(dataset_dict, dict) or len(dataset_dict) <= 1:
#         return warnings
        
#     train_split = dataset_dict.get("train")
#     if not train_split or len(train_split) == 0:
#         return warnings
        
#     columns = train_split.column_names
#     # Try to identify main text fields
#     text_cols = [c for c in columns if train_split.features[c].dtype == "string"] if hasattr(train_split, "features") else []
#     if not text_cols:
#         text_cols = [columns[0]]
        
#     train_signatures = set()
#     sample_train_size = min(len(train_split), 5000)
#     for i in range(sample_train_size):
#         row = train_split[i]
#         sig = "||".join(str(row.get(c) or "").strip() for c in text_cols)
#         if sig:
#             train_signatures.add(sig)
            
#     for split_name, ds in dataset_dict.items():
#         if split_name == "train" or len(ds) == 0:
#             continue
            
#         overlap_count = 0
#         sample_test_size = min(len(ds), 1000)
#         for i in range(sample_test_size):
#             row = ds[i]
#             sig = "||".join(str(row.get(c) or "").strip() for c in text_cols)
#             if sig in train_signatures:
#                 overlap_count += 1
                
#         if sample_test_size > 0 and (overlap_count / sample_test_size) > 0.01:
#             warnings.append(f"Train-Test leakage: Split '{split_name}' contains {overlap_count}/{sample_test_size} rows matching train exactly.")
            
#     return warnings


# def remove_duplicate_rows(dataset, text_columns: list[str]) -> tuple[Any, int]:
#     """Remove duplicate rows from dataset based on text columns."""
#     before = len(dataset)
#     try:
#         import pandas as pd
#         if text_columns:
#             subset = list(text_columns)
#             cols_to_remove = [c for c in dataset.column_names if c not in subset]
#             small_ds = dataset.remove_columns(cols_to_remove)
#             df_small = small_ds.to_pandas()
#             df_unique = df_small.drop_duplicates(subset=subset)
#             unique_indices = df_unique.index.tolist()
#         else:
#             df_all = dataset.to_pandas()
#             df_unique = df_all.drop_duplicates()
#             unique_indices = df_unique.index.tolist()
            
#         removed = before - len(unique_indices)
#         if removed > 0:
#             logger.info(f"Removed {removed} duplicate rows")
#             dataset = dataset.select(unique_indices)
#         return dataset, removed
#     except Exception as e:
#         logger.warning(f"Duplicate removal skipped: {e}")
#         return dataset, 0


# def _clean_and_audit_batch(batch, text_columns, label_columns, task_info):
#     """Batch mapping cleaning and auditing function."""
#     task = task_info.get("task")
#     label_col = task_info.get("label_column")
    
#     cleaned_batch = {col: [] for col in batch.keys()}
    
#     quality_scores = []
#     quality_categories = []
#     audit_issues_list = []
#     mojibake_fixes = []
#     html_escapes = []
#     spaces_normalized = []
#     unicode_normalizations = []
#     task_specific_updates = []
    
#     n_rows = len(next(iter(batch.values())))
    
#     for i in range(n_rows):
#         row = {col: batch[col][i] for col in batch.keys()}
        
#         row_mojibake = 0
#         row_html = 0
#         row_space = 0
#         row_unicode = 0
#         row_task = 0
#         row_issues = []
        
#         cleaned_row = row.copy()
#         for col in text_columns:
#             val = row.get(col)
#             if isinstance(val, str):
#                 cleaned_val, counts = clean_text_generic(val)
#                 cleaned_row[col] = cleaned_val
                
#                 row_mojibake += counts.get("mojibake_fixes", 0)
#                 row_html += counts.get("html_escapes", 0)
#                 row_space += counts.get("spaces_normalized", 0)
#                 row_unicode += counts.get("unicode_normalizations", 0)
                
#                 if cleaned_val != val:
#                     if counts.get("mojibake_fixes", 0) > 0:
#                         row_issues.append("Mojibake fixed")
#                     if counts.get("html_escapes", 0) > 0:
#                         row_issues.append("HTML entities decoded")
#                     if counts.get("spaces_normalized", 0) > 0:
#                         row_issues.append("Whitespace normalized")
#             elif isinstance(val, list):
#                 cleaned_list = []
#                 list_changed = False
#                 for item in val:
#                     if isinstance(item, str):
#                         cleaned_item, counts = clean_text_generic(item)
#                         cleaned_list.append(cleaned_item)
#                         if cleaned_item != item:
#                             list_changed = True
#                             row_mojibake += counts.get("mojibake_fixes", 0)
#                             row_html += counts.get("html_escapes", 0)
#                             row_space += counts.get("spaces_normalized", 0)
#                             row_unicode += counts.get("unicode_normalizations", 0)
#                     else:
#                         cleaned_list.append(item)
#                 cleaned_row[col] = cleaned_list
#                 if list_changed:
#                     row_issues.append("Sequence strings normalized")
                    
#         cleaned_row, task_warnings, task_errors = clean_row_task_specific(cleaned_row, task_info)
        
#         for col in batch.keys():
#             if cleaned_row.get(col) != row.get(col):
#                 if col == label_col and task == "regression":
#                     try:
#                         if float(row.get(col)) == float(cleaned_row.get(col)):
#                             continue
#                     except (ValueError, TypeError):
#                         pass
#                 row_task += 1
                
#         for warn in task_warnings:
#             row_issues.append(f"Warning: {warn}")
#         for err in task_errors:
#             row_issues.append(f"Error: {err}")
            
#         score, category = compute_row_quality(row, text_columns, label_columns, task_info)
        
#         if task_errors:
#             category = "CORRUPTED"
#             score = min(score, 30.0)
            
#         quality_scores.append(score)
#         quality_categories.append(category)
#         audit_issues_list.append(json.dumps(row_issues))
        
#         mojibake_fixes.append(row_mojibake)
#         html_escapes.append(row_html)
#         spaces_normalized.append(row_space)
#         unicode_normalizations.append(row_unicode)
#         task_specific_updates.append(row_task)
        
#         for col in batch.keys():
#             cleaned_batch[col].append(cleaned_row.get(col))
            
#     cleaned_batch["__quality_score"] = quality_scores
#     cleaned_batch["__quality_category"] = quality_categories
#     cleaned_batch["__audit_issues"] = audit_issues_list
#     cleaned_batch["__mojibake_fixes"] = mojibake_fixes
#     cleaned_batch["__html_escapes"] = html_escapes
#     cleaned_batch["__spaces_normalized"] = spaces_normalized
#     cleaned_batch["__unicode_normalizations"] = unicode_normalizations
#     cleaned_batch["__task_specific_updates"] = task_specific_updates
    
#     return cleaned_batch


# def compile_reports(dataset, split_name: str | None = None) -> tuple[dict, dict, list, dict]:
#     """Consolidate batch quality/audit logs into final summary metrics."""
#     scores = dataset["__quality_score"]
#     categories = dataset["__quality_category"]
#     issues_json = dataset["__audit_issues"]
    
#     mojibake = sum(dataset["__mojibake_fixes"])
#     html_e = sum(dataset["__html_escapes"])
#     spaces = sum(dataset["__spaces_normalized"])
#     unicode = sum(dataset["__unicode_normalizations"])
#     task_up = sum(dataset["__task_specific_updates"])
    
#     total_rows = len(dataset)
#     category_counts = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0, "REVIEW_REQUIRED": 0}
#     for cat in categories:
#         category_counts[cat] = category_counts.get(cat, 0) + 1
        
#     avg_score = sum(scores) / total_rows if total_rows > 0 else 100.0
    
#     total_warnings = 0
#     total_errors = 0
#     for issues_str in issues_json:
#         row_issues = json.loads(issues_str)
#         for issue in row_issues:
#             if issue.startswith("Warning:"):
#                 total_warnings += 1
#             elif issue.startswith("Error:"):
#                 total_errors += 1
                
#     row_level_audit = []
#     limit = min(total_rows, 1000)
#     for idx in range(limit):
#         row_level_audit.append({
#             "row_index": idx,
#             "quality_score": scores[idx],
#             "category": categories[idx],
#             "issues": json.loads(issues_json[idx])
#         })
        
#     quality_report = {
#         "average_quality_score": round(avg_score, 2),
#         "category_counts": category_counts,
#         "category_ratios": {cat: round(cnt / total_rows, 4) for cat, cnt in category_counts.items()} if total_rows > 0 else {}
#     }
    
#     cleaning_report = {
#         "total_rows_processed": total_rows,
#         "total_rows_cleaned": sum(1 for x in issues_json if len(json.loads(x)) > 0),
#         "mojibake_fixes": mojibake,
#         "html_escapes": html_e,
#         "spaces_normalized": spaces,
#         "unicode_normalizations": unicode,
#         "task_specific_updates": task_up
#     }
    
#     issue_summary = {
#         "total_warnings": total_warnings,
#         "total_errors": total_errors,
#         "corrupted_rows_count": category_counts.get("CORRUPTED", 0)
#     }
    
#     return cleaning_report, quality_report, row_level_audit, issue_summary


# def generate_recommendations(quality_report: dict, issue_summary: dict, task_info: dict) -> list[str]:
#     """Produce actionable recommendation steps based on the audit reports."""
#     recs = []
#     avg_score = quality_report.get("average_quality_score", 100.0)
#     if avg_score < 80.0:
#         recs.append(f"Average quality score is low ({avg_score:.2f}). Consider vetting higher quality source data.")
        
#     corrupted_count = issue_summary.get("corrupted_rows_count", 0)
#     if corrupted_count > 0:
#         recs.append(f"Automatically removed {corrupted_count} corrupted rows (missing input/label values or list length mismatches).")
        
#     cat_counts = quality_report.get("category_counts", {})
#     low_qual = cat_counts.get("LOW_QUALITY", 0)
#     if low_qual > 0:
#         recs.append(f"Identified {low_qual} low-quality rows. Inspect for keyboard mashes, excessive noise, or repeated patterns.")
        
#     if issue_summary.get("target_leakage_detected", False):
#         recs.append("WARNING: Target label leakage detected. Verbatim labels exist in feature text. Strip labels before training.")
        
#     if issue_summary.get("train_test_leakage_detected", False):
#         recs.append("WARNING: Split leakage detected. Duplicate strings overlap between train/validation/test. Deduplicate dataset splits.")
        
#     if len(recs) == 0:
#         recs.append("Dataset quality is high. Ready for training.")
        
#     return recs


# def preprocess_dataset(dataset, cfg: DictConfig | None = None, task_info: dict[str, Any] | None = None):
#     """
#     Universal Preprocessing Entry Point.
#     Profiles the schema, runs text sanitation, noise audits, and returns:
#     (cleaned_dataset, cleaning_report, quality_report, row_level_audit, issue_summary, warnings, recommendations)
#     """
#     from datasets import Dataset, DatasetDict
    
#     if task_info is None:
#         task_name = cfg.task.name if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "name") else "classification"
#         task_info = {
#             "task": task_name,
#             "input_column": cfg.dataset.text_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "text_column") else None,
#             "output_column": cfg.dataset.label_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "label_column") else None,
#             "label_column": cfg.dataset.label_column if cfg and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "label_column") else None,
#             "text_columns": [cfg.dataset.text_column] if cfg and hasattr(cfg, "dataset") and cfg.dataset.text_column else [],
#             "label_columns": [cfg.dataset.label_column] if cfg and hasattr(cfg, "dataset") and cfg.dataset.label_column else [],
#             "num_labels": cfg.task.num_labels if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "num_labels") else None,
#             "label2id": cfg.task.label2id if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "label2id") else None,
#             "id2label": cfg.task.id2label if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "id2label") else None,
#         }
#         task_info["detected_columns"] = {
#             "text": task_info["text_columns"],
#             "label": task_info["label_columns"]
#         }
#     else:
#         task_name = task_info.get("task", "classification")
    
#     logger.info(f"Universal Preprocessing Engine: Target Task = '{task_name}'")
    
#     detected_cols = task_info.get("detected_columns", {})
#     text_columns = set(detected_cols.get("text", []))
#     label_columns = set(detected_cols.get("label", []))
    
#     input_col = task_info.get("input_column")
#     label_col = task_info.get("label_column")
#     if input_col:
#         text_columns.add(input_col)
#     if label_col:
#         label_columns.add(label_col)
        
#     text_columns = list(text_columns)
#     label_columns = list(label_columns)
    
#     if isinstance(dataset, DatasetDict):
#         cleaned_splits = {}
#         split_reports = {}
        
#         for split_name, ds in dataset.items():
#             if len(ds) == 0:
#                 cleaned_splits[split_name] = ds
#                 continue
                
#             logger.info(f"Preprocessing split '{split_name}' ({len(ds)} rows)...")
            
#             mapped_ds = ds.map(
#                 lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
#                 batched=True,
#                 desc=f"Cleaning & Auditing '{split_name}' split"
#             )
            
#             cleaning_rep, quality_rep, row_audit, issue_sum = compile_reports(mapped_ds, split_name)
#             split_reports[split_name] = {
#                 "cleaning_report": cleaning_rep,
#                 "quality_report": quality_rep,
#                 "row_level_audit": row_audit,
#                 "issue_summary": issue_sum
#             }
            
#             before_len = len(mapped_ds)
#             cleaned_ds = mapped_ds.filter(
#                 lambda x: x["__quality_category"] != "CORRUPTED",
#                 desc=f"Filtering corrupted rows from '{split_name}'"
#             )
#             removed_corrupted = before_len - len(cleaned_ds)
#             split_reports[split_name]["issue_summary"]["corrupted_rows_removed"] = removed_corrupted
            
#             cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)
#             split_reports[split_name]["issue_summary"]["duplicates_removed"] = removed_dupes
            
#             target_leakage_warnings = check_target_leakage(cleaned_ds, text_columns, label_col, task_name)
            
#             row_warnings = set()
#             for issues_str in mapped_ds["__audit_issues"]:
#                 for issue in json.loads(issues_str):
#                     if issue.startswith("Warning:"):
#                         row_warnings.add(issue.replace("Warning: ", "", 1))
            
#             split_reports[split_name]["warnings"] = target_leakage_warnings + list(row_warnings)
#             split_reports[split_name]["issue_summary"]["target_leakage_detected"] = len(target_leakage_warnings) > 0
            
#             cols_to_remove = [c for c in cleaned_ds.column_names if c.startswith("__")]
#             final_ds = cleaned_ds.remove_columns(cols_to_remove)
#             cleaned_splits[split_name] = final_ds
            
#         combined_cleaned_ds = DatasetDict(cleaned_splits)
        
#         total_rows_processed = 0
#         total_rows_cleaned = 0
#         mojibake_fixes = 0
#         html_escapes = 0
#         spaces_normalized = 0
#         unicode_normalizations = 0
#         task_specific_updates = 0
        
#         total_warnings = 0
#         total_errors = 0
#         total_corrupted = 0
#         total_corrupted_removed = 0
#         total_duplicates_removed = 0
#         target_leakage_detected = False
        
#         category_counts = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0, "REVIEW_REQUIRED": 0}
#         total_score_sum = 0.0
        
#         row_level_audit = {}
#         warnings = []
        
#         for split_name, rep in split_reports.items():
#             total_rows_processed += rep["cleaning_report"]["total_rows_processed"]
#             total_rows_cleaned += rep["cleaning_report"]["total_rows_cleaned"]
#             mojibake_fixes += rep["cleaning_report"]["mojibake_fixes"]
#             html_escapes += rep["cleaning_report"]["html_escapes"]
#             spaces_normalized += rep["cleaning_report"]["spaces_normalized"]
#             unicode_normalizations += rep["cleaning_report"]["unicode_normalizations"]
#             task_specific_updates += rep["cleaning_report"]["task_specific_updates"]
            
#             total_warnings += rep["issue_summary"]["total_warnings"]
#             total_errors += rep["issue_summary"]["total_errors"]
#             total_corrupted += rep["issue_summary"]["corrupted_rows_count"]
#             total_corrupted_removed += rep["issue_summary"]["corrupted_rows_removed"]
#             total_duplicates_removed += rep["issue_summary"]["duplicates_removed"]
#             if rep["issue_summary"]["target_leakage_detected"]:
#                 target_leakage_detected = True
                
#             for cat, cnt in rep["quality_report"]["category_counts"].items():
#                 category_counts[cat] += cnt
#             total_score_sum += rep["quality_report"]["average_quality_score"] * rep["cleaning_report"]["total_rows_processed"]
            
#             row_level_audit[split_name] = rep["row_level_audit"]
#             for w in rep["warnings"]:
#                 warnings.append(f"[{split_name}] {w}")
                
#         split_leakage_warnings = check_train_test_leakage(combined_cleaned_ds)
#         for w in split_leakage_warnings:
#             warnings.append(w)
            
#         avg_quality_score = total_score_sum / total_rows_processed if total_rows_processed > 0 else 100.0
        
#         cleaning_report = {
#             "total_rows_processed": total_rows_processed,
#             "total_rows_cleaned": total_rows_cleaned,
#             "mojibake_fixes": mojibake_fixes,
#             "html_escapes": html_escapes,
#             "spaces_normalized": spaces_normalized,
#             "unicode_normalizations": unicode_normalizations,
#             "task_specific_updates": task_specific_updates
#         }
        
#         quality_report = {
#             "average_quality_score": round(avg_quality_score, 2),
#             "category_counts": category_counts,
#             "category_ratios": {cat: round(cnt / total_rows_processed, 4) for cat, cnt in category_counts.items()} if total_rows_processed > 0 else {}
#         }
        
#         issue_summary = {
#             "total_warnings": total_warnings + len(split_leakage_warnings),
#             "total_errors": total_errors,
#             "corrupted_rows_count": total_corrupted,
#             "corrupted_rows_removed": total_corrupted_removed,
#             "duplicates_removed": total_duplicates_removed,
#             "target_leakage_detected": target_leakage_detected,
#             "train_test_leakage_detected": len(split_leakage_warnings) > 0
#         }
        
#         recommendations = generate_recommendations(quality_report, issue_summary, task_info)
        
#         return (
#             combined_cleaned_ds,
#             cleaning_report,
#             quality_report,
#             row_level_audit,
#             issue_summary,
#             warnings,
#             recommendations
#         )
        
#     else:
#         if len(dataset) == 0:
#             return dataset, {}, {}, [], {}, [], []
            
#         logger.info(f"Preprocessing single dataset ({len(dataset)} rows)...")
        
#         mapped_ds = dataset.map(
#             lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
#             batched=True,
#             desc="Cleaning & Auditing dataset"
#         )
        
#         cleaning_report, quality_report, row_level_audit, issue_summary = compile_reports(mapped_ds)
        
#         before_len = len(mapped_ds)
#         cleaned_ds = mapped_ds.filter(
#             lambda x: x["__quality_category"] != "CORRUPTED",
#             desc="Filtering corrupted rows"
#         )
#         removed_corrupted = before_len - len(cleaned_ds)
#         issue_summary["corrupted_rows_removed"] = removed_corrupted
        
#         cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)
#         issue_summary["duplicates_removed"] = removed_dupes
        
#         target_leakage_warnings = check_target_leakage(cleaned_ds, text_columns, label_col, task_name)
#         issue_summary["target_leakage_detected"] = len(target_leakage_warnings) > 0
        
#         row_warnings = set()
#         for issues_str in mapped_ds["__audit_issues"]:
#             for issue in json.loads(issues_str):
#                 if issue.startswith("Warning:"):
#                     row_warnings.add(issue.replace("Warning: ", "", 1))
                    
#         warnings = target_leakage_warnings + list(row_warnings)
#         recommendations = generate_recommendations(quality_report, issue_summary, task_info)
        
#         cols_to_remove = [c for c in cleaned_ds.column_names if c.startswith("__")]
#         final_ds = cleaned_ds.remove_columns(cols_to_remove)
        
#         return (
#             final_ds,
#             cleaning_report,
#             quality_report,
#             row_level_audit,
#             issue_summary,
#             warnings,
#             recommendations
#         )


"""
Universal NLP Dataset Preprocessing Engine  (v2 — Production Grade)
════════════════════════════════════════════════════════════════════
Pipeline stages:
  1. Generic text sanitization   — encoding heal, HTML, unicode, whitespace
  2. PII detection & redaction   — email, phone, SSN, credit-card, IP, passport, Aadhaar
  3. Advanced noise analysis     — 15+ noise signals per cell
  4. Heuristic quality filtering — language-agnostic word/char ratio guards
  5. Task-specific sanitization  — per-task label normalisation + field validation
  6. Row-level quality scoring   — 0-100 with weighted penalty model
  7. Exact + near-duplicate removal — hash dedup + n-gram Jaccard fuzzy dedup
  8. Target leakage detection    — verbatim + substring label-in-text check
  9. Split (train/test) leakage  — exact-match signature scan
 10. Recommendations & reporting — actionable per-split summaries

Supported NLP tasks (task_info["task"]):
  classification, multi_label_classification, text_pair_classification,
  natural_language_inference, regression, question_answering, summarization,
  translation, instruction_tuning, chat, token_classification,
  information_extraction, dialogue, grammar_correction, text_simplification,
  style_transfer, text_generation, ranking, retrieval, text_similarity,
  sentiment_analysis, keyword_generation, title_generation
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from omegaconf import DictConfig

from src.utils.common import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NULL_VALUES: Set[str] = {"", "na", "n/a", "null", "none", "nan", "missing", "nil", "undefined", "N/A", "NULL"}

# Tasks that REQUIRE a label column to be meaningful
LABEL_REQUIRED_TASKS = {
    "classification", "multi_label_classification", "text_pair_classification",
    "natural_language_inference", "regression", "sentiment_analysis",
    "token_classification", "ranking",
}

# Tasks that use TWO or more text columns
DUAL_TEXT_TASKS = {
    "question_answering", "summarization", "translation",
    "text_pair_classification", "natural_language_inference",
    "grammar_correction", "text_simplification", "style_transfer",
    "instruction_tuning", "ranking", "retrieval", "text_similarity",
    "dialogue",
}

# Tasks where label is itself a generated text (not a short category)
GENERATIVE_LABEL_TASKS = {
    "summarization", "translation", "grammar_correction",
    "text_simplification", "style_transfer", "question_answering",
    "keyword_generation", "title_generation",
}

# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────────────────────────────────────

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
)
SSN_PATTERN = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")
IP_ADDRESS_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
PASSPORT_PATTERN = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_PATTERN = re.compile(r"\*\*|__|^#+\s|\[.*?\]\(.*?\)", re.MULTILINE)
RT_PATTERN = re.compile(r"\bRT\b")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200d\uFEFF\u00AD\u2060]")
OCR_CORRUPTION_PATTERN = re.compile(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b")
REPEATED_PUNCT_PATTERN = re.compile(r"[.,;:!?\-+=_*]{3,}")
REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-Z])\1{3,}")
BROKEN_TOK_PATTERN = re.compile(r"\b[a-zA-Z]+[.,;:!?][a-zA-Z]+\b")
CONTRACTION_MAP = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "let's": "let us", "who's": "who is",
}
CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# PII Redaction
# ─────────────────────────────────────────────────────────────────────────────

# Ordered: more-specific patterns first
PII_RULES: List[Tuple[str, re.Pattern, str]] = [
    ("SSN",          SSN_PATTERN,          "[SSN_REDACTED]"),
    ("CREDIT_CARD",  CREDIT_CARD_PATTERN,  "[CREDIT_CARD_REDACTED]"),
    ("AADHAAR",      AADHAAR_PATTERN,      "[AADHAAR_REDACTED]"),
    ("PASSPORT",     PASSPORT_PATTERN,     "[PASSPORT_REDACTED]"),
    ("EMAIL",        EMAIL_PATTERN,        "[EMAIL_REDACTED]"),
    ("PHONE",        PHONE_PATTERN,        "[PHONE_REDACTED]"),
    ("IP_ADDRESS",   IP_ADDRESS_PATTERN,   "[IP_REDACTED]"),
]


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Replace PII entities with typed placeholders.
    Returns (redacted_text, {pii_type: count}).
    """
    if not isinstance(text, str):
        return text, {}
    found: Dict[str, int] = {}
    for pii_type, pattern, placeholder in PII_RULES:
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = len(matches)
            text = pattern.sub(placeholder, text)
    return text, found


# ─────────────────────────────────────────────────────────────────────────────
# Encoding / Mojibake Healing
# ─────────────────────────────────────────────────────────────────────────────

_MOJIBAKE_STATIC: Dict[str, str] = {
    "â€™": "'", "â€œ": '"', "â€": '"', "â€¢": "•",
    "â€"": "—", "â€"": "–", "Ã©": "é", "Ã¡": "á",
    "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ", "Ã­": "í",
    "Ã": "à", "â€˜": "'", "â€ž": "„", "â€¦": "…",
}


def heal_mojibake(text: str) -> str:
    """Fix common Mojibake (CP1252 mis-decoded as UTF-8 and vice-versa)."""
    if not isinstance(text, str):
        return text
    # Dynamic heal: try re-encoding as cp1252 then decoding as utf-8
    try:
        if any(ord(c) > 127 for c in text):
            candidate = text.encode("cp1252").decode("utf-8")
            # Only accept if result has fewer high-byte chars
            if sum(ord(c) > 127 for c in candidate) < sum(ord(c) > 127 for c in text):
                return candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Static fallback
    for bad, good in _MOJIBAKE_STATIC.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Generic Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

_SMART_QUOTE_MAP: Dict[str, str] = {
    "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u2033": '"',
    "\u2032": "'", "\u2018": "'", "\u2019": "'",
}
_DASH_MAP: Dict[str, str] = {
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
}


def clean_text_generic(
    text: Any,
    expand_contractions: bool = False,
) -> Tuple[Any, Dict[str, int]]:
    """
    Stage-1 generic sanitisation.
    Returns (cleaned_text, change_count_dict).
    """
    if not isinstance(text, str):
        return text, {}

    counts: Dict[str, int] = defaultdict(int)

    # 1. HTML entity decode
    unescaped = html.unescape(text)
    if unescaped != text:
        counts["html_escapes"] += 1
        text = unescaped

    # 2. Mojibake heal
    healed = heal_mojibake(text)
    if healed != text:
        counts["mojibake_fixes"] += 1
        text = healed

    # 3. NFKC unicode normalisation
    norm = unicodedata.normalize("NFKC", text)
    if norm != text:
        counts["unicode_normalizations"] += 1
        text = norm

    # 4. Remove invisible / zero-width unicode
    cleaned = INVISIBLE_UNICODE_PATTERN.sub("", text)
    if cleaned != text:
        counts["invisible_removed"] += 1
        text = cleaned

    # 5. Strip C-category control characters (keep \n \t)
    result_chars: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] != "C" or ch in ("\n", "\t"):
            result_chars.append(ch)
    text = "".join(result_chars)

    # 6. Smart quotes & dashes → ASCII
    for src, tgt in _SMART_QUOTE_MAP.items():
        text = text.replace(src, tgt)
    for src, tgt in _DASH_MAP.items():
        text = text.replace(src, tgt)

    # 7. Tab → space; collapse multiple spaces
    text = text.replace("\t", " ")
    space_norm = re.sub(r"[ \t]+", " ", text)
    if space_norm != text:
        counts["spaces_normalized"] += 1
        text = space_norm

    # 8. Normalise newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 9. Optional contraction expansion (useful for classification)
    if expand_contractions:
        def _expand(m: re.Match) -> str:
            return CONTRACTION_MAP.get(m.group(0).lower(), m.group(0))
        expanded = CONTRACTION_PATTERN.sub(_expand, text)
        if expanded != text:
            counts["contractions_expanded"] += 1
            text = expanded

    text = text.strip()
    return text, dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
# Noise Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_noise(text: str) -> Dict[str, float]:
    """
    Compute 15+ noise signals for a single text string.
    All counts are absolute integers unless stated.
    """
    if not isinstance(text, str) or not text.strip():
        return {"low_information": 1.0}

    words = text.split()
    total_words = len(words)
    total_chars = len(text)

    # Basic regex counts
    noise: Dict[str, float] = {
        "urls":                  len(URL_PATTERN.findall(text)),
        "emails":                len(EMAIL_PATTERN.findall(text)),
        "phones":                len(PHONE_PATTERN.findall(text)),
        "html_tags":             len(HTML_TAG_PATTERN.findall(text)),
        "markdown":              len(MARKDOWN_PATTERN.findall(text)),
        "rt":                    len(RT_PATTERN.findall(text)),
        "mentions":              len(MENTION_PATTERN.findall(text)),
        "hashtags":              len(HASHTAG_PATTERN.findall(text)),
        "invisible":             len(INVISIBLE_UNICODE_PATTERN.findall(text)),
        "ocr_corruption":        len(OCR_CORRUPTION_PATTERN.findall(text)),
        "repeated_punctuation":  len(REPEATED_PUNCT_PATTERN.findall(text)),
        "repeated_chars":        len(REPEATED_CHAR_PATTERN.findall(text)),
        "broken_tokenization":   len(BROKEN_TOK_PATTERN.findall(text)),
    }

    # Keyboard mash: words with ≥80 % consonants or triple-trigram repetition
    vowels = set("aeiouAEIOU")
    mash = 0
    for w in words:
        if len(w) < 5:
            continue
        letters = [c for c in w if c.isalpha()]
        if not letters:
            continue
        consonant_ratio = sum(1 for c in letters if c not in vowels) / len(letters)
        if consonant_ratio >= 0.80:
            mash += 1
        elif len(w) >= 9 and any(w.count(w[i:i+3]) >= 3 for i in range(len(w) - 3)):
            mash += 1
    noise["keyboard_mash"] = mash

    # Low-information text (too short to be useful)
    noise["low_information"] = 1.0 if (total_chars < 5 or total_words < 2) else 0.0

    # Script diversity: multiple non-ASCII scripts mixed (possible garbled text)
    scripts: Set[str] = set()
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
            if name:
                script = name.split()[0]
                scripts.add(script)
        except Exception:
            pass
    noise["script_mixing"] = max(0, len(scripts) - 5)   # penalise >5 distinct scripts

    # High punctuation density
    punc_chars = sum(1 for c in text if unicodedata.category(c).startswith("P"))
    noise["punct_density"] = punc_chars / max(total_chars, 1)

    # Digit ratio (very high → probably not natural language)
    digit_chars = sum(1 for c in text if c.isdigit())
    noise["digit_ratio"] = digit_chars / max(total_chars, 1)

    # All-caps ratio (shouting / OCR artefacts)
    alpha_chars = [c for c in text if c.isalpha()]
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    noise["all_caps_ratio"] = upper_ratio

    return noise


# ─────────────────────────────────────────────────────────────────────────────
# Row Quality Scoring
# ─────────────────────────────────────────────────────────────────────────────

# Penalty weights per noise signal (tuned empirically)
_NOISE_PENALTIES: Dict[str, float] = {
    "html_tags":            15.0,
    "ocr_corruption":       20.0,
    "keyboard_mash":        30.0,
    "broken_tokenization":  10.0,
    "repeated_punctuation":  5.0,
    "repeated_chars":        5.0,
    "invisible":             5.0,
    "urls":                  5.0,
    "emails":                8.0,
    "phones":                8.0,
    "rt":                    3.0,
    "mentions":              3.0,
    "hashtags":              3.0,
    "low_information":      40.0,
    "script_mixing":         8.0,
}


def compute_row_quality(
    row: Dict[str, Any],
    text_cols: List[str],
    label_cols: List[str],
    task_info: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Score 0-100 + category for a single dataset row.
    Category: CLEAN (≥90) | FIXABLE (70–89) | LOW_QUALITY (40–69) | CORRUPTED (<40)
    """
    task = task_info.get("task", "")
    input_col = task_info.get("input_column")
    label_col = task_info.get("label_column")

    # Hard fail: no input text at all
    if not input_col or row.get(input_col) is None:
        return 0.0, "CORRUPTED"

    # Null / empty input
    raw_input = str(row.get(input_col, "")).strip().lower()
    if raw_input in NULL_VALUES or len(raw_input) == 0:
        return 0.0, "CORRUPTED"

    score = 100.0

    # ── Per-column noise penalties ──────────────────────────────────────────
    for col in text_cols:
        val = row.get(col)
        if not isinstance(val, str):
            continue
        noise = analyze_noise(val)
        for sig, pen in _NOISE_PENALTIES.items():
            val_n = noise.get(sig, 0.0)
            if isinstance(val_n, float) and val_n <= 1.0:
                score -= val_n * pen          # fraction signals
            else:
                score -= float(val_n) * pen   # count signals

        # Heuristic: punct_density > 40 % → severe penalty
        if noise.get("punct_density", 0) > 0.40:
            score -= 20.0
        # all-caps > 70 % → moderate penalty
        if noise.get("all_caps_ratio", 0) > 0.70:
            score -= 10.0
        # digit_ratio > 60 % → non-text penalty
        if noise.get("digit_ratio", 0) > 0.60:
            score -= 15.0

    # ── Label checks ────────────────────────────────────────────────────────
    if task in LABEL_REQUIRED_TASKS:
        if not label_col or row.get(label_col) is None:
            return max(0.0, min(score, 20.0)), "CORRUPTED"
        raw_label = str(row.get(label_col, "")).strip().lower()
        if raw_label in NULL_VALUES or raw_label == "":
            return max(0.0, min(score, 20.0)), "CORRUPTED"

    # ── Token-classification: tokens / tags list length must match ───────────
    if task == "token_classification" and label_col:
        tok_col = task_info.get("text_column") or (text_cols[0] if text_cols else None)
        if tok_col:
            tok_val = row.get(tok_col)
            lab_val = row.get(label_col)
            if isinstance(tok_val, list) and isinstance(lab_val, list):
                if len(tok_val) != len(lab_val):
                    score -= 50.0

    score = max(0.0, min(100.0, score))

    if score >= 90:
        return score, "CLEAN"
    elif score >= 70:
        return score, "FIXABLE"
    elif score >= 40:
        return score, "LOW_QUALITY"
    else:
        return score, "CORRUPTED"


# ─────────────────────────────────────────────────────────────────────────────
# Task-Specific Sanitization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label_string(val: Any) -> str:
    """Lowercase + strip classification labels."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, bool):
        return str(val).lower()
    return val


def clean_row_task_specific(
    row: Dict[str, Any],
    task_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Apply task-specific field cleaning / validation.
    Returns (cleaned_row, warnings, errors).
    """
    task = task_info.get("task", "")
    label_col = task_info.get("label_column")
    input_col = task_info.get("input_column")
    cleaned = row.copy()
    warnings: List[str] = []
    errors: List[str] = []

    # ── 1. Classification / Sentiment ───────────────────────────────────────
    if task in ("classification", "sentiment_analysis", "text_pair_classification",
                "natural_language_inference"):
        if label_col and cleaned.get(label_col) is not None:
            cleaned[label_col] = _normalize_label_string(cleaned[label_col])

    # ── 2. Multi-label classification ───────────────────────────────────────
    elif task == "multi_label_classification":
        if label_col and cleaned.get(label_col) is not None:
            val = cleaned[label_col]
            if isinstance(val, str):
                # "label1, label2" → ["label1", "label2"]
                parts = [p.strip().lower() for p in re.split(r"[,;|]", val) if p.strip()]
                cleaned[label_col] = parts
            elif isinstance(val, list):
                cleaned[label_col] = [str(v).strip().lower() for v in val]

    # ── 3. Regression ───────────────────────────────────────────────────────
    elif task == "regression":
        if label_col and cleaned.get(label_col) is not None:
            try:
                cleaned[label_col] = float(cleaned[label_col])
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value '{cleaned[label_col]}' in regression target")
                cleaned[label_col] = None

    # ── 4. Ranking ──────────────────────────────────────────────────────────
    elif task == "ranking":
        if label_col and cleaned.get(label_col) is not None:
            try:
                cleaned[label_col] = int(float(cleaned[label_col]))
            except (ValueError, TypeError):
                errors.append(f"Invalid rank value '{cleaned[label_col]}'")
                cleaned[label_col] = None

    # ── 5. Token classification ─────────────────────────────────────────────
    elif task == "token_classification":
        if label_col and isinstance(cleaned.get(label_col), list):
            cleaned[label_col] = [str(t).strip().upper() for t in cleaned[label_col]]
        elif label_col and isinstance(cleaned.get(label_col), str):
            # Try to parse JSON-ish list strings
            val = cleaned[label_col].strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = json.loads(val)
                    cleaned[label_col] = [str(t).upper() for t in parsed]
                except json.JSONDecodeError:
                    pass

    # ── 6. Question Answering ───────────────────────────────────────────────
    elif task == "question_answering":
        # Answer must not be empty
        if label_col and cleaned.get(label_col) is not None:
            ans = str(cleaned[label_col]).strip()
            if not ans or ans.lower() in NULL_VALUES:
                errors.append("Empty answer field in QA row")
                cleaned[label_col] = None

    # ── 7. Summarization ────────────────────────────────────────────────────
    elif task == "summarization":
        if label_col and cleaned.get(label_col) is not None:
            summary = str(cleaned[label_col]).strip()
            source = str(cleaned.get(input_col, "")).strip()
            if len(summary) > len(source) * 0.9 and len(source) > 0:
                warnings.append("Summary is nearly as long as the source document; check column assignment.")
            if not summary:
                errors.append("Empty summary field")
                cleaned[label_col] = None

    # ── 8. Translation ──────────────────────────────────────────────────────
    elif task == "translation":
        if label_col and cleaned.get(label_col) is not None:
            trans = str(cleaned[label_col]).strip()
            if not trans:
                errors.append("Empty translation field")
                cleaned[label_col] = None
            # Same-text translation is suspicious
            src = str(cleaned.get(input_col, "")).strip()
            if src and src.lower() == trans.lower():
                warnings.append("Source and translation are identical — possible data error.")

    # ── 9. Instruction tuning / Chat ────────────────────────────────────────
    elif task in ("instruction_tuning", "chat"):
        if label_col and cleaned.get(label_col) is not None:
            resp = str(cleaned[label_col]).strip()
            if not resp or resp.lower() in NULL_VALUES:
                errors.append("Empty output/response in instruction-tuning row")
                cleaned[label_col] = None

    # ── 10. Grammar correction / simplification / style transfer ────────────
    elif task in ("grammar_correction", "text_simplification", "style_transfer"):
        if label_col and cleaned.get(label_col) is not None:
            out = str(cleaned[label_col]).strip()
            src = str(cleaned.get(input_col, "")).strip()
            if not out:
                errors.append(f"Empty target field for {task}")
                cleaned[label_col] = None
            elif src and src.lower() == out.lower():
                warnings.append("Input and output are identical — no transformation applied.")

    # ── 11. NLI ─────────────────────────────────────────────────────────────
    elif task == "natural_language_inference":
        if label_col and cleaned.get(label_col) is not None:
            val = _normalize_label_string(cleaned[label_col])
            cleaned[label_col] = val
            valid_nli = {"entailment", "contradiction", "neutral", "0", "1", "2"}
            if val not in valid_nli:
                warnings.append(f"Unexpected NLI label '{val}'; expected entailment/contradiction/neutral.")

    return cleaned, warnings, errors


# ─────────────────────────────────────────────────────────────────────────────
# Batch Cleaning & Auditing
# ─────────────────────────────────────────────────────────────────────────────

def _choose_expand_contractions(task: str) -> bool:
    """Contraction expansion helps classification/sentiment but hurts generation tasks."""
    return task in ("classification", "sentiment_analysis", "text_pair_classification",
                    "natural_language_inference", "multi_label_classification")


def _clean_and_audit_batch(
    batch: Dict[str, List],
    text_columns: List[str],
    label_columns: List[str],
    task_info: Dict[str, Any],
) -> Dict[str, List]:
    """
    HuggingFace-compatible batch map function.
    Adds __quality_score, __quality_category, __audit_issues, and per-fix counters.
    """
    task = task_info.get("task", "")
    label_col = task_info.get("label_column")
    expand_contractions = _choose_expand_contractions(task)

    # Prepare output accumulators
    out: Dict[str, List] = {col: [] for col in batch.keys()}
    quality_scores: List[float] = []
    quality_categories: List[str] = []
    audit_issues_list: List[str] = []
    mojibake_fixes: List[int] = []
    html_escapes: List[int] = []
    spaces_normalized: List[int] = []
    unicode_normalizations: List[int] = []
    pii_redactions: List[int] = []
    task_specific_updates: List[int] = []

    n = len(next(iter(batch.values())))

    for i in range(n):
        row = {col: batch[col][i] for col in batch}
        cleaned_row = row.copy()
        row_issues: List[str] = []

        r_mojibake = r_html = r_space = r_unicode = r_pii = r_task = 0

        # ── Stage 1: generic text cleaning + PII redaction ──────────────────
        for col in text_columns:
            val = row.get(col)

            if isinstance(val, str):
                # Generic clean
                cv, counts = clean_text_generic(val, expand_contractions=expand_contractions)
                r_mojibake += counts.get("mojibake_fixes", 0)
                r_html     += counts.get("html_escapes", 0)
                r_space    += counts.get("spaces_normalized", 0)
                r_unicode  += counts.get("unicode_normalizations", 0)

                # PII redaction
                cv, pii_found = redact_pii(cv)
                if pii_found:
                    r_pii += sum(pii_found.values())
                    row_issues.append(f"PII redacted: {pii_found}")

                cleaned_row[col] = cv
                if cv != val:
                    if counts.get("mojibake_fixes"):     row_issues.append("Mojibake fixed")
                    if counts.get("html_escapes"):        row_issues.append("HTML entities decoded")
                    if counts.get("spaces_normalized"):   row_issues.append("Whitespace normalized")
                    if counts.get("contractions_expanded"): row_issues.append("Contractions expanded")

            elif isinstance(val, list):
                # Lists of strings (e.g. token classification tokens)
                cleaned_list: List[Any] = []
                for item in val:
                    if isinstance(item, str):
                        ci, counts = clean_text_generic(item, expand_contractions=False)
                        r_mojibake += counts.get("mojibake_fixes", 0)
                        r_html     += counts.get("html_escapes", 0)
                        r_space    += counts.get("spaces_normalized", 0)
                        r_unicode  += counts.get("unicode_normalizations", 0)
                        cleaned_list.append(ci)
                    else:
                        cleaned_list.append(item)
                cleaned_row[col] = cleaned_list

        # ── Stage 2: task-specific cleaning ─────────────────────────────────
        cleaned_row, t_warnings, t_errors = clean_row_task_specific(cleaned_row, task_info)
        for col in batch:
            if cleaned_row.get(col) != row.get(col) and col == label_col:
                r_task += 1
        row_issues += [f"Warning: {w}" for w in t_warnings]
        row_issues += [f"Error: {e}" for e in t_errors]

        # ── Stage 3: quality score ───────────────────────────────────────────
        score, category = compute_row_quality(row, text_columns, label_columns, task_info)
        if t_errors:
            category = "CORRUPTED"
            score = min(score, 30.0)

        quality_scores.append(score)
        quality_categories.append(category)
        audit_issues_list.append(json.dumps(row_issues))
        mojibake_fixes.append(r_mojibake)
        html_escapes.append(r_html)
        spaces_normalized.append(r_space)
        unicode_normalizations.append(r_unicode)
        pii_redactions.append(r_pii)
        task_specific_updates.append(r_task)

        for col in batch:
            out[col].append(cleaned_row.get(col))

    out["__quality_score"]        = quality_scores
    out["__quality_category"]     = quality_categories
    out["__audit_issues"]         = audit_issues_list
    out["__mojibake_fixes"]       = mojibake_fixes
    out["__html_escapes"]         = html_escapes
    out["__spaces_normalized"]    = spaces_normalized
    out["__unicode_normalizations"] = unicode_normalizations
    out["__pii_redactions"]       = pii_redactions
    out["__task_specific_updates"] = task_specific_updates
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication — Exact + Near-Duplicate (N-gram Jaccard)
# ─────────────────────────────────────────────────────────────────────────────

def _text_signature(row: Dict[str, Any], text_cols: List[str]) -> str:
    return "||".join(str(row.get(c) or "").strip() for c in text_cols)


def _ngram_shingles(text: str, n: int = 3) -> Set[str]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return {text.lower()}
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def remove_duplicate_rows(
    dataset: Any,
    text_columns: List[str],
    fuzzy: bool = True,
    jaccard_threshold: float = 0.85,
) -> Tuple[Any, int]:
    """
    Stage 1: Exact hash dedup.
    Stage 2 (optional): Near-duplicate removal via n-gram Jaccard similarity.
    Returns (deduplicated_dataset, n_removed).
    """
    before = len(dataset)
    try:
        import pandas as pd

        cols_present = [c for c in text_columns if c in dataset.column_names]
        df = dataset.select_columns(cols_present).to_pandas() if cols_present else dataset.to_pandas()

        # ── Exact dedup ──
        dedup_df = df.drop_duplicates(subset=cols_present if cols_present else None)
        exact_removed = before - len(dedup_df)
        keep_indices = list(dedup_df.index)

        # ── Fuzzy dedup (n-gram Jaccard) ────────────────────────────────────
        fuzzy_removed = 0
        if fuzzy and cols_present and len(keep_indices) > 1:
            main_col = cols_present[0]
            texts = dedup_df[main_col].fillna("").astype(str).tolist()
            shingle_sets = [_ngram_shingles(t) for t in texts]
            keep_mask = [True] * len(keep_indices)

            for i in range(len(shingle_sets)):
                if not keep_mask[i]:
                    continue
                for j in range(i + 1, len(shingle_sets)):
                    if not keep_mask[j]:
                        continue
                    a, b = shingle_sets[i], shingle_sets[j]
                    union = len(a | b)
                    if union == 0:
                        continue
                    jaccard = len(a & b) / union
                    if jaccard >= jaccard_threshold:
                        keep_mask[j] = False
                        fuzzy_removed += 1

            keep_indices = [idx for idx, keep in zip(keep_indices, keep_mask) if keep]

        total_removed = before - len(keep_indices)
        if total_removed > 0:
            logger.info(
                f"Dedup: exact={exact_removed}, fuzzy={fuzzy_removed}, "
                f"total_removed={total_removed}"
            )
            dataset = dataset.select(keep_indices)

        return dataset, total_removed

    except Exception as e:
        logger.warning(f"Dedup skipped: {e}")
        return dataset, 0


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Detection
# ─────────────────────────────────────────────────────────────────────────────

def check_target_leakage(
    dataset: Any,
    text_cols: List[str],
    label_col: Optional[str],
    task_name: str,
) -> List[str]:
    """
    Detect verbatim label presence inside input texts.
    Applies only to classification-type tasks.
    """
    warnings: List[str] = []
    if not label_col or not text_cols:
        return warnings
    if task_name not in (
        "classification", "sentiment_analysis", "text_pair_classification",
        "natural_language_inference", "multi_label_classification",
    ):
        return warnings

    sample_size = min(len(dataset), 500)
    leakage_count = 0
    for i in range(sample_size):
        row = dataset[i]
        label_val = str(row.get(label_col) or "").strip().lower()
        if not label_val or len(label_val) < 3:
            continue
        for tc in text_cols:
            text_val = str(row.get(tc) or "").strip().lower()
            # Substring check + word-boundary check
            if re.search(r"\b" + re.escape(label_val) + r"\b", text_val):
                leakage_count += 1
                break

    rate = leakage_count / sample_size if sample_size > 0 else 0
    if rate > 0.15:
        warnings.append(
            f"Target leakage ({rate:.1%}): label '{label_col}' appears verbatim "
            f"inside input text in {leakage_count}/{sample_size} sampled rows. "
            f"Strip labels before training to avoid leakage."
        )
    return warnings


def check_train_test_leakage(dataset_dict: Any) -> List[str]:
    """Exact-text signature overlap between train split and other splits."""
    warnings: List[str] = []
    if not isinstance(dataset_dict, dict) or len(dataset_dict) <= 1:
        return warnings

    train_ds = dataset_dict.get("train")
    if not train_ds or len(train_ds) == 0:
        return warnings

    cols = (
        [c for c in train_ds.column_names
         if hasattr(train_ds, "features") and train_ds.features[c].dtype == "string"]
        or [train_ds.column_names[0]]
    )

    train_sigs: Set[str] = set()
    for i in range(min(len(train_ds), 10_000)):
        sig = _text_signature(train_ds[i], cols)
        if sig:
            train_sigs.add(sig)

    for split_name, ds in dataset_dict.items():
        if split_name == "train" or len(ds) == 0:
            continue
        overlap = 0
        sample = min(len(ds), 2000)
        for i in range(sample):
            if _text_signature(ds[i], cols) in train_sigs:
                overlap += 1
        rate = overlap / sample if sample > 0 else 0
        if rate > 0.01:
            warnings.append(
                f"Split leakage ({rate:.1%}): '{split_name}' has {overlap}/{sample} rows "
                f"matching train exactly. Deduplicate across splits before training."
            )
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Report Compilation
# ─────────────────────────────────────────────────────────────────────────────

def compile_reports(
    dataset: Any,
    split_name: Optional[str] = None,
) -> Tuple[Dict, Dict, List[Dict], Dict]:
    """Aggregate per-batch audit columns into summary dicts."""
    scores       = dataset["__quality_score"]
    categories   = dataset["__quality_category"]
    issues_list  = dataset["__audit_issues"]

    total = len(dataset)
    cat_counts = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    avg_score = sum(scores) / total if total else 100.0
    total_warnings = total_errors = 0
    for issues_str in issues_list:
        for issue in json.loads(issues_str):
            if issue.startswith("Warning:"):
                total_warnings += 1
            elif issue.startswith("Error:"):
                total_errors += 1

    row_audit = [
        {
            "row_index": idx,
            "quality_score": scores[idx],
            "category": categories[idx],
            "issues": json.loads(issues_list[idx]),
        }
        for idx in range(min(total, 1000))
    ]

    cleaning_report = {
        "total_rows_processed": total,
        "total_rows_cleaned": sum(1 for s in issues_list if json.loads(s)),
        "mojibake_fixes":           sum(dataset["__mojibake_fixes"]),
        "html_escapes":             sum(dataset["__html_escapes"]),
        "spaces_normalized":        sum(dataset["__spaces_normalized"]),
        "unicode_normalizations":   sum(dataset["__unicode_normalizations"]),
        "pii_redactions":           sum(dataset["__pii_redactions"]),
        "task_specific_updates":    sum(dataset["__task_specific_updates"]),
    }
    quality_report = {
        "average_quality_score": round(avg_score, 2),
        "category_counts": cat_counts,
        "category_ratios": {k: round(v / total, 4) for k, v in cat_counts.items()} if total else {},
    }
    issue_summary = {
        "total_warnings": total_warnings,
        "total_errors":   total_errors,
        "corrupted_rows_count": cat_counts.get("CORRUPTED", 0),
    }
    return cleaning_report, quality_report, row_audit, issue_summary


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    quality_report: Dict,
    issue_summary: Dict,
    task_info: Dict,
) -> List[str]:
    recs: List[str] = []
    avg  = quality_report.get("average_quality_score", 100.0)
    cats = quality_report.get("category_counts", {})
    task = task_info.get("task", "unknown")

    if avg < 60:
        recs.append(
            f"[CRITICAL] Average quality score is {avg:.1f}/100. "
            "Dataset needs significant cleanup before training."
        )
    elif avg < 80:
        recs.append(
            f"[WARNING] Average quality score is {avg:.1f}/100. "
            "Review FIXABLE and LOW_QUALITY rows."
        )

    corrupted = issue_summary.get("corrupted_rows_removed", 0)
    if corrupted:
        recs.append(f"Removed {corrupted} CORRUPTED rows (missing inputs/labels or token-tag length mismatch).")

    dupes = issue_summary.get("duplicates_removed", 0)
    if dupes:
        recs.append(f"Removed {dupes} duplicate rows (exact + near-duplicate at Jaccard ≥ 0.85).")

    pii = issue_summary.get("pii_redactions_total", 0)
    if pii:
        recs.append(
            f"Redacted {pii} PII instances (emails, phones, SSNs, credit cards, etc.) "
            "with typed placeholders. Verify redaction before sharing dataset."
        )

    low_q = cats.get("LOW_QUALITY", 0)
    if low_q:
        recs.append(
            f"Found {low_q} LOW_QUALITY rows. Inspect for keyboard mashes, "
            "excessive noise, or very short texts."
        )

    if issue_summary.get("target_leakage_detected"):
        recs.append(
            "[WARNING] Target leakage detected. Label values appear verbatim in input text. "
            "Strip or anonymize labels in input columns before training."
        )

    if issue_summary.get("train_test_leakage_detected"):
        recs.append(
            "[WARNING] Train-test split leakage detected. "
            "Identical rows exist across splits — deduplicate splits before evaluation."
        )

    # Task-specific guidance
    if task == "summarization":
        recs.append("For summarization: verify source/summary length ratio is >3:1. "
                    "Short summaries relative to source improve abstractive quality.")
    elif task == "token_classification":
        recs.append("For token classification: ensure token list and label list lengths match for every row.")
    elif task == "translation":
        recs.append("For translation: check that source/target are in different languages and not identical.")
    elif task == "instruction_tuning":
        recs.append("For instruction tuning: ensure every (instruction, output) pair is non-empty and diverse.")

    if not recs:
        recs.append("Dataset quality is high. Ready for training.")

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    dataset: Any,
    cfg: Optional[DictConfig] = None,
    task_info: Optional[Dict[str, Any]] = None,
):
    """
    Universal Preprocessing Entry Point.

    Args:
        dataset:   HuggingFace Dataset or DatasetDict
        cfg:       Hydra DictConfig (optional if task_info is provided)
        task_info: Dict with keys: task, input_column, label_column,
                   text_columns, label_columns, detected_columns, etc.

    Returns:
        (cleaned_dataset, cleaning_report, quality_report,
         row_level_audit, issue_summary, warnings, recommendations)
    """
    from datasets import Dataset, DatasetDict

    # ── Resolve task_info ────────────────────────────────────────────────────
    if task_info is None:
        task_name = (
            cfg.task.name
            if cfg and hasattr(cfg, "task") and hasattr(cfg.task, "name")
            else "classification"
        )
        text_col  = cfg.dataset.text_column  if cfg and hasattr(cfg.dataset, "text_column")  else None
        label_col = cfg.dataset.label_column if cfg and hasattr(cfg.dataset, "label_column") else None
        task_info = {
            "task":           task_name,
            "input_column":   text_col,
            "output_column":  label_col,
            "label_column":   label_col,
            "text_columns":   [text_col]  if text_col  else [],
            "label_columns":  [label_col] if label_col else [],
            "detected_columns": {"text": [text_col] if text_col else [],
                                 "label": [label_col] if label_col else []},
        }

    task_name   = task_info.get("task", "classification")
    detected    = task_info.get("detected_columns", {})
    text_cols   = list(set(detected.get("text",  []) + task_info.get("text_columns",  [])))
    label_cols  = list(set(detected.get("label", []) + task_info.get("label_columns", [])))
    input_col   = task_info.get("input_column")
    label_col   = task_info.get("label_column")
    if input_col and input_col not in text_cols:
        text_cols.append(input_col)
    if label_col and label_col not in label_cols:
        label_cols.append(label_col)

    logger.info(f"[Preprocessor] task='{task_name}' | text_cols={text_cols} | label_cols={label_cols}")

    # ── Helper: process a single split ───────────────────────────────────────
    def _process_split(ds: Any, sname: str) -> Tuple[Any, Dict]:
        logger.info(f"  Split '{sname}': {len(ds)} rows")

        # Stage 1-3: clean + audit
        mapped = ds.map(
            lambda batch: _clean_and_audit_batch(batch, text_cols, label_cols, task_info),
            batched=True,
            desc=f"Cleaning '{sname}'",
        )

        cr, qr, ra, iss = compile_reports(mapped, sname)

        # Stage 4: filter corrupted
        before = len(mapped)
        cleaned = mapped.filter(
            lambda x: x["__quality_category"] != "CORRUPTED",
            desc=f"Filtering corrupted rows '{sname}'",
        )
        iss["corrupted_rows_removed"] = before - len(cleaned)

        # Stage 5: deduplication
        cleaned, n_dupes = remove_duplicate_rows(cleaned, text_cols)
        iss["duplicates_removed"] = n_dupes
        iss["pii_redactions_total"] = sum(mapped["__pii_redactions"])

        # Stage 6: target leakage
        tl_warns = check_target_leakage(cleaned, text_cols, label_col, task_name)
        iss["target_leakage_detected"] = len(tl_warns) > 0

        # Collect row-level warnings
        row_warns: Set[str] = set()
        for issues_str in mapped["__audit_issues"]:
            for issue in json.loads(issues_str):
                if issue.startswith("Warning:"):
                    row_warns.add(issue[len("Warning: "):])

        split_warns = tl_warns + list(row_warns)

        # Remove helper columns
        final = cleaned.remove_columns([c for c in cleaned.column_names if c.startswith("__")])

        rep = {
            "cleaning_report": cr,
            "quality_report":  qr,
            "row_level_audit": ra,
            "issue_summary":   iss,
            "warnings":        split_warns,
        }
        return final, rep

    # ── DatasetDict branch ───────────────────────────────────────────────────
    if isinstance(dataset, DatasetDict):
        cleaned_splits: Dict[str, Any] = {}
        split_reports:  Dict[str, Dict] = {}

        for sname, ds in dataset.items():
            if len(ds) == 0:
                cleaned_splits[sname] = ds
                continue
            cleaned_splits[sname], split_reports[sname] = _process_split(ds, sname)

        combined = DatasetDict(cleaned_splits)

        # Cross-split leakage
        split_leakage_warns = check_train_test_leakage(combined)

        # Aggregate metrics
        total_rows = total_cleaned = mo = he = sp = un = pii_tot = tu = 0
        total_warns = total_errs = total_corrupt = total_corrupt_rm = total_dupes = 0
        target_leak = False
        cat_counts  = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0}
        score_sum   = 0.0
        row_audit:  Dict[str, List] = {}
        warnings:   List[str] = []

        for sn, rep in split_reports.items():
            cr  = rep["cleaning_report"]
            qr  = rep["quality_report"]
            iss = rep["issue_summary"]

            total_rows    += cr["total_rows_processed"]
            total_cleaned += cr["total_rows_cleaned"]
            mo   += cr["mojibake_fixes"]
            he   += cr["html_escapes"]
            sp   += cr["spaces_normalized"]
            un   += cr["unicode_normalizations"]
            pii_tot += cr.get("pii_redactions", 0)
            tu   += cr["task_specific_updates"]

            total_warns   += iss["total_warnings"]
            total_errs    += iss["total_errors"]
            total_corrupt += iss["corrupted_rows_count"]
            total_corrupt_rm += iss["corrupted_rows_removed"]
            total_dupes   += iss["duplicates_removed"]
            if iss["target_leakage_detected"]:
                target_leak = True

            for cat, cnt in qr["category_counts"].items():
                cat_counts[cat] = cat_counts.get(cat, 0) + cnt
            score_sum += qr["average_quality_score"] * cr["total_rows_processed"]

            row_audit[sn] = rep["row_level_audit"]
            warnings += [f"[{sn}] {w}" for w in rep["warnings"]]

        warnings += split_leakage_warns

        avg_q = score_sum / total_rows if total_rows else 100.0

        cleaning_report = {
            "total_rows_processed": total_rows,
            "total_rows_cleaned":   total_cleaned,
            "mojibake_fixes":       mo,
            "html_escapes":         he,
            "spaces_normalized":    sp,
            "unicode_normalizations": un,
            "pii_redactions":       pii_tot,
            "task_specific_updates": tu,
        }
        quality_report = {
            "average_quality_score": round(avg_q, 2),
            "category_counts": cat_counts,
            "category_ratios": {k: round(v / total_rows, 4) for k, v in cat_counts.items()} if total_rows else {},
        }
        issue_summary = {
            "total_warnings":           total_warns + len(split_leakage_warns),
            "total_errors":             total_errs,
            "corrupted_rows_count":     total_corrupt,
            "corrupted_rows_removed":   total_corrupt_rm,
            "duplicates_removed":       total_dupes,
            "pii_redactions_total":     pii_tot,
            "target_leakage_detected":  target_leak,
            "train_test_leakage_detected": len(split_leakage_warns) > 0,
        }
        recommendations = generate_recommendations(quality_report, issue_summary, task_info)

        return (combined, cleaning_report, quality_report, row_audit,
                issue_summary, warnings, recommendations)

    # ── Single Dataset branch ────────────────────────────────────────────────
    if len(dataset) == 0:
        return dataset, {}, {}, [], {}, [], []

    final_ds, rep = _process_split(dataset, "dataset")

    cr  = rep["cleaning_report"]
    qr  = rep["quality_report"]
    ra  = rep["row_level_audit"]
    iss = rep["issue_summary"]
    warnings = rep["warnings"]
    recommendations = generate_recommendations(qr, iss, task_info)

    return (final_ds, cr, qr, ra, iss, warnings, recommendations)