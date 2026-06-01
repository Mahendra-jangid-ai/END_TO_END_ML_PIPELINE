"""
Universal NLP Dataset Preprocessing Engine
Automatically detects the NLP task type, profiles the schema, applies generic cleaning,
audits row-level quality (0-100), performs task-specific sanitization and validation,
and executes advanced checks for target leakage and split leakage.
"""
from __future__ import annotations

import html
import re
import unicodedata
import json
import difflib
from typing import Any, Dict, List, Optional, Tuple, Set
from omegaconf import DictConfig

from src.utils.common import get_logger
from src.data.task_detector import TaskDetector

logger = get_logger(__name__)

# Constants
NULL_VALUES = {
    "", "na", "n/a", "null", "none", "nan", "missing"
}

# Regex Patterns for Noise Detection
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_PATTERN = re.compile(r"\*\*|__|^#+\s|\[.*?\]\(.*?\)", re.MULTILINE)
RT_PATTERN = re.compile(r"\bRT\b")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200d\uFEFF\u00AD]")
OCR_CORRUPTION_PATTERN = re.compile(r"\b[a-zA-Z]+[0-9]+[a-zA-Z]+\b")
REPEATED_PUNCT_PATTERN = re.compile(r"[.,;:!?\-+=_*]{3,}")
REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-Z])\1{3,}")


def heal_mojibake(text: str) -> str:
    """Heal common Mojibake encoding artifacts."""
    if not isinstance(text, str):
        return text
    # Try dynamic healing first
    try:
        if any(c in text for c in "âãåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"):
            re_encoded = text.encode('cp1252').decode('utf-8')
            return re_encoded
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
        
    # Static mappings for remaining issues
    mojibake_map = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¢": "•",
        "â€”": "—",
        "â€“": "–",
        "â€": "",
        "Ã©": "é",
        "Ã¡": "á",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
        "Ã­": "í",
        "Ã": "à",
    }
    for bad, good in mojibake_map.items():
        text = text.replace(bad, good)
    return text


def clean_text_generic(text: Any) -> tuple[Any, dict[str, int]]:
    """
    Applies generic text sanitization.
    Returns: (cleaned_text, change_counts)
    """
    if not isinstance(text, str):
        return text, {}
        
    counts = {
        "mojibake_fixes": 0,
        "html_escapes": 0,
        "spaces_normalized": 0,
        "unicode_normalizations": 0
    }
    
    orig = text
    
    # 1. HTML entity decoding
    unescaped = html.unescape(text)
    if unescaped != text:
        counts["html_escapes"] += 1
        text = unescaped
        
    # 2. Heal mojibake
    healed = heal_mojibake(text)
    if healed != text:
        counts["mojibake_fixes"] += 1
        text = healed
        
    # 3. Unicode normalization
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        counts["unicode_normalizations"] += 1
        text = normalized
        
    # 4. Remove control characters (except tab and newline)
    cleaned_chars = []
    for ch in text:
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"):
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)
    
    # 5. Normalize smart quotes and dashes to standard ASCII equivalents
    quote_map = {
        '“': '"', '”': '"', '‟': '"', '″': '"', '′': "'", '‘': "'", '’': "'"
    }
    for q_src, q_tgt in quote_map.items():
        text = text.replace(q_src, q_tgt)
        
    dash_map = {
        '–': '-', '—': '-', '―': '-', '−': '-'
    }
    for d_src, d_tgt in dash_map.items():
        text = text.replace(d_src, d_tgt)
        
    # 6. Normalize tabs and spacing
    text = text.replace("\t", " ")
    
    # Replace multiple spaces with single space
    space_norm = re.sub(r"[ \t]+", " ", text)
    if space_norm != text:
        counts["spaces_normalized"] += 1
        text = space_norm
        
    # 7. Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Trim whitespace
    text = text.strip()
    
    return text, counts


def analyze_noise(text: str) -> dict[str, float]:
    """Detect noise sources and anomalies in text."""
    if not isinstance(text, str):
        return {}
        
    urls = URL_PATTERN.findall(text)
    emails = EMAIL_PATTERN.findall(text)
    phones = PHONE_PATTERN.findall(text)
    html_tags = HTML_TAG_PATTERN.findall(text)
    markdown = MARKDOWN_PATTERN.findall(text)
    rt = RT_PATTERN.findall(text)
    mentions = MENTION_PATTERN.findall(text)
    hashtags = HASHTAG_PATTERN.findall(text)
    invisible = INVISIBLE_UNICODE_PATTERN.findall(text)
    ocr = OCR_CORRUPTION_PATTERN.findall(text)
    
    # Keyboard mash heuristic: consonant ratio and repetition check
    words = text.split()
    mash_count = 0
    for w in words:
        if len(w) >= 5:
            vowels = set("aeiouAEIOU")
            consonants = sum(1 for c in w if c.isalpha() and c not in vowels)
            letters = sum(1 for c in w if c.isalpha())
            if letters > 0 and consonants / letters >= 0.80:
                mash_count += 1
            elif any(w.count(w[i:i+3]) >= 3 for i in range(len(w)-3)):
                mash_count += 1
                
    # Broken tokenization (e.g. hello.world)
    broken_tok = re.findall(r"\b[a-zA-Z]+[.,;:!?][a-zA-Z]+\b", text)
    
    # Repeated punctuation/characters
    rep_punct = REPEATED_PUNCT_PATTERN.findall(text)
    rep_chars = REPEATED_CHAR_PATTERN.findall(text)
    
    # Low information text
    is_low_info = len(text.strip()) < 5 or len(words) < 2
    
    return {
        "urls": len(urls),
        "emails": len(emails),
        "phones": len(phones),
        "html_tags": len(html_tags),
        "markdown": len(markdown),
        "rt": len(rt),
        "mentions": len(mentions),
        "hashtags": len(hashtags),
        "invisible": len(invisible),
        "ocr": len(ocr),
        "keyboard_mash": mash_count,
        "broken_tokenization": len(broken_tok),
        "repeated_punctuation": len(rep_punct),
        "repeated_chars": len(rep_chars),
        "low_information": 1.0 if is_low_info else 0.0
    }


def compute_row_quality(row: dict[str, Any], text_cols: list[str], label_cols: list[str], task_info: dict[str, Any]) -> tuple[float, str]:
    """Calculate quality score (0-100) and category for a single row."""
    input_col = task_info.get("input_column")
    label_col = task_info.get("label_column")
    task = task_info.get("task")
    
    if not input_col or row.get(input_col) is None:
        return 0.0, "CORRUPTED"
        
    score = 100.0
    
    # Check text noise
    for col in text_cols:
        val = row.get(col)
        if not isinstance(val, str):
            continue
            
        noise = analyze_noise(val)
        if not noise:
            continue
            
        score -= noise.get("html_tags", 0) * 15
        score -= noise.get("ocr", 0) * 20
        score -= noise.get("keyboard_mash", 0) * 30
        score -= noise.get("broken_tokenization", 0) * 10
        score -= noise.get("repeated_punctuation", 0) * 5
        score -= noise.get("repeated_chars", 0) * 5
        score -= noise.get("invisible", 0) * 5
        score -= noise.get("urls", 0) * 10
        score -= noise.get("emails", 0) * 10
        score -= noise.get("phones", 0) * 10
        
        if noise.get("low_information", 0.0) > 0.0:
            score -= 40
            
    score = max(0.0, min(100.0, score))
    
    # Determine base category
    if score >= 90:
        category = "CLEAN"
    elif score >= 70:
        category = "FIXABLE"
    elif score >= 40:
        category = "LOW_QUALITY"
    else:
        category = "CORRUPTED"
        
    # Check if target label is missing for label-dependent tasks
    if label_col and row.get(label_col) is None:
        if task in ("classification", "multi_label_classification", "regression", "token_classification", "ner"):
            category = "CORRUPTED"
            score = min(score, 30.0)
            
    return score, category


def _get_string_similarity(s1: str, s2: str) -> float:
    """Standard SequenceMatcher similarity score."""
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def clean_row_task_specific(row: dict[str, Any], task_info: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Apply cleaning rules and validations tailored to the NLP task."""
    task = task_info.get("task")
    input_col = task_info.get("input_column")
    label_col = task_info.get("label_column")
    
    cleaned = row.copy()
    warnings = []
    errors = []
    
    # Classification / Single Label
    if task in ("classification", "sentiment_analysis"):
        if label_col in cleaned and cleaned[label_col] is not None:
            val = cleaned[label_col]
            if isinstance(val, str):
                cleaned[label_col] = val.strip().lower()
            elif isinstance(val, bool):
                cleaned[label_col] = str(val).lower()
                
    # Multi Label Classification
    elif task == "multi_label_classification":
        if label_col in cleaned and cleaned[label_col] is not None:
            val = cleaned[label_col]
            if isinstance(val, str):
                val_s = val.strip()
                if val_s.startswith("[") and val_s.endswith("]"):
                    try:
                        parsed = json.loads(val_s)
                        if isinstance(parsed, list):
                            cleaned[label_col] = [str(x).strip() for x in parsed]
                        else:
                            cleaned[label_col] = [str(parsed).strip()]
                    except ValueError:
                        cleaned[label_col] = [x.strip() for x in val_s[1:-1].split(",") if x.strip()]
                elif "|" in val_s:
                    cleaned[label_col] = [x.strip() for x in val_s.split("|") if x.strip()]
                elif "," in val_s:
                    cleaned[label_col] = [x.strip() for x in val_s.split(",") if x.strip()]
                else:
                    cleaned[label_col] = [val_s]
            elif isinstance(val, (list, tuple)):
                cleaned[label_col] = [str(x).strip() for x in val]
            else:
                cleaned[label_col] = [str(val).strip()]
                
    # Regression
    elif task == "regression":
        if label_col in cleaned and cleaned[label_col] is not None:
            val = cleaned[label_col]
            try:
                cleaned[label_col] = float(val)
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value '{val}' in regression target")
                cleaned[label_col] = None
                
    # Token Classification / NER / POS / Chunking
    elif task in ("token_classification", "ner", "pos_tagging", "chunking"):
        tokens_col = input_col
        tags_col = label_col
        if tokens_col in cleaned and tags_col in cleaned:
            tokens_val = cleaned[tokens_col]
            tags_val = cleaned[tags_col]
            
            def parse_sequence(val):
                if isinstance(val, (list, tuple)):
                    return list(val)
                if isinstance(val, str):
                    val_s = val.strip()
                    if val_s.startswith("[") and val_s.endswith("]"):
                        try:
                            parsed = json.loads(val_s)
                            if isinstance(parsed, list):
                                return parsed
                        except ValueError:
                            pass
                    return val_s.split()
                return []
                
            tokens = parse_sequence(tokens_val)
            tags = parse_sequence(tags_val)
            
            if tokens and tags:
                if len(tokens) != len(tags):
                    errors.append(f"Token/Tag length mismatch: len(tokens)={len(tokens)} vs len(tags)={len(tags)}")
                cleaned[tokens_col] = tokens
                cleaned[tags_col] = tags
            else:
                errors.append("Empty token or tag sequence")
                
    # Question Answering
    elif task in ("question_answering", "extractive_qa", "abstractive_qa"):
        q_col = input_col
        a_col = task_info.get("output_column") or label_col
        c_col = task_info.get("context_column")
        
        q_val = cleaned.get(q_col)
        a_val = cleaned.get(a_col)
        c_val = cleaned.get(c_col)
        
        if not q_val or not a_val:
            errors.append("Missing question or answer field")
        else:
            ans_str = ""
            if isinstance(a_val, dict) and "text" in a_val:
                texts = a_val["text"]
                ans_str = str(texts[0]) if isinstance(texts, list) and len(texts) > 0 else str(texts)
            elif isinstance(a_val, list) and len(a_val) > 0:
                ans_str = str(a_val[0])
            else:
                ans_str = str(a_val)
                
            if c_col and c_val:
                c_str = str(c_val)
                if task in ("question_answering", "extractive_qa") and ans_str.lower() not in c_str.lower():
                    warnings.append(f"Answer '{ans_str}' not found in context")
                    
    # Summarization / Title Generation / Text Simplification
    elif task in ("summarization", "title_generation", "text_simplification"):
        src_col = input_col
        tgt_col = task_info.get("output_column") or label_col
        
        src_val = cleaned.get(src_col)
        tgt_val = cleaned.get(tgt_col)
        
        if src_val and tgt_val:
            src_len = len(str(src_val))
            tgt_len = len(str(tgt_val))
            if tgt_len >= src_len:
                warnings.append(f"Target text ({tgt_len} chars) is longer than source text ({src_len} chars)")
            if task == "title_generation" and tgt_len > src_len * 0.4:
                warnings.append(f"Title is too long relative to source text (title={tgt_len}, source={src_len})")
                
    # Translation
    elif task == "translation":
        src_col = input_col
        tgt_col = task_info.get("output_column") or label_col
        
        src_val = cleaned.get(src_col)
        tgt_val = cleaned.get(tgt_col)
        
        if src_val and tgt_val:
            if str(src_val).strip().lower() == str(tgt_val).strip().lower():
                errors.append("Source and target translation sentences are identical")
                
    # Instruction Tuning
    elif task == "instruction_tuning":
        inst_col = task_info.get("instruction_column") or input_col
        out_col = task_info.get("output_column") or label_col
        
        inst_val = cleaned.get(inst_col)
        out_val = cleaned.get(out_col)
        
        if inst_val and out_val:
            inst_str = str(inst_val).strip().lower()
            out_str = str(out_val).strip().lower()
            if out_str in inst_str and len(out_str) > 10:
                warnings.append("Instruction contains the exact output text (leakage)")
                
    # Chat / Conversational AI
    elif task in ("chat", "conversational_ai", "dialogue"):
        conv_col = input_col
        conv_val = cleaned.get(conv_col)
        if isinstance(conv_val, list):
            last_role = None
            for idx, turn in enumerate(conv_val):
                if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                    errors.append(f"Broken chat structure at turn {idx}")
                    continue
                role = turn["role"]
                if role not in ("system", "user", "assistant", "bot"):
                    warnings.append(f"Unknown chat role '{role}' at turn {idx}")
                if last_role and role == last_role and role not in ("system",):
                    warnings.append(f"Consecutive duplicate role '{role}' at turn {idx}")
                last_role = role
        else:
            warnings.append(f"Chat column '{conv_col}' is not a list")

    # Retrieval / Ranking / Reranking
    elif task in ("retrieval", "ranking", "reranking"):
        q_col = input_col
        doc_col = task_info.get("output_column") or label_col
        
        q_val = cleaned.get(q_col)
        doc_val = cleaned.get(doc_col)
        
        if not q_val or not doc_val:
            errors.append("Missing query or document text")

    # Text Similarity / Semantic Similarity
    elif task in ("text_similarity", "semantic_similarity"):
        t1_col = input_col
        other_texts = [c for c in task_info.get("detected_columns", {}).get("text", []) if c != t1_col]
        t2_col = other_texts[0] if other_texts else (task_info.get("output_column") or label_col)
        score_col = label_col if label_col != t2_col else None
        
        t1_val = cleaned.get(t1_col)
        t2_val = cleaned.get(t2_col)
        score_val = cleaned.get(score_col) if score_col else None
        
        if not t1_val or not t2_val:
            errors.append("Missing text pair for similarity")
        if score_val is not None:
            try:
                cleaned[score_col] = float(score_val)
            except (ValueError, TypeError):
                errors.append("Similarity score is not numeric")

    # Information Extraction
    elif task in ("information_extraction", "entity_extraction", "relation_extraction"):
        out_col = task_info.get("output_column") or label_col
        out_val = cleaned.get(out_col)
        if isinstance(out_val, str):
            out_s = out_val.strip()
            if (out_s.startswith("{") and out_s.endswith("}")) or (out_s.startswith("[") and out_s.endswith("]")):
                try:
                    cleaned[out_col] = json.loads(out_s)
                except ValueError:
                    warnings.append("Failed to parse output string as JSON")

    # NLI / Text Pair Classification
    elif task in ("natural_language_inference", "text_pair_classification"):
        t1_col = input_col
        other_texts = [c for c in task_info.get("detected_columns", {}).get("text", []) if c != t1_col]
        t2_col = other_texts[0] if other_texts else "hypothesis"
        l_col = label_col
        
        t1_val = cleaned.get(t1_col)
        t2_val = cleaned.get(t2_col)
        l_val = cleaned.get(l_col)
        
        if not t1_val or not t2_val:
            errors.append(f"Missing premise/hypothesis pair")
        if task == "natural_language_inference" and l_val is not None:
            val_s = str(l_val).strip().lower()
            nli_map = {
                "entailment": "entailment", "0": "entailment",
                "neutral": "neutral", "1": "neutral",
                "contradiction": "contradiction", "contradictory": "contradiction", "2": "contradiction"
            }
            if val_s in nli_map:
                cleaned[l_col] = nli_map[val_s]
            else:
                warnings.append(f"Unrecognized NLI label '{l_val}'")

    # Grammar Correction
    elif task == "grammar_correction":
        src_col = input_col
        tgt_col = task_info.get("output_column") or label_col
        
        src_val = cleaned.get(src_col)
        tgt_val = cleaned.get(tgt_col)
        
        if src_val and tgt_val:
            sim = _get_string_similarity(str(src_val), str(tgt_val))
            if sim < 0.65:
                warnings.append(f"Low semantic similarity ({sim:.2f}) for grammar correction")
            if sim == 1.0:
                warnings.append("Source and corrected sentences are identical")

    # Style Transfer / Paraphrase Generation
    elif task in ("style_transfer", "paraphrase_generation"):
        src_col = input_col
        tgt_col = task_info.get("output_column") or label_col
        
        src_val = cleaned.get(src_col)
        tgt_val = cleaned.get(tgt_col)
        
        if src_val and tgt_val:
            sim = _get_string_similarity(str(src_val), str(tgt_val))
            if sim < 0.35:
                warnings.append(f"Low semantic similarity ({sim:.2f}) - style transfer drift")

    # Keyword Generation
    elif task == "keyword_generation":
        kw_col = task_info.get("output_column") or label_col
        kw_val = cleaned.get(kw_col)
        if kw_val is not None:
            if isinstance(kw_val, str):
                cleaned[kw_col] = [x.strip() for x in kw_val.split(",") if x.strip()]
            elif isinstance(kw_val, list):
                cleaned[kw_col] = [str(x).strip() for x in kw_val]

    return cleaned, warnings, errors


def check_target_leakage(dataset, text_cols: list[str], label_col: str | None, task_name: str) -> list[str]:
    """Scan sample of dataset for target label leakage in input text fields."""
    warnings = []
    if not label_col or not text_cols:
        return warnings
        
    sample_size = min(len(dataset), 200)
    leakage_count = 0
    
    for i in range(sample_size):
        row = dataset[i]
        label_val = str(row.get(label_col) or "").strip().lower()
        if not label_val or len(label_val) < 3:
            continue
            
        for text_col in text_cols:
            text_val = str(row.get(text_col) or "").strip().lower()
            if label_val in text_val and task_name in ("classification", "multi_label_classification", "natural_language_inference"):
                leakage_count += 1
                break
                
    if sample_size > 0 and (leakage_count / sample_size) > 0.15:
        warnings.append(f"Verbatim target leakage detected: label column '{label_col}' is present inside input texts in {leakage_count}/{sample_size} checked rows.")
        
    return warnings


def check_train_test_leakage(dataset_dict) -> list[str]:
    """Scan across dataset splits to check for identical input text leakage."""
    warnings = []
    if not isinstance(dataset_dict, dict) or len(dataset_dict) <= 1:
        return warnings
        
    train_split = dataset_dict.get("train")
    if not train_split or len(train_split) == 0:
        return warnings
        
    columns = train_split.column_names
    # Try to identify main text fields
    text_cols = [c for c in columns if train_split.features[c].dtype == "string"] if hasattr(train_split, "features") else []
    if not text_cols:
        text_cols = [columns[0]]
        
    train_signatures = set()
    sample_train_size = min(len(train_split), 5000)
    for i in range(sample_train_size):
        row = train_split[i]
        sig = "||".join(str(row.get(c) or "").strip() for c in text_cols)
        if sig:
            train_signatures.add(sig)
            
    for split_name, ds in dataset_dict.items():
        if split_name == "train" or len(ds) == 0:
            continue
            
        overlap_count = 0
        sample_test_size = min(len(ds), 1000)
        for i in range(sample_test_size):
            row = ds[i]
            sig = "||".join(str(row.get(c) or "").strip() for c in text_cols)
            if sig in train_signatures:
                overlap_count += 1
                
        if sample_test_size > 0 and (overlap_count / sample_test_size) > 0.01:
            warnings.append(f"Train-Test leakage: Split '{split_name}' contains {overlap_count}/{sample_test_size} rows matching train exactly.")
            
    return warnings


def remove_duplicate_rows(dataset, text_columns: list[str]) -> tuple[Any, int]:
    """Remove duplicate rows from dataset based on text columns."""
    before = len(dataset)
    try:
        import pandas as pd
        if text_columns:
            subset = list(text_columns)
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
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
            dataset = dataset.select(unique_indices)
        return dataset, removed
    except Exception as e:
        logger.warning(f"Duplicate removal skipped: {e}")
        return dataset, 0


def _clean_and_audit_batch(batch, text_columns, label_columns, task_info):
    """Batch mapping cleaning and auditing function."""
    task = task_info.get("task")
    label_col = task_info.get("label_column")
    
    cleaned_batch = {col: [] for col in batch.keys()}
    
    quality_scores = []
    quality_categories = []
    audit_issues_list = []
    mojibake_fixes = []
    html_escapes = []
    spaces_normalized = []
    unicode_normalizations = []
    task_specific_updates = []
    
    n_rows = len(next(iter(batch.values())))
    
    for i in range(n_rows):
        row = {col: batch[col][i] for col in batch.keys()}
        
        row_mojibake = 0
        row_html = 0
        row_space = 0
        row_unicode = 0
        row_task = 0
        row_issues = []
        
        cleaned_row = row.copy()
        for col in text_columns:
            val = row.get(col)
            if isinstance(val, str):
                cleaned_val, counts = clean_text_generic(val)
                cleaned_row[col] = cleaned_val
                
                row_mojibake += counts.get("mojibake_fixes", 0)
                row_html += counts.get("html_escapes", 0)
                row_space += counts.get("spaces_normalized", 0)
                row_unicode += counts.get("unicode_normalizations", 0)
                
                if cleaned_val != val:
                    if counts.get("mojibake_fixes", 0) > 0:
                        row_issues.append("Mojibake fixed")
                    if counts.get("html_escapes", 0) > 0:
                        row_issues.append("HTML entities decoded")
                    if counts.get("spaces_normalized", 0) > 0:
                        row_issues.append("Whitespace normalized")
            elif isinstance(val, list):
                cleaned_list = []
                list_changed = False
                for item in val:
                    if isinstance(item, str):
                        cleaned_item, counts = clean_text_generic(item)
                        cleaned_list.append(cleaned_item)
                        if cleaned_item != item:
                            list_changed = True
                            row_mojibake += counts.get("mojibake_fixes", 0)
                            row_html += counts.get("html_escapes", 0)
                            row_space += counts.get("spaces_normalized", 0)
                            row_unicode += counts.get("unicode_normalizations", 0)
                    else:
                        cleaned_list.append(item)
                cleaned_row[col] = cleaned_list
                if list_changed:
                    row_issues.append("Sequence strings normalized")
                    
        cleaned_row, task_warnings, task_errors = clean_row_task_specific(cleaned_row, task_info)
        
        for col in batch.keys():
            if cleaned_row.get(col) != row.get(col):
                if col == label_col and task == "regression":
                    try:
                        if float(row.get(col)) == float(cleaned_row.get(col)):
                            continue
                    except (ValueError, TypeError):
                        pass
                row_task += 1
                
        for warn in task_warnings:
            row_issues.append(f"Warning: {warn}")
        for err in task_errors:
            row_issues.append(f"Error: {err}")
            
        score, category = compute_row_quality(row, text_columns, label_columns, task_info)
        
        if task_errors:
            category = "CORRUPTED"
            score = min(score, 30.0)
            
        quality_scores.append(score)
        quality_categories.append(category)
        audit_issues_list.append(json.dumps(row_issues))
        
        mojibake_fixes.append(row_mojibake)
        html_escapes.append(row_html)
        spaces_normalized.append(row_space)
        unicode_normalizations.append(row_unicode)
        task_specific_updates.append(row_task)
        
        for col in batch.keys():
            cleaned_batch[col].append(cleaned_row.get(col))
            
    cleaned_batch["__quality_score"] = quality_scores
    cleaned_batch["__quality_category"] = quality_categories
    cleaned_batch["__audit_issues"] = audit_issues_list
    cleaned_batch["__mojibake_fixes"] = mojibake_fixes
    cleaned_batch["__html_escapes"] = html_escapes
    cleaned_batch["__spaces_normalized"] = spaces_normalized
    cleaned_batch["__unicode_normalizations"] = unicode_normalizations
    cleaned_batch["__task_specific_updates"] = task_specific_updates
    
    return cleaned_batch


def compile_reports(dataset, split_name: str | None = None) -> tuple[dict, dict, list, dict]:
    """Consolidate batch quality/audit logs into final summary metrics."""
    scores = dataset["__quality_score"]
    categories = dataset["__quality_category"]
    issues_json = dataset["__audit_issues"]
    
    mojibake = sum(dataset["__mojibake_fixes"])
    html_e = sum(dataset["__html_escapes"])
    spaces = sum(dataset["__spaces_normalized"])
    unicode = sum(dataset["__unicode_normalizations"])
    task_up = sum(dataset["__task_specific_updates"])
    
    total_rows = len(dataset)
    category_counts = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0, "REVIEW_REQUIRED": 0}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    avg_score = sum(scores) / total_rows if total_rows > 0 else 100.0
    
    total_warnings = 0
    total_errors = 0
    for issues_str in issues_json:
        row_issues = json.loads(issues_str)
        for issue in row_issues:
            if issue.startswith("Warning:"):
                total_warnings += 1
            elif issue.startswith("Error:"):
                total_errors += 1
                
    row_level_audit = []
    limit = min(total_rows, 1000)
    for idx in range(limit):
        row_level_audit.append({
            "row_index": idx,
            "quality_score": scores[idx],
            "category": categories[idx],
            "issues": json.loads(issues_json[idx])
        })
        
    quality_report = {
        "average_quality_score": round(avg_score, 2),
        "category_counts": category_counts,
        "category_ratios": {cat: round(cnt / total_rows, 4) for cat, cnt in category_counts.items()} if total_rows > 0 else {}
    }
    
    cleaning_report = {
        "total_rows_processed": total_rows,
        "total_rows_cleaned": sum(1 for x in issues_json if len(json.loads(x)) > 0),
        "mojibake_fixes": mojibake,
        "html_escapes": html_e,
        "spaces_normalized": spaces,
        "unicode_normalizations": unicode,
        "task_specific_updates": task_up
    }
    
    issue_summary = {
        "total_warnings": total_warnings,
        "total_errors": total_errors,
        "corrupted_rows_count": category_counts.get("CORRUPTED", 0)
    }
    
    return cleaning_report, quality_report, row_level_audit, issue_summary


def generate_recommendations(quality_report: dict, issue_summary: dict, task_info: dict) -> list[str]:
    """Produce actionable recommendation steps based on the audit reports."""
    recs = []
    avg_score = quality_report.get("average_quality_score", 100.0)
    if avg_score < 80.0:
        recs.append(f"Average quality score is low ({avg_score:.2f}). Consider vetting higher quality source data.")
        
    corrupted_count = issue_summary.get("corrupted_rows_count", 0)
    if corrupted_count > 0:
        recs.append(f"Automatically removed {corrupted_count} corrupted rows (missing input/label values or list length mismatches).")
        
    cat_counts = quality_report.get("category_counts", {})
    low_qual = cat_counts.get("LOW_QUALITY", 0)
    if low_qual > 0:
        recs.append(f"Identified {low_qual} low-quality rows. Inspect for keyboard mashes, excessive noise, or repeated patterns.")
        
    if issue_summary.get("target_leakage_detected", False):
        recs.append("WARNING: Target label leakage detected. Verbatim labels exist in feature text. Strip labels before training.")
        
    if issue_summary.get("train_test_leakage_detected", False):
        recs.append("WARNING: Split leakage detected. Duplicate strings overlap between train/validation/test. Deduplicate dataset splits.")
        
    if len(recs) == 0:
        recs.append("Dataset quality is high. Ready for training.")
        
    return recs


def preprocess_dataset(dataset, cfg: DictConfig | None = None):
    """
    Universal Preprocessing Entry Point.
    Detects the task type, profiles the schema, runs text sanitation, noise audits, and returns:
    (cleaned_dataset, cleaning_report, quality_report, row_level_audit, issue_summary, warnings, recommendations)
    """
    from datasets import Dataset, DatasetDict
    
    detector = TaskDetector()
    task_info = detector.detect(dataset, cfg or DictConfig({}))
    task_name = task_info.get("task", "classification")
    
    logger.info(f"Universal Preprocessing Engine: Detected Task = '{task_name}' (Confidence = {task_info.get('confidence', 0.0)})")
    
    detected_cols = task_info.get("detected_columns", {})
    text_columns = set(detected_cols.get("text", []))
    label_columns = set(detected_cols.get("label", []))
    
    input_col = task_info.get("input_column")
    label_col = task_info.get("label_column")
    if input_col:
        text_columns.add(input_col)
    if label_col:
        label_columns.add(label_col)
        
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
            
            mapped_ds = ds.map(
                lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
                batched=True,
                desc=f"Cleaning & Auditing '{split_name}' split"
            )
            
            cleaning_rep, quality_rep, row_audit, issue_sum = compile_reports(mapped_ds, split_name)
            split_reports[split_name] = {
                "cleaning_report": cleaning_rep,
                "quality_report": quality_rep,
                "row_level_audit": row_audit,
                "issue_summary": issue_sum
            }
            
            before_len = len(mapped_ds)
            cleaned_ds = mapped_ds.filter(
                lambda x: x["__quality_category"] != "CORRUPTED",
                desc=f"Filtering corrupted rows from '{split_name}'"
            )
            removed_corrupted = before_len - len(cleaned_ds)
            split_reports[split_name]["issue_summary"]["corrupted_rows_removed"] = removed_corrupted
            
            cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)
            split_reports[split_name]["issue_summary"]["duplicates_removed"] = removed_dupes
            
            target_leakage_warnings = check_target_leakage(cleaned_ds, text_columns, label_col, task_name)
            
            row_warnings = set()
            for issues_str in mapped_ds["__audit_issues"]:
                for issue in json.loads(issues_str):
                    if issue.startswith("Warning:"):
                        row_warnings.add(issue.replace("Warning: ", "", 1))
            
            split_reports[split_name]["warnings"] = target_leakage_warnings + list(row_warnings)
            split_reports[split_name]["issue_summary"]["target_leakage_detected"] = len(target_leakage_warnings) > 0
            
            cols_to_remove = [c for c in cleaned_ds.column_names if c.startswith("__")]
            final_ds = cleaned_ds.remove_columns(cols_to_remove)
            cleaned_splits[split_name] = final_ds
            
        combined_cleaned_ds = DatasetDict(cleaned_splits)
        
        total_rows_processed = 0
        total_rows_cleaned = 0
        mojibake_fixes = 0
        html_escapes = 0
        spaces_normalized = 0
        unicode_normalizations = 0
        task_specific_updates = 0
        
        total_warnings = 0
        total_errors = 0
        total_corrupted = 0
        total_corrupted_removed = 0
        total_duplicates_removed = 0
        target_leakage_detected = False
        
        category_counts = {"CLEAN": 0, "FIXABLE": 0, "LOW_QUALITY": 0, "CORRUPTED": 0, "REVIEW_REQUIRED": 0}
        total_score_sum = 0.0
        
        row_level_audit = {}
        warnings = []
        
        for split_name, rep in split_reports.items():
            total_rows_processed += rep["cleaning_report"]["total_rows_processed"]
            total_rows_cleaned += rep["cleaning_report"]["total_rows_cleaned"]
            mojibake_fixes += rep["cleaning_report"]["mojibake_fixes"]
            html_escapes += rep["cleaning_report"]["html_escapes"]
            spaces_normalized += rep["cleaning_report"]["spaces_normalized"]
            unicode_normalizations += rep["cleaning_report"]["unicode_normalizations"]
            task_specific_updates += rep["cleaning_report"]["task_specific_updates"]
            
            total_warnings += rep["issue_summary"]["total_warnings"]
            total_errors += rep["issue_summary"]["total_errors"]
            total_corrupted += rep["issue_summary"]["corrupted_rows_count"]
            total_corrupted_removed += rep["issue_summary"]["corrupted_rows_removed"]
            total_duplicates_removed += rep["issue_summary"]["duplicates_removed"]
            if rep["issue_summary"]["target_leakage_detected"]:
                target_leakage_detected = True
                
            for cat, cnt in rep["quality_report"]["category_counts"].items():
                category_counts[cat] += cnt
            total_score_sum += rep["quality_report"]["average_quality_score"] * rep["cleaning_report"]["total_rows_processed"]
            
            row_level_audit[split_name] = rep["row_level_audit"]
            for w in rep["warnings"]:
                warnings.append(f"[{split_name}] {w}")
                
        split_leakage_warnings = check_train_test_leakage(combined_cleaned_ds)
        for w in split_leakage_warnings:
            warnings.append(w)
            
        avg_quality_score = total_score_sum / total_rows_processed if total_rows_processed > 0 else 100.0
        
        cleaning_report = {
            "total_rows_processed": total_rows_processed,
            "total_rows_cleaned": total_rows_cleaned,
            "mojibake_fixes": mojibake_fixes,
            "html_escapes": html_escapes,
            "spaces_normalized": spaces_normalized,
            "unicode_normalizations": unicode_normalizations,
            "task_specific_updates": task_specific_updates
        }
        
        quality_report = {
            "average_quality_score": round(avg_quality_score, 2),
            "category_counts": category_counts,
            "category_ratios": {cat: round(cnt / total_rows_processed, 4) for cat, cnt in category_counts.items()} if total_rows_processed > 0 else {}
        }
        
        issue_summary = {
            "total_warnings": total_warnings + len(split_leakage_warnings),
            "total_errors": total_errors,
            "corrupted_rows_count": total_corrupted,
            "corrupted_rows_removed": total_corrupted_removed,
            "duplicates_removed": total_duplicates_removed,
            "target_leakage_detected": target_leakage_detected,
            "train_test_leakage_detected": len(split_leakage_warnings) > 0
        }
        
        recommendations = generate_recommendations(quality_report, issue_summary, task_info)
        
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
        if len(dataset) == 0:
            return dataset, {}, {}, [], {}, [], []
            
        logger.info(f"Preprocessing single dataset ({len(dataset)} rows)...")
        
        mapped_ds = dataset.map(
            lambda batch: _clean_and_audit_batch(batch, text_columns, label_columns, task_info),
            batched=True,
            desc="Cleaning & Auditing dataset"
        )
        
        cleaning_report, quality_report, row_level_audit, issue_summary = compile_reports(mapped_ds)
        
        before_len = len(mapped_ds)
        cleaned_ds = mapped_ds.filter(
            lambda x: x["__quality_category"] != "CORRUPTED",
            desc="Filtering corrupted rows"
        )
        removed_corrupted = before_len - len(cleaned_ds)
        issue_summary["corrupted_rows_removed"] = removed_corrupted
        
        cleaned_ds, removed_dupes = remove_duplicate_rows(cleaned_ds, text_columns)
        issue_summary["duplicates_removed"] = removed_dupes
        
        target_leakage_warnings = check_target_leakage(cleaned_ds, text_columns, label_col, task_name)
        issue_summary["target_leakage_detected"] = len(target_leakage_warnings) > 0
        
        row_warnings = set()
        for issues_str in mapped_ds["__audit_issues"]:
            for issue in json.loads(issues_str):
                if issue.startswith("Warning:"):
                    row_warnings.add(issue.replace("Warning: ", "", 1))
                    
        warnings = target_leakage_warnings + list(row_warnings)
        recommendations = generate_recommendations(quality_report, issue_summary, task_info)
        
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
