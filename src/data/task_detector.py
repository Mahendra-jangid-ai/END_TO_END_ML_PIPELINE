# """
# Automatic NLP task detection from dataset schema and content using rule-based heuristics.
# """
# from __future__ import annotations

# import re
# import json
# import difflib
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Set, Tuple
# from omegaconf import DictConfig

# from src.utils.common import get_logger

# logger = get_logger(__name__)

# # ---------------------------------------------------------------------------
# # Constants & Helper Patterns
# # ---------------------------------------------------------------------------

# URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# QUESTION_PATTERN = re.compile(
#     r"\b(who|what|where|when|why|how|whose|which|whom|is|are|can|do|does|did|will|should|would|could)\b", 
#     re.IGNORECASE
# )
# METADATA_NAME_PATTERN = re.compile(
#     r"^(id|uuid|index|timestamp|date|created_at|updated_at|row_id|sample_id|time)$|_id$|^id_|_id_|^index_", 
#     re.IGNORECASE
# )
# TEXT_COLUMN_NAME_PATTERN = re.compile(
#     r"^(text|sentence|document|article|summary|abstract|highlights|question|context|passage|answer|incorrect|corrected|original|simple|complex|formal|informal|prompt|completion|prefix|suffix|message|conversation|query|dialogue|translation)s?$", 
#     re.IGNORECASE
# )

# # ---------------------------------------------------------------------------
# # Dataclasses for Phase 1 (Dataset Profiling)
# # ---------------------------------------------------------------------------

# @dataclass
# class ColumnProfile:
#     dtype: str
#     unique_ratio: float
#     null_ratio: float
#     avg_length: float
#     avg_tokens: float
#     contains_urls: bool
#     contains_questions: bool
#     contains_json: bool
#     contains_lists: bool
#     contains_entities: bool
#     numeric_ratio: float
#     text_ratio: float

# @dataclass
# class DatasetProfile:
#     rows: int
#     columns: List[str]
#     dtypes: Dict[str, str]
#     null_ratios: Dict[str, float]
#     unique_ratios: Dict[str, float]
#     avg_text_lengths: Dict[str, float]
#     avg_token_counts: Dict[str, float]
#     numeric_distributions: Dict[str, Dict[str, float]]
#     categorical_distributions: Dict[str, Dict[str, int]]
#     list_columns: List[str]
#     dict_columns: List[str]
#     nested_structures: Dict[str, str]
#     column_profiles: Dict[str, ColumnProfile]

# # ---------------------------------------------------------------------------
# # Profiling Helper Functions
# # ---------------------------------------------------------------------------

# def _get_column_values(ds: Any, col_name: str, limit: int = 1000) -> List[Any]:
#     """Safely sample first limit values for a given column from a HF dataset."""
#     try:
#         return ds[:limit][col_name]
#     except Exception as e:
#         logger.warning(f"Error getting values for column '{col_name}': {e}")
#         return []

# def _looks_like_json(val: Any) -> bool:
#     """Check if value is a stringified JSON dictionary or list."""
#     if not isinstance(val, str):
#         return False
#     val_s = val.strip()
#     if (val_s.startswith("{") and val_s.endswith("}")) or (val_s.startswith("[") and val_s.endswith("]")):
#         try:
#             json.loads(val_s)
#             return True
#         except ValueError:
#             pass
#     return False

# def _looks_like_list(val: Any) -> bool:
#     """Check if value represents a list/sequence."""
#     if isinstance(val, (list, tuple, set)):
#         return True
#     if not isinstance(val, str):
#         return False
#     val_s = val.strip()
#     if val_s.startswith("[") and val_s.endswith("]"):
#         try:
#             parsed = json.loads(val_s)
#             return isinstance(parsed, list)
#         except ValueError:
#             pass
#         if "," in val_s:
#             return True
#     return False

# def _looks_like_entity(val: Any) -> bool:
#     """Heuristic check to identify if a value contains entity annotations/tags."""
#     if isinstance(val, dict):
#         keys = set(val.keys())
#         entity_keys = {"entity", "label", "start", "end"}
#         if len(keys & entity_keys) >= 2:
#             return True
#     elif isinstance(val, list) and len(val) > 0:
#         if all(isinstance(x, dict) and len(set(x.keys()) & {"entity", "label", "start", "end"}) >= 2 for x in val[:5]):
#             return True
#         if all(isinstance(x, str) and (len(x) <= 10 and (x == "O" or "-" in x)) for x in val[:10]):
#             return True
#     return False

# def _get_string_similarity(s1: str, s2: str) -> float:
#     """Fast similarity ratio between two strings using standard SequenceMatcher."""
#     return difflib.SequenceMatcher(None, s1, s2).ratio()

# def _generate_dataset_profile(ds: Any, limit: int = 1000) -> DatasetProfile:
#     """Build profiles for each column and generate DatasetProfile."""
#     rows = len(ds)
#     columns = ds.column_names
#     dtypes = {}
#     null_ratios = {}
#     unique_ratios = {}
#     avg_text_lengths = {}
#     avg_token_counts = {}
#     numeric_distributions = {}
#     categorical_distributions = {}
#     list_columns = []
#     dict_columns = []
#     nested_structures = {}
#     column_profiles = {}

#     from datasets.features import Value, ClassLabel, Sequence

#     for col in columns:
#         vals = _get_column_values(ds, col, limit=limit)
        
#         is_list = False
#         is_dict = False
#         nested_desc = "scalar"
        
#         feat = ds.features.get(col)
#         if isinstance(feat, Sequence):
#             is_list = True
#             nested_desc = f"list[{feat.feature}]"
#         elif isinstance(feat, dict):
#             is_dict = True
#             nested_desc = "dict"
        
#         non_null_vals = []
#         null_count = 0
#         for v in vals:
#             if v is None:
#                 null_count += 1
#             elif isinstance(v, str) and v.lower() in {"", "nan", "null", "none", "n/a", "na"}:
#                 null_count += 1
#             else:
#                 non_null_vals.append(v)
                
#         if not is_list and any(isinstance(v, (list, tuple)) for v in non_null_vals):
#             is_list = True
#             nested_desc = "list"
#         if not is_dict and any(isinstance(v, dict) for v in non_null_vals):
#             is_dict = True
#             nested_desc = "dict"
            
#         if is_list:
#             list_columns.append(col)
#         if is_dict:
#             dict_columns.append(col)
            
#         nested_structures[col] = nested_desc
        
#         flat_vals = []
#         if is_list:
#             for v in non_null_vals:
#                 if isinstance(v, (list, tuple)):
#                     flat_vals.extend(v)
#                 elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
#                     try:
#                         parsed = json.loads(v)
#                         if isinstance(parsed, list):
#                             flat_vals.extend(parsed)
#                         else:
#                             flat_vals.append(v)
#                     except ValueError:
#                         flat_vals.append(v)
#                 else:
#                     flat_vals.append(v)
#         else:
#             flat_vals = non_null_vals
            
#         flat_vals = [f for f in flat_vals if f is not None]
        
#         str_vals = [str(v) for v in flat_vals if isinstance(v, (str, bytes))]
#         text_ratio = len(str_vals) / len(flat_vals) if flat_vals else 0.0
        
#         num_vals = []
#         for v in flat_vals:
#             if isinstance(v, (int, float)) and not isinstance(v, bool):
#                 num_vals.append(float(v))
#             elif isinstance(v, str):
#                 try:
#                     num_vals.append(float(v))
#                 except ValueError:
#                     pass
#         numeric_ratio = len(num_vals) / len(flat_vals) if flat_vals else 0.0
        
#         lengths = [len(v) for v in str_vals]
#         avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        
#         token_counts = [len(v.split()) for v in str_vals]
#         avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
        
#         null_ratio = null_count / len(vals) if vals else 1.0
#         unique_ratio = len(set(str(v) for v in non_null_vals)) / len(non_null_vals) if non_null_vals else 0.0
        
#         contains_urls = any(isinstance(v, str) and URL_PATTERN.search(v) for v in str_vals)
#         contains_questions = any(isinstance(v, str) and ("?" in v or QUESTION_PATTERN.match(v)) for v in str_vals)
#         contains_json = any(_looks_like_json(v) for v in non_null_vals)
#         contains_lists = is_list or any(_looks_like_list(v) for v in non_null_vals)
#         contains_entities = any(_looks_like_entity(v) for v in non_null_vals) or (is_list and any(_looks_like_entity(v) for v in flat_vals))
        
#         if is_list:
#             dtype = "list"
#         elif is_dict:
#             dtype = "dict"
#         elif numeric_ratio > 0.8:
#             if all(float(v).is_integer() for v in num_vals):
#                 dtype = "int"
#             else:
#                 dtype = "float"
#         elif text_ratio > 0.8:
#             dtype = "string"
#         else:
#             dtype = "unknown"
            
#         dtypes[col] = dtype
#         null_ratios[col] = null_ratio
#         unique_ratios[col] = unique_ratio
#         avg_text_lengths[col] = avg_length
#         avg_token_counts[col] = avg_tokens
        
#         if num_vals:
#             numeric_distributions[col] = {
#                 "min": min(num_vals),
#                 "max": max(num_vals),
#                 "mean": sum(num_vals) / len(num_vals),
#             }
            
#         unique_vals_count = len(set(str(v) for v in non_null_vals))
#         if unique_vals_count <= 200 and len(non_null_vals) > 0:
#             counts = {}
#             for v in non_null_vals:
#                 counts[str(v)] = counts.get(str(v), 0) + 1
#             categorical_distributions[col] = counts
            
#         column_profiles[col] = ColumnProfile(
#             dtype=dtype,
#             unique_ratio=unique_ratio,
#             null_ratio=null_ratio,
#             avg_length=avg_length,
#             avg_tokens=avg_tokens,
#             contains_urls=contains_urls,
#             contains_questions=contains_questions,
#             contains_json=contains_json,
#             contains_lists=contains_lists,
#             contains_entities=contains_entities,
#             numeric_ratio=numeric_ratio,
#             text_ratio=text_ratio,
#         )
        
#     return DatasetProfile(
#         rows=rows,
#         columns=columns,
#         dtypes=dtypes,
#         null_ratios=null_ratios,
#         unique_ratios=unique_ratios,
#         avg_text_lengths=avg_text_lengths,
#         avg_token_counts=avg_token_counts,
#         numeric_distributions=numeric_distributions,
#         categorical_distributions=categorical_distributions,
#         list_columns=list_columns,
#         dict_columns=dict_columns,
#         nested_structures=nested_structures,
#         column_profiles=column_profiles,
#     )

# # ---------------------------------------------------------------------------
# # Phase 2 & 3: Column Role Classification & Identification Heuristics
# # ---------------------------------------------------------------------------

# def _is_metadata_column(col_name: str, profile: ColumnProfile, rows_count: int) -> bool:
#     """Determine if a column holds metadata/indexes instead of model targets/features."""
#     if METADATA_NAME_PATTERN.search(col_name):
#         return True
#     if profile.unique_ratio == 1.0 and rows_count > 5:
#         if profile.dtype in ("int", "string") and profile.avg_length < 40:
#             return True
#     if profile.null_ratio > 0.95:
#         return True
#     return False

# def _score_input_text_candidate(col_name: str, col_prof: ColumnProfile) -> float:
#     """Heuristic scoring for text input candidates. Prefers long, unique natural text."""
#     if METADATA_NAME_PATTERN.search(col_name):
#         return 0.0
    
#     score = 0.0
#     score += min(col_prof.avg_length / 200.0, 1.0) * 0.3
#     score += min(col_prof.avg_tokens / 50.0, 1.0) * 0.3
#     score += col_prof.unique_ratio * 0.2
    
#     if col_prof.dtype == "string":
#         score += 0.2
#     if re.search(r"^(text|sentence|prompt|input|content|document|premise|context|article|story|passage)s?$", col_name, re.IGNORECASE):
#         score += 0.15
        
#     return score

# def _score_label_candidate(col_name: str, col_prof: ColumnProfile, input_col: str) -> float:
#     """Heuristic scoring for category label candidates."""
#     if col_name == input_col or METADATA_NAME_PATTERN.search(col_name):
#         return 0.0
    
#     score = 0.0
#     if 0.0 < col_prof.unique_ratio < 0.1:
#         score += 0.3
#     elif col_prof.unique_ratio >= 0.1 and col_prof.unique_ratio < 0.3:
#         score += 0.15
        
#     if col_prof.avg_tokens <= 2.0:
#         score += 0.2
#     if col_prof.dtype in ("string", "category", "int"):
#         score += 0.2
#     if re.search(r"^(label|sentiment|target|class|category|tag|output|intent|y)s?$", col_name, re.IGNORECASE):
#         score += 0.3
        
#     return score

# def _score_numeric_target(col_name: str, col_prof: ColumnProfile, input_col: str) -> float:
#     """Heuristic scoring for continuous regression targets."""
#     if col_name == input_col or METADATA_NAME_PATTERN.search(col_name):
#         return 0.0
        
#     score = 0.0
#     if col_prof.numeric_ratio > 0.95 and col_prof.unique_ratio > 0.01:
#         score += 0.6
#         if col_prof.dtype == "float":
#             score += 0.2
#         if re.search(r"^(score|rating|value|target|loss)s?$", col_name, re.IGNORECASE):
#             score += 0.2
#     return score

# def _score_list_target(col_name: str, col_prof: ColumnProfile, input_col: str) -> float:
#     """Heuristic scoring for list targets (e.g. multi-label lists)."""
#     if col_name == input_col or METADATA_NAME_PATTERN.search(col_name):
#         return 0.0
        
#     score = 0.0
#     if col_prof.contains_lists or col_prof.dtype == "list":
#         score += 0.6
#         if re.search(r"^(labels|tags|categories|genres|targets)s?$", col_name, re.IGNORECASE):
#             score += 0.2
#     return score

# # ---------------------------------------------------------------------------
# # Phase 5: Individual Task Evaluators (24 Tasks)
# # ---------------------------------------------------------------------------

# def _evaluate_classification(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if col not in text_cols and re.search(r"^(label|sentiment|class|category|target|intent|y)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("column_name_matches_classification_label")
            
#     active_labels = [c for c in label_cols if c not in text_cols]
#     active_nums = [c for c in num_cols if c not in text_cols]
    
#     if len(text_cols) == 1 and len(active_labels) >= 1:
#         struct_score = 1.0
#         rules.append("one_text_input_one_categorical_label")
#     elif len(text_cols) > 1 and len(active_labels) >= 1:
#         struct_score = 0.5
#         rules.append("multiple_text_inputs_penalized_for_simple_classification")
#     elif len(text_cols) >= 1 and len(active_nums) >= 1:
#         for col in active_nums:
#             if col in active_labels:
#                 struct_score = 0.8
#                 rules.append("one_text_input_one_numeric_categorical_label")
#                 break
                
#     best_label_col = active_labels[0] if active_labels else (active_nums[0] if active_nums else None)
#     if best_label_col:
#         vals = [r.get(best_label_col) for r in sample_rows if r.get(best_label_col) is not None]
#         if vals:
#             is_valid = all(isinstance(v, (int, str, bool)) and len(str(v)) < 30 for v in vals)
#             if is_valid:
#                 sample_score = 1.0
#                 rules.append("sample_labels_are_short_categories")
                
#     if best_label_col and best_label_col in profile.column_profiles:
#         prof = profile.column_profiles[best_label_col]
#         unique_count = len(profile.categorical_distributions.get(best_label_col, {}))
#         if 2 <= unique_count <= 200:
#             stat_score = 1.0
#             rules.append("label_cardinality_within_classification_range")
#             if prof.unique_ratio < 0.85:
#                 stat_score = max(stat_score, 1.0)
#                 rules.append("label_has_high_repetition_rate")
                
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_multi_label_classification(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if col not in text_cols and re.search(r"^(labels|tags|genres|categories|targets)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("column_name_matches_multilabel_patterns")
            
#     active_lists = [c for c in list_cols if c not in text_cols]
#     active_labels = [c for c in label_cols if c not in text_cols]
    
#     if len(text_cols) >= 1 and len(active_lists) >= 1:
#         struct_score = 1.0
#         rules.append("text_input_and_list_column_detected")
#     elif len(text_cols) >= 1 and len(active_labels) >= 1:
#         struct_score = 0.5
        
#     best_list_col = active_lists[0] if active_lists else (active_labels[0] if active_labels else None)
#     if best_list_col:
#         vals = [r.get(best_list_col) for r in sample_rows if r.get(best_list_col) is not None]
#         if vals:
#             if any(isinstance(v, (list, tuple)) for v in vals):
#                 sample_score = 1.0
#                 rules.append("samples_contain_list_labels")
#             elif any(isinstance(v, str) and "," in v for v in vals):
#                 sample_score = 0.8
#                 rules.append("samples_contain_comma_separated_strings")
                
#     if best_list_col and best_list_col in profile.column_profiles:
#         prof = profile.column_profiles[best_list_col]
#         if prof.contains_lists or prof.dtype == "list":
#             stat_score = 1.0
#             rules.append("target_profile_indicates_lists")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_regression(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if col not in text_cols and not METADATA_NAME_PATTERN.search(col):
#             if re.search(r"^(score|rating|value|target|loss|numeric|y)$", col, re.IGNORECASE):
#                 col_score = max(col_score, 1.0)
#                 rules.append("column_name_matches_regression_target")
                
#     non_meta_num_cols = [c for c in num_cols if not METADATA_NAME_PATTERN.search(c) and c not in text_cols]
#     if len(text_cols) >= 1 and len(non_meta_num_cols) >= 1:
#         struct_score = 1.0
#         rules.append("text_input_and_numeric_target_detected")
        
#     best_num_col = non_meta_num_cols[0] if non_meta_num_cols else None
#     if best_num_col:
#         vals = [r.get(best_num_col) for r in sample_rows if r.get(best_num_col) is not None]
#         if vals:
#             if any(isinstance(v, float) for v in vals):
#                 sample_score = 1.0
#                 rules.append("samples_contain_float_targets")
#             elif all(isinstance(v, (int, float)) for v in vals):
#                 sample_score = 0.7
#                 rules.append("samples_contain_all_numeric_targets")
                
#     if best_num_col and best_num_col in profile.column_profiles:
#         prof = profile.column_profiles[best_num_col]
#         if prof.unique_ratio > 0.05:
#             stat_score = 1.0
#             rules.append("numeric_target_has_continuous_distribution")
#         elif prof.unique_ratio > 0.01:
#             stat_score = 0.6
#             rules.append("numeric_target_has_moderate_unique_ratio")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_token_classification(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(tokens|words|ner_tags|pos_tags|chunk_tags|tags)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("column_name_matches_token_classification_patterns")
            
#     if len(list_cols) >= 2:
#         struct_score = 1.0
#         rules.append("multiple_list_columns_detected")
#     elif len(list_cols) == 1 and len(text_cols) >= 1:
#         struct_score = 0.5
#         rules.append("single_list_column_with_text_input")
        
#     if len(list_cols) >= 2:
#         lengths_match = True
#         for row in sample_rows[:10]:
#             val1 = row.get(list_cols[0])
#             val2 = row.get(list_cols[1])
#             if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
#                 if len(val1) != len(val2):
#                     lengths_match = False
#                     break
#             else:
#                 lengths_match = False
#                 break
#         if lengths_match:
#             sample_score = 1.0
#             rules.append("aligned_parallel_token_and_tag_lists")
            
#     ner_tag_found = False
#     for col in list_cols:
#         prof = profile.column_profiles[col]
#         if prof.contains_entities:
#             ner_tag_found = True
#             break
#     if ner_tag_found:
#         stat_score = 1.0
#         rules.append("entity_tags_profile_matches")
#     elif len(list_cols) >= 2:
#         stat_score = 0.7
#         rules.append("aligned_sequence_profile")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_question_answering(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_q = any(re.search(r"^(question|query|q)$", col, re.IGNORECASE) for col in profile.columns)
#     has_a = any(re.search(r"^(answer|answers|a)$", col, re.IGNORECASE) for col in profile.columns)
#     has_c = any(re.search(r"^(context|passage|document|c)$", col, re.IGNORECASE) for col in profile.columns)
            
#     if has_q and has_a:
#         col_score = 1.0
#         rules.append("column_names_match_question_and_answer")
#         if has_c:
#             col_score = 1.0
#             rules.append("column_name_matches_context")
#     elif has_q or has_a:
#         col_score = 0.5
        
#     if len(text_cols) >= 3:
#         struct_score = 1.0
#         rules.append("three_text_columns_for_qa")
#     elif len(text_cols) == 2:
#         struct_score = 0.7
#         rules.append("two_text_columns_for_qa")
        
#     q_col = None
#     for col in text_cols:
#         prof = profile.column_profiles[col]
#         if prof.contains_questions:
#             q_col = col
#             sample_score = max(sample_score, 0.8)
#             rules.append(f"column_{col}_contains_questions")
            
#     c_col = None
#     a_col = None
#     for col in text_cols:
#         if col == q_col:
#             continue
#         if c_col is None or profile.column_profiles[col].avg_length > profile.column_profiles[c_col].avg_length:
#             c_col = col
            
#     for col in text_cols:
#         if col != q_col and col != c_col:
#             a_col = col
#             break
            
#     if q_col and c_col and a_col:
#         substring_match = 0
#         valid_samples = 0
#         for r in sample_rows:
#             c_val = r.get(c_col)
#             a_val = r.get(a_col)
#             if isinstance(c_val, str) and isinstance(a_val, str) and a_val:
#                 valid_samples += 1
#                 if a_val.lower() in c_val.lower():
#                     substring_match += 1
#         if valid_samples > 0 and substring_match / valid_samples > 0.4:
#             sample_score = 1.0
#             rules.append("answers_are_substrings_of_context")
            
#     if q_col and c_col:
#         q_len = profile.column_profiles[q_col].avg_length
#         c_len = profile.column_profiles[c_col].avg_length
#         if q_len < c_len * 0.5:
#             if a_col:
#                 stat_score = 1.0
#             else:
#                 stat_score = max(stat_score, 0.8)
#             rules.append("question_length_shorter_than_context")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_summarization(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_src = any(re.search(r"^(document|article|text|body|passage)$", col, re.IGNORECASE) for col in profile.columns)
#     has_tgt = any(re.search(r"^(summary|highlights|abstract)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_src and has_tgt:
#         col_score = 1.0
#         rules.append("column_names_match_summarization_source_target")
#     elif has_tgt:
#         col_score = 0.6
#         rules.append("column_name_matches_summary")
        
#     if len(text_cols) >= 2:
#         struct_score = 1.0
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         longest_col = max(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         shortest_col = min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         if longest_col != shortest_col:
#             longer_vals = [len(str(r.get(longest_col) or "")) for r in sample_rows]
#             shorter_vals = [len(str(r.get(shortest_col) or "")) for r in sample_rows]
#             ratio_matches = sum(l > s * 1.5 for l, s in zip(longer_vals, shorter_vals) if s > 0)
#             if len(longer_vals) > 0 and ratio_matches / len(longer_vals) > 0.8:
#                 sample_score = 1.0
#                 rules.append("source_text_consistently_longer_than_summary")
                
#     if len(text_cols) >= 2:
#         longest_col = max(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         shortest_col = min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         longest_prof = profile.column_profiles[longest_col]
#         shortest_prof = profile.column_profiles[shortest_col]
#         if shortest_prof.avg_length > 0:
#             compression_ratio = longest_prof.avg_length / shortest_prof.avg_length
#             if 2.0 <= compression_ratio <= 50.0:
#                 stat_score = 1.0
#                 rules.append(f"compression_ratio_matched_at_{compression_ratio:.1f}x")
#             elif compression_ratio > 1.2:
#                 stat_score = 0.5
#                 rules.append("mild_compression_ratio")
                
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_translation(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     lang_codes = {"en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"}
#     has_translation_col = "translation" in profile.columns
#     has_lang_cols = len([c for c in profile.columns if c.lower() in lang_codes or "_" in c or "lang" in c.lower()]) >= 2
#     if has_translation_col:
#         col_score = 1.0
#         rules.append("translation_column_detected")
#     elif has_lang_cols:
#         col_score = 0.8
#         rules.append("language_code_columns_detected")
        
#     if has_translation_col and "translation" in profile.dict_columns:
#         struct_score = 1.0
#         rules.append("translation_dict_structure_profile")
#     elif len(text_cols) >= 2:
#         struct_score = 0.8
#         rules.append("two_text_columns_for_translation")
        
#     if has_translation_col:
#         vals = [r.get("translation") for r in sample_rows if isinstance(r.get("translation"), dict)]
#         if vals and any(len(v) >= 2 for v in vals):
#             sample_score = 1.0
#             rules.append("translation_dictionary_contains_multilingual_keys")
#     elif len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         vals1 = [str(r.get(t1) or "") for r in sample_rows]
#         vals2 = [str(r.get(t2) or "") for r in sample_rows]
#         if vals1 and vals2:
#             similar_lengths = sum(abs(len(v1) - len(v2)) / max(len(v1), len(v2), 1) < 0.4 for v1, v2 in zip(vals1, vals2))
#             if len(vals1) > 0 and similar_lengths / len(vals1) > 0.7:
#                 sample_score = 0.8
#                 rules.append("similar_sentence_lengths_across_languages")
                
#     if len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         overlap = 0.0
#         if sample_rows:
#             words1 = set(" ".join([str(r.get(t1) or "") for r in sample_rows]).lower().split())
#             words2 = set(" ".join([str(r.get(t2) or "") for r in sample_rows]).lower().split())
#             if words1 and words2:
#                 overlap = len(words1 & words2) / len(words1 | words2)
#         if overlap < 0.15:
#             stat_score = 1.0
#             rules.append(f"low_word_overlap_ratio_{overlap:.2f}")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_text_generation(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(prompt|completion|text|generation|response|output)$", col, re.IGNORECASE):
#             col_score = max(col_score, 0.8)
#             rules.append("column_name_suggests_generation")
            
#     if len(text_cols) >= 1:
#         struct_score = 0.8
#         rules.append("text_columns_available")
        
#     text_col = text_cols[0] if text_cols else None
#     if text_col:
#         vals = [str(r.get(text_col) or "") for r in sample_rows]
#         if vals and all(len(v) > 30 for v in vals if v):
#             sample_score = 0.8
#             rules.append("samples_are_long_freeform_texts")
            
#     if text_col and text_col in profile.column_profiles:
#         prof = profile.column_profiles[text_col]
#         if prof.unique_ratio > 0.8 and prof.avg_tokens > 10.0:
#             stat_score = 0.8
#             rules.append("high_uniqueness_and_token_length")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_instruction_tuning(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_inst = any(re.search(r"^(instruction|prompt)$", col, re.IGNORECASE) for col in profile.columns)
#     has_out = any(re.search(r"^(output|response|completion)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_inst and has_out:
#         col_score = 1.0
#         rules.append("instruction_and_output_columns_detected")
#     elif has_inst:
#         col_score = 0.6
#         rules.append("instruction_column_detected")
        
#     if len(text_cols) >= 2:
#         struct_score = 1.0
#         rules.append("at_least_two_text_columns_detected")
        
#     inst_col = next((c for c in text_cols if re.search(r"instruction|prompt", c, re.IGNORECASE)), text_cols[0] if text_cols else None)
#     if inst_col:
#         vals = [str(r.get(inst_col) or "") for r in sample_rows]
#         commands = {"write", "explain", "create", "how", "what", "translate", "summarize", "find", "list", "give", "generate"}
#         command_hits = sum(any(v.lower().startswith(cmd) for cmd in commands) for v in vals if v)
#         if len(vals) > 0 and command_hits / len(vals) > 0.2:
#             sample_score = 1.0
#             rules.append("instruction_column_starts_with_commands")
            
#     if len(text_cols) >= 2:
#         stat_score = 0.8
#         rules.append("instruction_dataset_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_chat(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     chat_col = None
#     for col in profile.columns:
#         if re.search(r"^(messages|conversations|chat|dialog|dialogue)$", col, re.IGNORECASE):
#             col_score = 1.0
#             chat_col = col
#             rules.append(f"column_{col}_matches_chat_patterns")
            
#     if chat_col and (chat_col in profile.list_columns or chat_col in profile.dict_columns):
#         struct_score = 1.0
#         rules.append("chat_column_has_nested_structure")
        
#     if chat_col:
#         vals = [r.get(chat_col) for r in sample_rows if r.get(chat_col) is not None]
#         if vals:
#             is_chat = True
#             for val in vals[:5]:
#                 if not isinstance(val, list):
#                     is_chat = False
#                     break
#                 if len(val) == 0:
#                     continue
#                 first_turn = val[0]
#                 if not isinstance(first_turn, dict) or not ("role" in first_turn or "from" in first_turn):
#                     is_chat = False
#                     break
#             if is_chat:
#                 sample_score = 1.0
#                 rules.append("chat_messages_have_role_and_content")
                
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("chat_profile_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_retrieval(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_q = any(re.search(r"^(query|question|search_query)$", col, re.IGNORECASE) for col in profile.columns)
#     has_d = any(re.search(r"^(document|passage|context|text|positive|negative)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_q and has_d:
#         col_score = 1.0
#         rules.append("query_and_document_columns_detected")
#     elif has_q:
#         col_score = 0.5
        
#     if len(text_cols) >= 2:
#         struct_score = 0.9
#         if len(text_cols) >= 3:
#             struct_score *= 0.5
#         rules.append("at_least_two_text_columns_for_retrieval")
        
#     q_cols = [c for c in text_cols if re.search(r"query|question", c, re.IGNORECASE)]
#     d_cols = [c for c in text_cols if c not in q_cols]
#     if q_cols and d_cols:
#         q_col = q_cols[0]
#         d_col = d_cols[0]
#         q_len = profile.column_profiles[q_col].avg_length
#         d_len = profile.column_profiles[d_col].avg_length
#         if q_len < d_len * 0.4:
#             sample_score = 1.0
#             rules.append("query_text_is_significantly_shorter_than_document")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("retrieval_lengths_asymmetry_matched")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_text_similarity(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_pair = any(re.search(r"^(sentence1|text1|s1|sentence2|text2|s2)$", col, re.IGNORECASE) for col in profile.columns)
#     has_score = any(re.search(r"^(similarity|score|sim_score)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_pair and has_score:
#         col_score = 1.0
#         rules.append("similarity_columns_detected")
#     elif has_pair:
#         col_score = 0.6
        
#     non_meta_num = [c for c in num_cols if not METADATA_NAME_PATTERN.search(c) and c not in text_cols]
#     if len(text_cols) >= 2 and len(non_meta_num) >= 1:
#         struct_score = 1.0
#         rules.append("two_text_inputs_and_numeric_score")
        
#     if len(text_cols) >= 2 and non_meta_num:
#         t1, t2 = text_cols[0], text_cols[1]
#         len1 = profile.column_profiles[t1].avg_length
#         len2 = profile.column_profiles[t2].avg_length
#         if abs(len1 - len2) / max(len1, len2, 1) < 0.3:
#             sample_score = 0.9
#             rules.append("two_inputs_have_similar_average_lengths")
            
#     if non_meta_num:
#         score_col = non_meta_num[0]
#         if profile.column_profiles[score_col].unique_ratio > 0.05:
#             stat_score = 1.0
#             rules.append("similarity_score_has_continuous_distribution")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_ranking(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_pos = any(re.search(r"^(positive|pos|anchor)$", col, re.IGNORECASE) for col in profile.columns)
#     has_neg = any(re.search(r"^(negative|neg)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_pos and has_neg:
#         col_score = 1.0
#         rules.append("ranking_positive_negative_columns_detected")
        
#     if len(text_cols) >= 3:
#         struct_score = 1.0
#         rules.append("three_text_columns_for_ranking")
        
#     if len(text_cols) >= 3:
#         sorted_cols = sorted(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         q_col = sorted_cols[0]
#         pos_col = sorted_cols[1]
#         if profile.column_profiles[q_col].avg_length < profile.column_profiles[pos_col].avg_length * 0.5:
#             sample_score = 0.9
#             rules.append("query_shorter_than_ranking_documents")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("ranking_profile_matched")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_information_extraction(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(entities|relations|extracted|json_data|ie_output|triplets)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("column_name_indicates_extraction")
            
#     has_dict_or_json = len(profile.dict_columns) >= 1 or any(profile.column_profiles[c].contains_json for c in profile.columns)
#     has_entity_target = any(profile.column_profiles[c].contains_entities for c in profile.columns)
#     if len(text_cols) >= 1 and (has_dict_or_json or has_entity_target):
#         struct_score = 1.0
#         rules.append("text_input_and_structured_output_detected")
        
#     target_col = None
#     for col in profile.columns:
#         if col in text_cols:
#             continue
#         if profile.column_profiles[col].contains_json or profile.column_profiles[col].contains_entities or col in profile.dict_columns:
#             target_col = col
#             break
            
#     if target_col:
#         vals = [r.get(target_col) for r in sample_rows if r.get(target_col) is not None]
#         if vals:
#             if any(isinstance(v, dict) for v in vals) or any(_looks_like_json(v) for v in vals if isinstance(v, str)):
#                 sample_score = 1.0
#                 rules.append("samples_contain_parsed_ie_structures")
                
#     if target_col and target_col in profile.column_profiles:
#         prof = profile.column_profiles[target_col]
#         if prof.contains_json or prof.contains_entities:
#             stat_score = 1.0
#             rules.append("target_matches_information_extraction_signature")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_text_pair_classification(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_pair = any(re.search(r"^(sentence1|text1|sentence2|text2|s1|s2|premise|hypothesis)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_pair and len(label_cols) >= 1:
#         col_score = 1.0
#         rules.append("text_pair_and_label_columns_detected")
        
#     if len(text_cols) >= 2 and len(label_cols) == 1:
#         struct_score = 1.0
#         rules.append("two_text_inputs_and_one_label_column")
        
#     if len(text_cols) >= 2 and label_cols:
#         l_col = label_cols[0]
#         vals = [r.get(l_col) for r in sample_rows if r.get(l_col) is not None]
#         if vals and all(isinstance(v, (int, str, bool)) for v in vals):
#             sample_score = 1.0
#             rules.append("samples_match_text_pair_classification_structure")
            
#     if len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         len1 = profile.column_profiles[t1].avg_length
#         len2 = profile.column_profiles[t2].avg_length
#         if abs(len1 - len2) / max(len1, len2, 1) < 0.4:
#             stat_score = 1.0
#             rules.append("text_pair_lengths_are_comparable")
            
#     # Reduce score if specific NLI columns match to let NLI win
#     if any(re.search(r"^(premise|hypothesis)$", c, re.IGNORECASE) for c in profile.columns):
#         col_score *= 0.5
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_nli(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_nli_names = any(re.search(r"^(premise|hypothesis)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_nli_names:
#         col_score = 1.0
#         rules.append("premise_and_hypothesis_columns_detected")
        
#     if len(text_cols) >= 2 and len(label_cols) == 1:
#         p_match = any(re.search(r"premise", c, re.IGNORECASE) for c in text_cols)
#         h_match = any(re.search(r"hypothesis", c, re.IGNORECASE) for c in text_cols)
#         if p_match and h_match:
#             struct_score = 1.0
#             rules.append("premise_hypothesis_structure_matched")
#         else:
#             struct_score = 0.6
#             rules.append("generic_pair_structure")
            
#     if label_cols:
#         l_col = label_cols[0]
#         vals = {str(r.get(l_col)).lower().strip() for r in sample_rows if r.get(l_col) is not None}
#         nli_labels = {"entailment", "neutral", "contradiction", "0", "1", "2"}
#         if vals and (vals.issubset(nli_labels) or len(vals & {"entailment", "neutral", "contradiction"}) > 0):
#             sample_score = 1.0
#             rules.append("labels_match_nli_vocabulary")
            
#     if label_cols:
#         l_col = label_cols[0]
#         unique_vals = set(profile.categorical_distributions.get(l_col, {}).keys())
#         unique_vals = {v for v in unique_vals if v not in {"none", "nan", "null", ""}}
#         if len(unique_vals) == 3 or (profile.rows <= 10 and len(unique_vals) >= 2):
#             stat_score = 1.0
#             rules.append("label_column_has_exactly_three_classes")
            
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_dialogue(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(dialogue|conversation|history|utterances|turns)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("column_name_matches_dialogue")
            
#     if len(text_cols) >= 1:
#         struct_score = 0.7
#         rules.append("text_input_column_detected")
        
#     text_col = text_cols[0] if text_cols else None
#     if text_col:
#         vals = [str(r.get(text_col) or "") for r in sample_rows]
#         dialogue_markers = [re.search(r"\b(speaker\s*[1-2]|user|assistant|customer|agent|person\s*[a-z]|[a-z]:)\b", v, re.IGNORECASE) is not None for v in vals]
#         if len(vals) > 0 and sum(dialogue_markers) / len(vals) > 0.4:
#             sample_score = 1.0
#             rules.append("samples_contain_speaker_dialogue_markers")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("dialogue_speaker_profile_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_grammar_correction(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_orig = any(re.search(r"^(original|input|source|incorrect|bad|raw)$", col, re.IGNORECASE) for col in profile.columns)
#     has_corr = any(re.search(r"^(corrected|output|target|correct|good|clean)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_orig and has_corr:
#         col_score = 1.0
#         rules.append("grammar_correction_column_names_detected")
        
#     if len(text_cols) >= 2:
#         struct_score = 0.9
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         vals1 = [str(r.get(t1) or "") for r in sample_rows]
#         vals2 = [str(r.get(t2) or "") for r in sample_rows]
#         if vals1 and vals2:
#             similarities = [_get_string_similarity(v1, v2) for v1, v2 in zip(vals1, vals2)]
#             high_similarity = sum(s > 0.85 and s < 1.0 for s in similarities)
#             if len(similarities) > 0 and high_similarity / len(similarities) > 0.5:
#                 sample_score = 1.0
#                 rules.append("samples_have_high_string_similarity_but_not_identical")
                
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("grammar_correction_overlap_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_text_simplification(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_comp = any(re.search(r"^(complex|original|input)$", col, re.IGNORECASE) for col in profile.columns)
#     has_simp = any(re.search(r"^(simple|simplified|output)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_comp and has_simp:
#         col_score = 1.0
#         rules.append("simplification_columns_detected")
        
#     if len(text_cols) >= 2:
#         struct_score = 0.9
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         len1 = profile.column_profiles[t1].avg_length
#         len2 = profile.column_profiles[t2].avg_length
#         if len1 > len2 * 1.1 and len1 < len2 * 1.6:
#             vals1 = [str(r.get(t1) or "") for r in sample_rows]
#             vals2 = [str(r.get(t2) or "") for r in sample_rows]
#             similarities = [_get_string_similarity(v1, v2) for v1, v2 in zip(vals1, vals2)]
#             moderate_similarity = sum(s > 0.5 and s < 0.85 for s in similarities)
#             if len(similarities) > 0 and moderate_similarity / len(similarities) > 0.4:
#                 sample_score = 1.0
#                 rules.append("samples_show_simplification_compression_and_overlap")
                
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("simplification_length_and_similarity_patterns")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_style_transfer(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_style = any(re.search(r"^(style|formal|informal|polite|rude|active|passive)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_style:
#         col_score = 1.0
#         rules.append("style_transfer_indicators_in_column_names")
        
#     if len(text_cols) >= 2:
#         struct_score = 0.8
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         t1, t2 = text_cols[0], text_cols[1]
#         len1 = profile.column_profiles[t1].avg_length
#         len2 = profile.column_profiles[t2].avg_length
#         if abs(len1 - len2) / max(len1, len2, 1) < 0.25:
#             vals1 = [str(r.get(t1) or "") for r in sample_rows]
#             vals2 = [str(r.get(t2) or "") for r in sample_rows]
#             similarities = [_get_string_similarity(v1, v2) for v1, v2 in zip(vals1, vals2)]
#             style_similarity = sum(s > 0.4 and s < 0.8 for s in similarities)
#             if len(similarities) > 0 and style_similarity / len(similarities) > 0.4:
#                 sample_score = 1.0
#                 rules.append("samples_have_moderate_overlap_and_similar_lengths")
                
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("style_transfer_statistical_pattern_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_prompt_completion(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     has_prompt = any(re.search(r"^(prompt)$", col, re.IGNORECASE) for col in profile.columns)
#     has_comp = any(re.search(r"^(completion|continuation)$", col, re.IGNORECASE) for col in profile.columns)
#     if has_prompt and has_comp:
#         col_score = 1.0
#         rules.append("prompt_and_completion_columns_detected")
#     elif has_prompt or has_comp:
#         col_score = 0.5
        
#     if len(text_cols) >= 2:
#         struct_score = 0.9
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         p_col = next((c for c in text_cols if re.search(r"prompt", c, re.IGNORECASE)), text_cols[0])
#         c_col = next((c for c in text_cols if c != p_col), text_cols[1])
#         prompts = [str(r.get(p_col) or "").strip() for r in sample_rows]
#         completions = [str(r.get(c_col) or "").strip() for r in sample_rows]
#         continuation_hits = 0
#         for p, c in zip(prompts, completions):
#             if p and c:
#                 if p[-1] not in {".", "?", "!"} or c[0].islower():
#                     continuation_hits += 1
#         if len(prompts) > 0 and continuation_hits / len(prompts) > 0.4:
#             sample_score = 1.0
#             rules.append("completion_continues_prompt_syntactically")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("prompt_completion_statistical_flow_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_autocomplete(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(prefix|autocomplete|partial|suffix)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("autocomplete_indicators_in_column_names")
            
#     if len(text_cols) >= 2:
#         struct_score = 0.9
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         p_col = min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
#         p_prof = profile.column_profiles[p_col]
#         if p_prof.avg_tokens < 6.0:
#             sample_score = 1.0
#             rules.append("prefix_has_very_short_token_count")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("autocomplete_prefix_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_keyword_generation(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(keywords|tags|keyphrases|labels)$", col, re.IGNORECASE):
#             col_score = max(col_score, 1.0)
#             rules.append("keyword_column_detected")
            
#     active_lists = [c for c in list_cols if c not in text_cols]
#     active_labels = [c for c in label_cols if c not in text_cols]
    
#     if len(text_cols) >= 1 and (len(active_lists) >= 1 or len(active_labels) >= 1):
#         struct_score = 1.0
#         rules.append("text_input_and_categorical_or_list_output")
        
#     text_col = text_cols[0] if text_cols else None
#     target_col = active_lists[0] if active_lists else (active_labels[0] if active_labels else None)
#     if text_col and target_col:
#         overlap_hits = 0
#         valid = 0
#         for r in sample_rows:
#             t_val = str(r.get(text_col) or "").lower()
#             kw_val = r.get(target_col)
#             if not kw_val:
#                 continue
#             valid += 1
#             if isinstance(kw_val, list):
#                 kws = [str(x).lower().strip() for x in kw_val]
#             else:
#                 kws = [x.lower().strip() for x in str(kw_val).split(",")]
#             if kws and all(kw in t_val for kw in kws if kw):
#                 overlap_hits += 1
#         if valid > 0 and overlap_hits / valid > 0.5:
#             sample_score = 1.0
#             rules.append("extracted_keywords_are_substrings_of_input")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("keyword_overlap_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# def _evaluate_title_generation(profile: DatasetProfile, text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str], sample_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str]]:
#     rules = []
#     col_score = 0.0
#     struct_score = 0.0
#     sample_score = 0.0
#     stat_score = 0.0
    
#     for col in profile.columns:
#         if re.search(r"^(title|headline|subject)$", col, re.IGNORECASE):
#             col_score = 1.0
#             rules.append("title_column_detected")
            
#     if len(text_cols) >= 2:
#         struct_score = 0.8
#         rules.append("two_text_columns_detected")
        
#     if len(text_cols) >= 2:
#         title_col = next((c for c in text_cols if re.search(r"title|headline|subject", c, re.IGNORECASE)), min(text_cols, key=lambda c: profile.column_profiles[c].avg_length))
#         text_col = next((c for c in text_cols if c != title_col), text_cols[0])
#         title_prof = profile.column_profiles[title_col]
#         text_prof = profile.column_profiles[text_col]
#         if title_prof.avg_tokens < 15.0 and text_prof.avg_length > title_prof.avg_length * 5.0:
#             sample_score = 1.0
#             rules.append("title_is_short_and_text_is_much_longer")
            
#     if sample_score > 0.8:
#         stat_score = 1.0
#         rules.append("title_generation_statistics_match")
        
#     return col_score, struct_score, sample_score, stat_score, rules

# # ---------------------------------------------------------------------------
# # Main TaskDetector Class
# # ---------------------------------------------------------------------------

# class TaskDetector:
#     """
#     Automatic NLP task detector. Inspects HuggingFace dataset and computes
#     confidence scores for 24 tasks, selecting the best fit with rule-based heuristics.
#     """

#     def detect(self, dataset: Any, cfg: DictConfig) -> Dict[str, Any]:
#         """
#         Detect NLP task info, input/label/metadata columns, and return metadata.
#         """
#         try:
#             from datasets import DatasetDict, Dataset
#             if isinstance(dataset, DatasetDict):
#                 split_name = "train" if "train" in dataset else list(dataset.keys())[0]
#                 ds = dataset[split_name]
#             else:
#                 ds = dataset
                
#             if len(ds) == 0:
#                 logger.warning("Empty dataset passed. Falling back to default task_info.")
#                 return self._fallback_result()
                
#             # Phase 1: Profiling
#             profile = _generate_dataset_profile(ds)
            
#             # Phase 2, 3 & 4: Multi-role candidate listing (extremely robust against scale)
#             text_cols = []
#             label_cols = []
#             num_cols = []
#             list_cols = []
#             metadata_cols = []
            
#             for col in profile.columns:
#                 col_prof = profile.column_profiles[col]
#                 if _is_metadata_column(col, col_prof, profile.rows):
#                     metadata_cols.append(col)
#                     continue
                    
#                 is_label_name = re.search(r"^(label|sentiment|class|category|target|intent|y)s?$", col, re.IGNORECASE) is not None
#                 is_text_name = TEXT_COLUMN_NAME_PATTERN.match(col) is not None
#                 approx_unique = col_prof.unique_ratio * min(profile.rows, 1000)
                
#                 is_likely_text = is_text_name or (col_prof.text_ratio > 0.8 and col_prof.avg_tokens > 3.0)
                
#                 # 1. Text Candidates
#                 if is_likely_text or (col_prof.text_ratio > 0.8 and col_prof.avg_tokens > 1.5):
#                     text_cols.append(col)
                    
#                 # 2. Label Candidates
#                 if is_label_name or (not is_likely_text and (approx_unique <= 200 or col_prof.unique_ratio < 0.25)):
#                     label_cols.append(col)
                    
#                 # 3. Numeric Candidates
#                 if col_prof.numeric_ratio > 0.9:
#                     num_cols.append(col)
                    
#                 # 4. List Candidates
#                 if col_prof.contains_lists or col_prof.dtype == "list":
#                     list_cols.append(col)
            
#             # Find main input column
#             input_col = None
#             if profile.columns:
#                 scored_inputs = [(col, _score_input_text_candidate(col, profile.column_profiles[col])) for col in profile.columns]
#                 scored_inputs = [si for si in scored_inputs if si[1] > 0.0]
#                 if scored_inputs:
#                     input_col = max(scored_inputs, key=lambda x: x[1])[0]
                    
#             if not input_col:
#                 non_meta_cols = [c for c in profile.columns if c not in metadata_cols]
#                 input_col = non_meta_cols[0] if non_meta_cols else profile.columns[0]
                
#             if input_col not in text_cols and input_col not in metadata_cols:
#                 text_cols.append(input_col)
                
#             # Find main label column
#             label_col = None
#             if profile.columns:
#                 scored_labels = [(col, _score_label_candidate(col, profile.column_profiles[col], input_col)) for col in profile.columns if col != input_col]
#                 scored_num_targets = [(col, _score_numeric_target(col, profile.column_profiles[col], input_col)) for col in profile.columns if col != input_col]
#                 scored_list_targets = [(col, _score_list_target(col, profile.column_profiles[col], input_col)) for col in profile.columns if col != input_col]
                
#                 all_targets = scored_labels + scored_num_targets + scored_list_targets
#                 all_targets = [t for t in all_targets if t[1] > 0.0]
#                 if all_targets:
#                     label_col = max(all_targets, key=lambda x: x[1])[0]
                    
#             # User overrides from config
#             manual_text_col = getattr(cfg.dataset, "text_column", None)
#             manual_label_col = getattr(cfg.dataset, "label_column", None)
            
#             if manual_text_col and manual_text_col in profile.columns:
#                 input_col = manual_text_col
#             if manual_label_col and manual_label_col in profile.columns:
#                 label_col = manual_label_col
                
#             # Get sample rows for the evaluators
#             sample_rows = ds.select(range(min(len(ds), 50))).to_list()
            
#             # Phase 5: Task Detection Scoring
#             task_scores = {}
#             task_matched_rules = {}
            
#             eval_args = (profile, text_cols, label_cols, num_cols, list_cols, sample_rows)
            
#             evaluators = {
#                 "classification": _evaluate_classification,
#                 "multi_label_classification": _evaluate_multi_label_classification,
#                 "regression": _evaluate_regression,
#                 "token_classification": _evaluate_token_classification,
#                 "question_answering": _evaluate_question_answering,
#                 "summarization": _evaluate_summarization,
#                 "translation": _evaluate_translation,
#                 "text_generation": _evaluate_text_generation,
#                 "instruction_tuning": _evaluate_instruction_tuning,
#                 "chat": _evaluate_chat,
#                 "retrieval": _evaluate_retrieval,
#                 "text_similarity": _evaluate_text_similarity,
#                 "ranking": _evaluate_ranking,
#                 "information_extraction": _evaluate_information_extraction,
#                 "text_pair_classification": _evaluate_text_pair_classification,
#                 "natural_language_inference": _evaluate_nli,
#                 "dialogue": _evaluate_dialogue,
#                 "grammar_correction": _evaluate_grammar_correction,
#                 "text_simplification": _evaluate_text_simplification,
#                 "style_transfer": _evaluate_style_transfer,
#                 "prompt_completion": _evaluate_prompt_completion,
#                 "autocomplete": _evaluate_autocomplete,
#                 "keyword_generation": _evaluate_keyword_generation,
#                 "title_generation": _evaluate_title_generation,
#             }
            
#             for task_name, eval_fn in evaluators.items():
#                 col_s, str_s, samp_s, stat_s, rules = eval_fn(*eval_args)
                
#                 # Critical Bug Preventions & Validation Gates
#                 # Prevent text pair classification when only one text column exists
#                 if task_name in ("text_pair_classification", "natural_language_inference", "text_similarity") and len(text_cols) < 2:
#                     col_s, str_s, samp_s, stat_s = 0.0, 0.0, 0.0, 0.0
                    
#                 # Lower scores if label-based tasks are missing valid target labels
#                 if task_name in ("classification", "multi_label_classification", "regression", "text_pair_classification", "natural_language_inference") and not label_col:
#                     str_s *= 0.1
#                     samp_s *= 0.1
                    
#                 score = 0.20 * col_s + 0.30 * str_s + 0.30 * samp_s + 0.20 * stat_s
#                 task_scores[task_name] = score
#                 task_matched_rules[task_name] = rules
                
#             # Rank top-3 tasks
#             sorted_tasks = sorted(task_scores.items(), key=lambda x: x[1], reverse=True)
#             top_task, top_confidence = sorted_tasks[0]
            
#             alternative_tasks = [t for t, s in sorted_tasks[1:] if s > 0.1][:2]
            
#             result = {
#                 "task": top_task,
#                 "confidence": round(float(top_confidence), 4),
#                 "matched_rules": task_matched_rules[top_task],
#                 "alternative_tasks": alternative_tasks,
#                 "input_column": input_col,
#                 "label_column": label_col,
#                 "metadata_columns": [], # Filled below
#             }
            
#             # Legacy/downstream compatibility keys
#             result["sub_task"] = top_task
#             result["text_column"] = input_col
#             result["output_column"] = label_col if top_task not in ("translation", "summarization", "instruction_tuning") else None
#             result["instruction_column"] = None
#             result["context_column"] = None
            
#             if top_task == "question_answering":
#                 q_cols = [c for c in text_cols if re.search(r"question|query", c, re.IGNORECASE)]
#                 a_cols = [c for c in text_cols if re.search(r"answer", c, re.IGNORECASE)]
#                 c_cols = [c for c in text_cols if re.search(r"context|passage", c, re.IGNORECASE)]
#                 result["input_column"] = q_cols[0] if q_cols else input_col
#                 result["output_column"] = a_cols[0] if a_cols else (text_cols[1] if len(text_cols) > 1 else None)
#                 result["context_column"] = c_cols[0] if c_cols else (text_cols[2] if len(text_cols) > 2 else None)
#             elif top_task in ("summarization", "translation", "grammar_correction", "text_simplification", "style_transfer", "prompt_completion", "autocomplete"):
#                 result["input_column"] = input_col
#                 other_text = [c for c in text_cols if c != input_col]
#                 result["output_column"] = other_text[0] if other_text else None
#             elif top_task == "instruction_tuning":
#                 inst_cols = [c for c in text_cols if re.search(r"instruction|prompt", c, re.IGNORECASE)]
#                 out_cols = [c for c in text_cols if re.search(r"output|response|completion", c, re.IGNORECASE)]
#                 result["instruction_column"] = inst_cols[0] if inst_cols else input_col
#                 result["output_column"] = out_cols[0] if out_cols else (text_cols[1] if len(text_cols) > 1 else None)
#             elif top_task == "chat":
#                 result["text_column"] = input_col
                
#             if top_task in ("classification", "multi_label_classification", "token_classification"):
#                 self._enrich_classification_info(ds, result)
#                 if top_task == "classification":
#                     n = result.get("num_labels")
#                     result["problem_type"] = "single_label_classification"
#                     result["sub_task"] = "binary_classification" if n == 2 else "multi_class_classification"
#                 elif top_task == "multi_label_classification":
#                     result["problem_type"] = "multi_label_classification"
#             elif top_task == "regression":
#                 result["num_labels"] = 1
#                 result["problem_type"] = "regression"
                
#             result["detected_columns"] = {
#                 "text": text_cols,
#                 "label": label_cols,
#                 "all": profile.columns,
#             }
            
#             # Compute metadata_columns cleanly
#             exclude_cols = {input_col, label_col}
#             if result.get("context_column"):
#                 exclude_cols.add(result["context_column"])
#             if result.get("instruction_column"):
#                 exclude_cols.add(result["instruction_column"])
#             if result.get("output_column"):
#                 exclude_cols.add(result["output_column"])
                
#             result["metadata_columns"] = [c for c in profile.columns if c not in exclude_cols]
            
#             # Override if forced task via config
#             forced_task = getattr(cfg.dataset, "task", None) or getattr(getattr(cfg, "task", None), "name", None)
#             if forced_task and forced_task != "auto":
#                 logger.info(f"Task forced by config: {forced_task}")
#                 result["task"] = forced_task
#                 result["sub_task"] = forced_task
                
#             return result
            
#         except Exception as e:
#             logger.exception(f"Error during Task Detection: {e}")
#             return self._fallback_result()

#     def _enrich_classification_info(self, split: Any, task_info: dict[str, Any]) -> None:
#         """Populate num_labels, label2id, id2label for classification datasets."""
#         label_col = task_info.get("label_column")
#         if not label_col:
#             return
#         features = split.features
#         if label_col in features:
#             feat = features[label_col]
#             from datasets.features import ClassLabel, Sequence
#             if isinstance(feat, ClassLabel):
#                 names = feat.names
#                 task_info["num_labels"] = len(names)
#                 task_info["label2id"]   = {n: i for i, n in enumerate(names)}
#                 task_info["id2label"]   = {i: n for i, n in enumerate(names)}
#                 return
#             elif isinstance(feat, Sequence) and isinstance(feat.feature, ClassLabel):
#                 names = feat.feature.names
#                 task_info["num_labels"] = len(names)
#                 task_info["label2id"]   = {n: i for i, n in enumerate(names)}
#                 task_info["id2label"]   = {i: n for i, n in enumerate(names)}
#                 return
                
#         try:
#             vals = split[label_col]
#             flat_vals = []
#             for v in vals:
#                 if isinstance(v, (list, tuple)):
#                     flat_vals.extend(v)
#                 else:
#                     flat_vals.append(v)
#             unique_labels = sorted(list(set(str(x) for x in flat_vals if x is not None)))
#             task_info["num_labels"] = len(unique_labels)
#             task_info["label2id"]   = {l: i for i, l in enumerate(unique_labels)}
#             task_info["id2label"]   = {i: l for i, l in enumerate(unique_labels)}
#         except Exception as e:
#             logger.warning(f"Failed to enrich classification info: {e}")

#     def _fallback_result(self) -> Dict[str, Any]:
#         """Return safe fallback result (causal_lm/text_generation) upon error or empty dataset."""
#         return {
#             "task": "causal_lm",
#             "confidence": 0.0,
#             "matched_rules": ["fallback_to_causal_lm"],
#             "alternative_tasks": [],
#             "input_column": "text",
#             "label_column": None,
#             "metadata_columns": [],
#             "sub_task": "text_generation",
#             "text_column": "text",
#             "num_labels": None,
#             "label2id": None,
#             "id2label": None,
#             "problem_type": None,
#             "detected_columns": {"text": [], "label": [], "all": []}
#         }

"""
Automatic NLP + Tabular task detection from dataset schema and content.
Robust rule-based heuristics with proper scoring and priority resolution.
"""
from __future__ import annotations

import re
import json
import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from omegaconf import DictConfig

from src.utils.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
QUESTION_PATTERN = re.compile(
    r"^(who|what|where|when|why|how|whose|which|whom|is|are|can|do|does|did|will|should|would|could)\b",
    re.IGNORECASE,
)
METADATA_NAME_PATTERN = re.compile(
    r"^(id|uuid|index|timestamp|date|created_at|updated_at|row_id|sample_id|time|idx|_id|serial)$"
    r"|_id$|^id_|^idx$|^index$",
    re.IGNORECASE,
)
TEXT_COLUMN_NAME_PATTERN = re.compile(
    r"^(text|sentence|document|article|summary|abstract|highlights|question|context|passage|"
    r"answer|incorrect|corrected|original|simple|complex|formal|informal|prompt|completion|"
    r"prefix|suffix|message|conversation|query|dialogue|translation|body|content|utterance|"
    r"review|comment|description|premise|hypothesis)s?$",
    re.IGNORECASE,
)

# Tasks that clearly win over others when their signals are strong
# Higher index = higher priority in tie-breaking
TASK_PRIORITY: Dict[str, int] = {
    "chat": 100,
    "token_classification": 90,
    "natural_language_inference": 85,
    "question_answering": 80,
    "translation": 75,
    "summarization": 70,
    "instruction_tuning": 65,
    "information_extraction": 60,
    "tabular_classification": 55,
    "tabular_regression": 50,
    "text_pair_classification": 45,
    "multi_label_classification": 40,
    "classification": 38,
    "regression": 35,
    "ranking": 30,
    "retrieval": 28,
    "text_similarity": 25,
    "dialogue": 20,
    "grammar_correction": 18,
    "text_simplification": 16,
    "style_transfer": 14,
    "keyword_generation": 12,
    "title_generation": 10,
    "prompt_completion": 8,
    "autocomplete": 6,
    "text_generation": 4,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    dtype: str                  # string | int | float | bool | list | dict | unknown
    unique_ratio: float         # unique values / total non-null values
    null_ratio: float
    avg_length: float           # avg char length (for string cols)
    avg_tokens: float           # avg whitespace token count
    num_unique: int             # absolute count of unique values
    contains_urls: bool
    contains_questions: bool
    contains_json: bool
    contains_lists: bool
    contains_entities: bool
    numeric_ratio: float        # fraction of values parseable as float
    text_ratio: float           # fraction of values that are strings
    is_boolean_like: bool       # 2 unique values, 0/1 or True/False

@dataclass
class DatasetProfile:
    rows: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_ratios: Dict[str, float]
    unique_ratios: Dict[str, float]
    avg_text_lengths: Dict[str, float]
    avg_token_counts: Dict[str, float]
    numeric_distributions: Dict[str, Dict[str, float]]
    categorical_distributions: Dict[str, Dict[str, int]]
    list_columns: List[str]
    dict_columns: List[str]
    nested_structures: Dict[str, str]
    column_profiles: Dict[str, ColumnProfile]
    # Derived helpers
    num_text_like_cols: int = 0
    num_numeric_cols: int = 0
    num_categorical_cols: int = 0

# ---------------------------------------------------------------------------
# Profiling Helpers
# ---------------------------------------------------------------------------

def _get_column_values(ds: Any, col_name: str, limit: int = 1000) -> List[Any]:
    try:
        return ds[:limit][col_name]
    except Exception as e:
        logger.warning(f"Error getting values for column '{col_name}': {e}")
        return []

def _looks_like_json(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            json.loads(s)
            return True
        except ValueError:
            pass
    return False

def _looks_like_list(val: Any) -> bool:
    if isinstance(val, (list, tuple, set)):
        return True
    if not isinstance(val, str):
        return False
    s = val.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            return isinstance(parsed, list)
        except ValueError:
            pass
        if "," in s:
            return True
    return False

def _looks_like_entity(val: Any) -> bool:
    if isinstance(val, dict):
        if len(set(val.keys()) & {"entity", "label", "start", "end"}) >= 2:
            return True
    elif isinstance(val, list) and len(val) > 0:
        sample = val[:5]
        if all(isinstance(x, dict) and len(set(x.keys()) & {"entity", "label", "start", "end"}) >= 2 for x in sample):
            return True
        if all(isinstance(x, str) and (x == "O" or re.match(r"^[BIO]-\w+$", x)) for x in val[:10]):
            return True
    return False

def _string_similarity(s1: str, s2: str) -> float:
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def _generate_dataset_profile(ds: Any, limit: int = 1000) -> DatasetProfile:
    rows = len(ds)
    columns = ds.column_names

    dtypes: Dict[str, str] = {}
    null_ratios: Dict[str, float] = {}
    unique_ratios: Dict[str, float] = {}
    avg_text_lengths: Dict[str, float] = {}
    avg_token_counts: Dict[str, float] = {}
    numeric_distributions: Dict[str, Dict[str, float]] = {}
    categorical_distributions: Dict[str, Dict[str, int]] = {}
    list_columns: List[str] = []
    dict_columns: List[str] = []
    nested_structures: Dict[str, str] = {}
    column_profiles: Dict[str, ColumnProfile] = {}

    try:
        from datasets.features import Value, ClassLabel, Sequence
    except ImportError:
        Value = ClassLabel = Sequence = None

    for col in columns:
        vals = _get_column_values(ds, col, limit=limit)
        if not vals:
            column_profiles[col] = ColumnProfile(
                dtype="unknown", unique_ratio=0, null_ratio=1, avg_length=0,
                avg_tokens=0, num_unique=0, contains_urls=False, contains_questions=False,
                contains_json=False, contains_lists=False, contains_entities=False,
                numeric_ratio=0, text_ratio=0, is_boolean_like=False,
            )
            dtypes[col] = "unknown"
            null_ratios[col] = 1.0
            unique_ratios[col] = 0.0
            avg_text_lengths[col] = 0.0
            avg_token_counts[col] = 0.0
            continue

        # Detect nesting from HF features
        is_list = False
        is_dict = False
        nested_desc = "scalar"
        if Sequence is not None:
            feat = ds.features.get(col)
            if isinstance(feat, Sequence):
                is_list = True
                nested_desc = f"list[{feat.feature}]"
            elif isinstance(feat, dict):
                is_dict = True
                nested_desc = "dict"

        # Separate null and non-null
        non_null_vals = []
        null_count = 0
        for v in vals:
            if v is None:
                null_count += 1
            elif isinstance(v, str) and v.lower() in {"", "nan", "null", "none", "n/a", "na"}:
                null_count += 1
            else:
                non_null_vals.append(v)

        # Runtime list/dict detection
        if not is_list and any(isinstance(v, (list, tuple)) for v in non_null_vals):
            is_list = True
            nested_desc = "list"
        if not is_dict and any(isinstance(v, dict) for v in non_null_vals):
            is_dict = True
            nested_desc = "dict"

        if is_list:
            list_columns.append(col)
        if is_dict:
            dict_columns.append(col)
        nested_structures[col] = nested_desc

        # Flatten lists for analysis
        flat_vals: List[Any] = []
        if is_list:
            for v in non_null_vals:
                if isinstance(v, (list, tuple)):
                    flat_vals.extend(v)
                elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                    try:
                        parsed = json.loads(v)
                        flat_vals.extend(parsed) if isinstance(parsed, list) else flat_vals.append(v)
                    except ValueError:
                        flat_vals.append(v)
                else:
                    flat_vals.append(v)
        else:
            flat_vals = non_null_vals

        flat_vals = [f for f in flat_vals if f is not None]

        str_vals = [v for v in flat_vals if isinstance(v, str)]
        text_ratio = len(str_vals) / len(flat_vals) if flat_vals else 0.0

        num_vals: List[float] = []
        for v in flat_vals:
            if isinstance(v, bool):
                pass  # booleans are ints in Python but we treat them separately
            elif isinstance(v, (int, float)):
                num_vals.append(float(v))
            elif isinstance(v, str):
                try:
                    num_vals.append(float(v))
                except ValueError:
                    pass
        numeric_ratio = len(num_vals) / len(flat_vals) if flat_vals else 0.0

        lengths = [len(v) for v in str_vals]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0

        token_counts = [len(v.split()) for v in str_vals]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

        null_ratio = null_count / len(vals) if vals else 1.0
        unique_set = set(str(v) for v in non_null_vals)
        unique_ratio = len(unique_set) / len(non_null_vals) if non_null_vals else 0.0
        num_unique = len(unique_set)

        # Boolean-like: exactly 2 unique values (could be 0/1, True/False, yes/no)
        is_boolean_like = (num_unique == 2 and not is_list and not is_dict)

        contains_urls = any(URL_PATTERN.search(v) for v in str_vals)
        contains_questions = any("?" in v or QUESTION_PATTERN.match(v) for v in str_vals)
        contains_json = any(_looks_like_json(v) for v in non_null_vals)
        contains_lists = is_list or any(_looks_like_list(v) for v in non_null_vals)
        contains_entities = (
            any(_looks_like_entity(v) for v in non_null_vals)
            or (is_list and any(_looks_like_entity(v) for v in flat_vals))
        )

        # Infer dtype
        if is_list:
            dtype = "list"
        elif is_dict:
            dtype = "dict"
        elif all(isinstance(v, bool) for v in non_null_vals):
            dtype = "bool"
        elif numeric_ratio > 0.85:
            dtype = "int" if all(float(v).is_integer() for v in num_vals) else "float"
        elif text_ratio > 0.75:
            dtype = "string"
        else:
            dtype = "unknown"

        dtypes[col] = dtype
        null_ratios[col] = null_ratio
        unique_ratios[col] = unique_ratio
        avg_text_lengths[col] = avg_length
        avg_token_counts[col] = avg_tokens

        if num_vals:
            numeric_distributions[col] = {
                "min": min(num_vals),
                "max": max(num_vals),
                "mean": sum(num_vals) / len(num_vals),
            }

        if num_unique <= 500 and non_null_vals:
            counts: Dict[str, int] = {}
            for v in non_null_vals:
                k = str(v)
                counts[k] = counts.get(k, 0) + 1
            categorical_distributions[col] = counts

        column_profiles[col] = ColumnProfile(
            dtype=dtype,
            unique_ratio=unique_ratio,
            null_ratio=null_ratio,
            avg_length=avg_length,
            avg_tokens=avg_tokens,
            num_unique=num_unique,
            contains_urls=contains_urls,
            contains_questions=contains_questions,
            contains_json=contains_json,
            contains_lists=contains_lists,
            contains_entities=contains_entities,
            numeric_ratio=numeric_ratio,
            text_ratio=text_ratio,
            is_boolean_like=is_boolean_like,
        )

    # Derived stats
    num_text_like = sum(1 for c in columns if column_profiles[c].dtype == "string" and column_profiles[c].avg_tokens > 1.5)
    num_numeric = sum(1 for c in columns if column_profiles[c].dtype in ("int", "float") and not METADATA_NAME_PATTERN.search(c))
    num_categorical = sum(1 for c in columns if column_profiles[c].num_unique <= 200 and column_profiles[c].dtype in ("string", "int", "bool"))

    profile = DatasetProfile(
        rows=rows,
        columns=columns,
        dtypes=dtypes,
        null_ratios=null_ratios,
        unique_ratios=unique_ratios,
        avg_text_lengths=avg_text_lengths,
        avg_token_counts=avg_token_counts,
        numeric_distributions=numeric_distributions,
        categorical_distributions=categorical_distributions,
        list_columns=list_columns,
        dict_columns=dict_columns,
        nested_structures=nested_structures,
        column_profiles=column_profiles,
        num_text_like_cols=num_text_like,
        num_numeric_cols=num_numeric,
        num_categorical_cols=num_categorical,
    )
    return profile

# ---------------------------------------------------------------------------
# Column Role Detection
# ---------------------------------------------------------------------------

def _is_metadata_column(col_name: str, profile: ColumnProfile, rows_count: int) -> bool:
    if METADATA_NAME_PATTERN.search(col_name):
        return True
    if profile.unique_ratio == 1.0 and rows_count > 5 and profile.dtype in ("int", "string") and profile.avg_length < 40:
        return True
    if profile.null_ratio > 0.95:
        return True
    return False

def _is_tabular_feature(col_name: str, profile: ColumnProfile) -> bool:
    """A column is a tabular feature if it is numeric or low-cardinality categorical (not free text)."""
    if METADATA_NAME_PATTERN.search(col_name):
        return False
    if profile.dtype in ("int", "float") and profile.avg_tokens < 2:
        return True
    if profile.dtype in ("string", "bool") and profile.num_unique <= 50 and profile.avg_tokens <= 3:
        return True
    return False

def _score_text_input(col_name: str, prof: ColumnProfile) -> float:
    """Score a column as a primary text input (higher = more text-like)."""
    if METADATA_NAME_PATTERN.search(col_name):
        return 0.0
    if prof.dtype not in ("string", "list") and prof.text_ratio < 0.5:
        return 0.0

    score = 0.0
    score += min(prof.avg_length / 200.0, 1.0) * 0.35
    score += min(prof.avg_tokens / 50.0, 1.0) * 0.30
    score += min(prof.unique_ratio, 1.0) * 0.15

    if prof.dtype == "string":
        score += 0.15
    if TEXT_COLUMN_NAME_PATTERN.match(col_name):
        score += 0.15

    # Penalize short columns that look like labels
    if prof.avg_tokens <= 2 and prof.num_unique <= 100:
        score *= 0.3

    return min(score, 1.0)

def _score_label_candidate(col_name: str, prof: ColumnProfile, input_col: str) -> float:
    if col_name == input_col or METADATA_NAME_PATTERN.search(col_name):
        return 0.0

    score = 0.0
    if 0.0 < prof.unique_ratio < 0.15:
        score += 0.35
    elif prof.unique_ratio < 0.40:
        score += 0.15

    if prof.avg_tokens <= 3:
        score += 0.25
    if prof.dtype in ("string", "int", "bool"):
        score += 0.15
    if re.search(r"^(label|sentiment|target|class|category|tag|output|intent|y|split|fold)s?$", col_name, re.IGNORECASE):
        score += 0.35

    # Penalize obvious text columns
    if TEXT_COLUMN_NAME_PATTERN.match(col_name) and prof.avg_tokens > 5:
        score *= 0.1

    return min(score, 1.0)

def _score_numeric_target(col_name: str, prof: ColumnProfile, input_col: str) -> float:
    if col_name == input_col or METADATA_NAME_PATTERN.search(col_name):
        return 0.0

    score = 0.0
    if prof.numeric_ratio > 0.9 and prof.unique_ratio > 0.01:
        score += 0.5
        if prof.dtype == "float":
            score += 0.25
        if re.search(r"^(score|rating|value|target|loss|price|amount|count|age|weight|height)s?$", col_name, re.IGNORECASE):
            score += 0.25
    return min(score, 1.0)

# ---------------------------------------------------------------------------
# Task Evaluators  (returns: col_score, struct_score, sample_score, stat_score, rules)
# Score range: 0.0 - 1.0 each
# Final = 0.20*col + 0.30*struct + 0.30*sample + 0.20*stat
# ---------------------------------------------------------------------------

TaskScore = Tuple[float, float, float, float, List[str]]

def _eval_tabular_classification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], tabular_feat_cols: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    if not tabular_feat_cols:
        return col_s, struct_s, samp_s, stat_s, rules

    # Need at least 2 tabular feature cols + 1 label
    non_label_feats = [c for c in tabular_feat_cols if c not in label_cols]
    if len(non_label_feats) >= 2:
        struct_s = 1.0
        rules.append("multiple_tabular_feature_columns")

    has_cat_label = any(
        re.search(r"^(label|target|class|category|y|output)s?$", c, re.IGNORECASE)
        for c in profile.columns
        if profile.column_profiles[c].num_unique <= 200
    )
    if has_cat_label:
        col_s = 1.0
        rules.append("categorical_label_column_found")

    # No long text columns → strong tabular signal
    if len(text_cols) == 0:
        struct_s = min(struct_s + 0.3, 1.0)
        rules.append("no_free_text_columns")

    label_col = next(
        (c for c in profile.columns if re.search(r"^(label|target|class|y)s?$", c, re.IGNORECASE)
         and profile.column_profiles[c].num_unique <= 200), None
    )
    if label_col and sample_rows:
        vals = [r.get(label_col) for r in sample_rows if r.get(label_col) is not None]
        if vals and all(isinstance(v, (int, str, bool)) and len(str(v)) < 30 for v in vals):
            samp_s = 1.0
            rules.append("label_values_are_short_categories")

    if label_col and label_col in profile.categorical_distributions:
        n_unique = len(profile.categorical_distributions[label_col])
        if 2 <= n_unique <= 100:
            stat_s = 1.0
            rules.append(f"label_has_{n_unique}_unique_classes")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_tabular_regression(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], tabular_feat_cols: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    if not tabular_feat_cols:
        return col_s, struct_s, samp_s, stat_s, rules

    non_meta_num = [c for c in num_cols if not METADATA_NAME_PATTERN.search(c)]
    if len(non_meta_num) >= 2 and len(text_cols) == 0:
        struct_s = 1.0
        rules.append("multiple_numeric_cols_no_text")

    target_col = next(
        (c for c in non_meta_num
         if re.search(r"^(score|rating|value|target|price|amount|loss|y)s?$", c, re.IGNORECASE)), None
    ) or (non_meta_num[-1] if non_meta_num else None)

    if target_col:
        col_s = 0.8
        rules.append(f"numeric_target_col_{target_col}")

    if target_col and sample_rows:
        vals = [r.get(target_col) for r in sample_rows if r.get(target_col) is not None]
        if vals and any(isinstance(v, float) for v in vals):
            samp_s = 1.0
            rules.append("float_target_values_found")
        elif vals and all(isinstance(v, (int, float)) for v in vals):
            samp_s = 0.7
            rules.append("all_numeric_target_values")

    if target_col and target_col in profile.column_profiles:
        if profile.column_profiles[target_col].unique_ratio > 0.05:
            stat_s = 1.0
            rules.append("continuous_target_distribution")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_classification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    active_labels = [c for c in label_cols if c not in text_cols]

    for c in profile.columns:
        if c not in text_cols and re.search(r"^(label|sentiment|class|category|target|intent|y)s?$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("column_name_is_classification_label")

    if len(text_cols) >= 1 and len(active_labels) >= 1:
        struct_s = 1.0 if len(text_cols) == 1 else 0.6
        rules.append("text_input_and_categorical_label")
    elif len(text_cols) >= 1 and any(
        profile.column_profiles[c].num_unique <= 30
        for c in num_cols if c not in text_cols
    ):
        struct_s = 0.6
        rules.append("text_input_numeric_categorical_label")

    best_label = active_labels[0] if active_labels else None
    if best_label and sample_rows:
        vals = [r.get(best_label) for r in sample_rows if r.get(best_label) is not None]
        if vals and all(isinstance(v, (int, str, bool)) and len(str(v)) < 30 for v in vals):
            samp_s = 1.0
            rules.append("label_values_are_short")

    if best_label and best_label in profile.categorical_distributions:
        n = len(profile.categorical_distributions[best_label])
        if 2 <= n <= 200:
            stat_s = 1.0
            rules.append(f"label_cardinality_{n}")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_multi_label_classification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    active_lists = [c for c in list_cols if c not in text_cols]

    for c in profile.columns:
        if c not in text_cols and re.search(r"^(labels|tags|genres|categories|targets)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("multi_label_column_name")

    if len(text_cols) >= 1 and active_lists:
        struct_s = 1.0
        rules.append("text_and_list_target")

    best_list = active_lists[0] if active_lists else None
    if best_list and sample_rows:
        vals = [r.get(best_list) for r in sample_rows if r.get(best_list) is not None]
        if vals:
            if any(isinstance(v, (list, tuple)) for v in vals):
                samp_s = 1.0
                rules.append("list_label_values")
            elif any(isinstance(v, str) and "," in v for v in vals):
                samp_s = 0.8
                rules.append("comma_separated_labels")

    if best_list and profile.column_profiles[best_list].contains_lists:
        stat_s = 1.0
        rules.append("target_profile_is_list")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_regression(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    non_meta_num = [c for c in num_cols if not METADATA_NAME_PATTERN.search(c) and c not in text_cols]

    for c in profile.columns:
        if c not in text_cols and not METADATA_NAME_PATTERN.search(c):
            if re.search(r"^(score|rating|value|target|loss|y)s?$", c, re.IGNORECASE):
                col_s = 1.0
                rules.append("regression_target_col_name")

    if len(text_cols) >= 1 and non_meta_num:
        struct_s = 1.0
        rules.append("text_input_numeric_target")

    best_num = non_meta_num[0] if non_meta_num else None
    if best_num and sample_rows:
        vals = [r.get(best_num) for r in sample_rows if r.get(best_num) is not None]
        if vals:
            samp_s = 1.0 if any(isinstance(v, float) for v in vals) else 0.7
            rules.append("float_or_numeric_target_values")

    if best_num and profile.column_profiles[best_num].unique_ratio > 0.05:
        stat_s = 1.0
        rules.append("continuous_distribution")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_token_classification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(tokens|words|ner_tags|pos_tags|chunk_tags|tags)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("token_classification_column_name")

    if len(list_cols) >= 2:
        struct_s = 1.0
        rules.append("two_list_columns")

    if len(list_cols) >= 2:
        matched = True
        for row in sample_rows[:10]:
            v1 = row.get(list_cols[0])
            v2 = row.get(list_cols[1])
            if not (isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)) and len(v1) == len(v2)):
                matched = False
                break
        if matched:
            samp_s = 1.0
            rules.append("parallel_token_tag_lists_same_length")

    for c in list_cols:
        if profile.column_profiles[c].contains_entities:
            stat_s = 1.0
            rules.append("entity_tag_profile_detected")
            break
    else:
        if len(list_cols) >= 2 and struct_s == 1.0:
            stat_s = 0.7
            rules.append("aligned_sequence_columns")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_question_answering(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_q = any(re.search(r"^(question|query|q)$", c, re.IGNORECASE) for c in profile.columns)
    has_a = any(re.search(r"^(answer|answers|a)$", c, re.IGNORECASE) for c in profile.columns)
    has_c = any(re.search(r"^(context|passage|document)$", c, re.IGNORECASE) for c in profile.columns)

    if has_q and has_a:
        col_s = 1.0
        rules.append("question_and_answer_columns")
    elif has_q or has_a:
        col_s = 0.5

    if has_c and (has_q or has_a):
        col_s = min(col_s + 0.2, 1.0)
        rules.append("context_column_also_present")

    if len(text_cols) >= 3:
        struct_s = 1.0
        rules.append("three_text_columns")
    elif len(text_cols) == 2:
        struct_s = 0.7
        rules.append("two_text_columns_for_qa")

    q_col = next((c for c in text_cols if re.search(r"question|query", c, re.IGNORECASE)), None)
    c_col = next((c for c in text_cols if re.search(r"context|passage", c, re.IGNORECASE)), None)
    a_col = next((c for c in text_cols if re.search(r"answer", c, re.IGNORECASE)), None)

    if q_col and profile.column_profiles[q_col].contains_questions:
        samp_s = 0.8
        rules.append("question_col_contains_questions")

    if c_col and a_col and sample_rows:
        hits = sum(
            1 for r in sample_rows
            if isinstance(r.get(c_col), str) and isinstance(r.get(a_col), str)
            and str(r.get(a_col)).lower() in str(r.get(c_col)).lower()
        )
        if hits / max(len(sample_rows), 1) > 0.35:
            samp_s = 1.0
            rules.append("answers_are_substrings_of_context")

    if q_col and c_col:
        q_len = profile.column_profiles[q_col].avg_length
        c_len = profile.column_profiles[c_col].avg_length
        if q_len < c_len * 0.5:
            stat_s = 1.0
            rules.append("question_shorter_than_context")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_summarization(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_src = any(re.search(r"^(document|article|text|body|passage)$", c, re.IGNORECASE) for c in profile.columns)
    has_tgt = any(re.search(r"^(summary|highlights|abstract)$", c, re.IGNORECASE) for c in profile.columns)

    if has_src and has_tgt:
        col_s = 1.0
        rules.append("source_and_summary_column_names")
    elif has_tgt:
        col_s = 0.6
        rules.append("summary_column_name")

    if len(text_cols) >= 2:
        struct_s = 1.0
        rules.append("two_text_columns")

    if len(text_cols) >= 2:
        long_c = max(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        short_c = min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        if long_c != short_c:
            ratios = [
                len(str(r.get(long_c) or "")) > len(str(r.get(short_c) or "")) * 1.5
                for r in sample_rows
                if r.get(short_c)
            ]
            if ratios and sum(ratios) / len(ratios) > 0.75:
                samp_s = 1.0
                rules.append("source_consistently_longer_than_summary")

    if len(text_cols) >= 2:
        long_c = max(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        short_c = min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        if profile.column_profiles[short_c].avg_length > 0:
            ratio = profile.column_profiles[long_c].avg_length / profile.column_profiles[short_c].avg_length
            if 2.0 <= ratio <= 50.0:
                stat_s = 1.0
                rules.append(f"compression_ratio_{ratio:.1f}x")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_translation(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    lang_codes = {"en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi", "nl", "tr", "pl", "sv"}
    has_translation_col = "translation" in profile.columns
    lang_col_matches = [c for c in profile.columns if c.lower() in lang_codes or re.search(r"_[a-z]{2}$|^[a-z]{2}_", c)]

    if has_translation_col:
        col_s = 1.0
        rules.append("translation_column")
    elif len(lang_col_matches) >= 2:
        col_s = 0.9
        rules.append("language_code_columns")

    if has_translation_col and "translation" in profile.dict_columns:
        struct_s = 1.0
        rules.append("translation_is_dict")
    elif len(text_cols) >= 2:
        struct_s = 0.8
        rules.append("two_text_columns")

    if has_translation_col and sample_rows:
        vals = [r.get("translation") for r in sample_rows if isinstance(r.get("translation"), dict)]
        if vals and any(len(v) >= 2 for v in vals):
            samp_s = 1.0
            rules.append("translation_dict_has_multiple_keys")

    if len(text_cols) >= 2:
        t1, t2 = text_cols[0], text_cols[1]
        words1 = set(" ".join(str(r.get(t1) or "") for r in sample_rows).lower().split())
        words2 = set(" ".join(str(r.get(t2) or "") for r in sample_rows).lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            if overlap < 0.15:
                stat_s = 1.0
                rules.append(f"low_word_overlap_{overlap:.2f}")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_text_generation(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(text|generation|story|output|response|document)s?$", c, re.IGNORECASE):
            col_s = max(col_s, 0.7)
            rules.append("text_generation_column_name")

    if len(text_cols) == 1:
        struct_s = 0.8
        rules.append("single_text_column")

    t = text_cols[0] if text_cols else None
    if t and sample_rows:
        vals = [str(r.get(t) or "") for r in sample_rows]
        if vals and all(len(v) > 30 for v in vals if v):
            samp_s = 0.8
            rules.append("long_free_text_samples")

    if t and profile.column_profiles[t].unique_ratio > 0.8 and profile.column_profiles[t].avg_tokens > 10:
        stat_s = 0.8
        rules.append("high_uniqueness_long_tokens")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_instruction_tuning(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_inst = any(re.search(r"^(instruction|prompt)$", c, re.IGNORECASE) for c in profile.columns)
    has_out = any(re.search(r"^(output|response|completion)$", c, re.IGNORECASE) for c in profile.columns)

    if has_inst and has_out:
        col_s = 1.0
        rules.append("instruction_and_output_columns")
    elif has_inst:
        col_s = 0.6

    if len(text_cols) >= 2:
        struct_s = 1.0
        rules.append("two_text_columns")

    inst_col = next((c for c in text_cols if re.search(r"instruction|prompt", c, re.IGNORECASE)), text_cols[0] if text_cols else None)
    if inst_col and sample_rows:
        commands = {"write", "explain", "create", "how", "what", "translate", "summarize", "list", "generate", "describe", "compare"}
        vals = [str(r.get(inst_col) or "").lower() for r in sample_rows]
        hits = sum(any(v.startswith(cmd) for cmd in commands) for v in vals if v)
        if vals and hits / len(vals) > 0.2:
            samp_s = 1.0
            rules.append("instruction_starts_with_command_verbs")

    if struct_s >= 1.0:
        stat_s = 0.8
        rules.append("instruction_tuning_statistics_match")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_chat(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    chat_col = None
    for c in profile.columns:
        if re.search(r"^(messages|conversations|chat|dialog|dialogue)$", c, re.IGNORECASE):
            col_s = 1.0
            chat_col = c
            rules.append(f"chat_column_{c}")

    if chat_col and (chat_col in profile.list_columns or chat_col in profile.dict_columns):
        struct_s = 1.0
        rules.append("chat_column_nested_structure")

    if chat_col and sample_rows:
        vals = [r.get(chat_col) for r in sample_rows if r.get(chat_col) is not None]
        if vals:
            is_valid = all(
                isinstance(v, list) and v and isinstance(v[0], dict) and ("role" in v[0] or "from" in v[0])
                for v in vals[:5]
            )
            if is_valid:
                samp_s = 1.0
                rules.append("chat_messages_have_role_field")

    if samp_s == 1.0:
        stat_s = 1.0
        rules.append("chat_structure_confirmed")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_retrieval(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_q = any(re.search(r"^(query|question|search_query)$", c, re.IGNORECASE) for c in profile.columns)
    has_d = any(re.search(r"^(document|passage|context|positive|negative)$", c, re.IGNORECASE) for c in profile.columns)

    if has_q and has_d:
        col_s = 1.0
        rules.append("query_and_document_columns")
    elif has_q:
        col_s = 0.5

    if len(text_cols) >= 2:
        struct_s = 0.9 if len(text_cols) == 2 else 0.6
        rules.append(f"{len(text_cols)}_text_columns_for_retrieval")

    q_cols = [c for c in text_cols if re.search(r"query|question", c, re.IGNORECASE)]
    d_cols = [c for c in text_cols if c not in q_cols]
    if q_cols and d_cols:
        q_len = profile.column_profiles[q_cols[0]].avg_length
        d_len = profile.column_profiles[d_cols[0]].avg_length
        if q_len < d_len * 0.4:
            samp_s = 1.0
            rules.append("query_much_shorter_than_document")

    if samp_s == 1.0:
        stat_s = 1.0
        rules.append("retrieval_asymmetry_confirmed")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_text_similarity(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_pair = any(re.search(r"^(sentence[12]|text[12]|s[12])$", c, re.IGNORECASE) for c in profile.columns)
    has_score = any(re.search(r"^(similarity|score|sim_score)$", c, re.IGNORECASE) for c in profile.columns)

    if has_pair and has_score:
        col_s = 1.0
        rules.append("similarity_pair_and_score_columns")
    elif has_pair:
        col_s = 0.6

    non_meta_num = [c for c in num_cols if not METADATA_NAME_PATTERN.search(c) and c not in text_cols]
    if len(text_cols) >= 2 and non_meta_num:
        struct_s = 1.0
        rules.append("two_text_inputs_and_numeric_score")

    if len(text_cols) >= 2:
        p1, p2 = profile.column_profiles[text_cols[0]], profile.column_profiles[text_cols[1]]
        if abs(p1.avg_length - p2.avg_length) / max(p1.avg_length, p2.avg_length, 1) < 0.3:
            samp_s = 0.9
            rules.append("similar_average_lengths")

    if non_meta_num and profile.column_profiles[non_meta_num[0]].unique_ratio > 0.05:
        stat_s = 1.0
        rules.append("continuous_score_distribution")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_ranking(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_pos = any(re.search(r"^(positive|pos|anchor)$", c, re.IGNORECASE) for c in profile.columns)
    has_neg = any(re.search(r"^(negative|neg)$", c, re.IGNORECASE) for c in profile.columns)

    if has_pos and has_neg:
        col_s = 1.0
        rules.append("positive_negative_columns")

    if len(text_cols) >= 3:
        struct_s = 1.0
        rules.append("three_text_columns")

    if len(text_cols) >= 3:
        sorted_c = sorted(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        if profile.column_profiles[sorted_c[0]].avg_length < profile.column_profiles[sorted_c[1]].avg_length * 0.5:
            samp_s = 0.9
            rules.append("query_shorter_than_docs")

    if samp_s >= 0.8:
        stat_s = 1.0
        rules.append("ranking_pattern_confirmed")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_information_extraction(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(entities|relations|extracted|triplets|ie_output|json_data)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("ie_output_column_name")

    has_struct_output = (
        bool(profile.dict_columns)
        or any(profile.column_profiles[c].contains_json for c in profile.columns)
        or any(profile.column_profiles[c].contains_entities for c in profile.columns)
    )

    if text_cols and has_struct_output:
        struct_s = 1.0
        rules.append("text_input_structured_output")

    target = next(
        (c for c in profile.columns if c not in text_cols
         and (profile.column_profiles[c].contains_json
              or profile.column_profiles[c].contains_entities
              or c in profile.dict_columns)), None
    )

    if target and sample_rows:
        vals = [r.get(target) for r in sample_rows if r.get(target) is not None]
        if vals and (any(isinstance(v, dict) for v in vals) or any(_looks_like_json(v) for v in vals if isinstance(v, str))):
            samp_s = 1.0
            rules.append("ie_structured_values_in_samples")

    if target and (profile.column_profiles[target].contains_json or profile.column_profiles[target].contains_entities):
        stat_s = 1.0
        rules.append("ie_profile_signature_confirmed")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_text_pair_classification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_pair = any(re.search(r"^(sentence[12]|text[12]|s[12])$", c, re.IGNORECASE) for c in profile.columns)

    if has_pair and label_cols:
        col_s = 1.0
        rules.append("text_pair_and_label")

    if len(text_cols) == 2 and len(label_cols) == 1:
        struct_s = 1.0
        rules.append("exactly_two_texts_one_label")

    if label_cols and sample_rows:
        vals = [r.get(label_cols[0]) for r in sample_rows if r.get(label_cols[0]) is not None]
        if vals and all(isinstance(v, (int, str, bool)) for v in vals):
            samp_s = 1.0
            rules.append("label_values_are_valid")

    if len(text_cols) >= 2:
        p1 = profile.column_profiles[text_cols[0]]
        p2 = profile.column_profiles[text_cols[1]]
        if abs(p1.avg_length - p2.avg_length) / max(p1.avg_length, p2.avg_length, 1) < 0.4:
            stat_s = 1.0
            rules.append("comparable_text_lengths")

    # Suppress if NLI-specific names exist
    if any(re.search(r"^(premise|hypothesis)$", c, re.IGNORECASE) for c in profile.columns):
        col_s *= 0.4
        rules.append("nli_columns_reduce_score")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_nli(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_nli = any(re.search(r"^(premise|hypothesis)$", c, re.IGNORECASE) for c in profile.columns)
    if has_nli:
        col_s = 1.0
        rules.append("premise_hypothesis_columns")

    if len(text_cols) >= 2 and label_cols:
        p_ok = any(re.search(r"premise", c, re.IGNORECASE) for c in text_cols)
        h_ok = any(re.search(r"hypothesis", c, re.IGNORECASE) for c in text_cols)
        struct_s = 1.0 if (p_ok and h_ok) else 0.6
        rules.append("nli_pair_structure")

    if label_cols and sample_rows:
        vals = {str(r.get(label_cols[0])).lower().strip() for r in sample_rows if r.get(label_cols[0]) is not None}
        nli_vocab = {"entailment", "neutral", "contradiction", "0", "1", "2"}
        if vals and (vals <= nli_vocab or len(vals & {"entailment", "neutral", "contradiction"}) > 0):
            samp_s = 1.0
            rules.append("labels_are_nli_vocab")

    if label_cols:
        unique_vals = {k for k in profile.categorical_distributions.get(label_cols[0], {}) if k not in {"", "nan", "null"}}
        if len(unique_vals) == 3:
            stat_s = 1.0
            rules.append("exactly_three_classes")

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_dialogue(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(dialogue|conversation|history|utterances|turns)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("dialogue_column_name")

    if text_cols:
        struct_s = 0.7
        rules.append("text_column_present")

    t = text_cols[0] if text_cols else None
    if t and sample_rows:
        vals = [str(r.get(t) or "") for r in sample_rows]
        speaker_hits = sum(
            bool(re.search(r"\b(speaker\s*[12]|user|assistant|customer|agent|[A-Z][a-z]+:)", v))
            for v in vals
        )
        if vals and speaker_hits / len(vals) > 0.4:
            samp_s = 1.0
            rules.append("speaker_markers_found")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_grammar_correction(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_src = any(re.search(r"^(original|input|source|incorrect|bad|raw)$", c, re.IGNORECASE) for c in profile.columns)
    has_tgt = any(re.search(r"^(corrected|output|target|correct|good|clean)$", c, re.IGNORECASE) for c in profile.columns)

    if has_src and has_tgt:
        col_s = 1.0
        rules.append("grammar_correction_column_names")

    if len(text_cols) >= 2:
        struct_s = 0.9
        rules.append("two_text_columns")

    if len(text_cols) >= 2 and sample_rows:
        t1, t2 = text_cols[0], text_cols[1]
        sims = [
            _string_similarity(str(r.get(t1) or ""), str(r.get(t2) or ""))
            for r in sample_rows
        ]
        high_sim = sum(0.85 <= s < 1.0 for s in sims)
        if sims and high_sim / len(sims) > 0.45:
            samp_s = 1.0
            rules.append("high_similarity_but_not_identical")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_text_simplification(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_comp = any(re.search(r"^(complex|original|input)$", c, re.IGNORECASE) for c in profile.columns)
    has_simp = any(re.search(r"^(simple|simplified|output)$", c, re.IGNORECASE) for c in profile.columns)

    if has_comp and has_simp:
        col_s = 1.0
        rules.append("simplification_column_names")

    if len(text_cols) >= 2:
        struct_s = 0.9

    if len(text_cols) >= 2 and sample_rows:
        t1, t2 = text_cols[0], text_cols[1]
        l1 = profile.column_profiles[t1].avg_length
        l2 = profile.column_profiles[t2].avg_length
        if l1 > l2 * 1.1 and l1 < l2 * 2.5:
            sims = [_string_similarity(str(r.get(t1) or ""), str(r.get(t2) or "")) for r in sample_rows]
            mid_sims = sum(0.5 < s < 0.85 for s in sims)
            if sims and mid_sims / len(sims) > 0.35:
                samp_s = 1.0
                rules.append("compression_with_moderate_overlap")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_style_transfer(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_style = any(re.search(r"^(style|formal|informal|polite|rude|active|passive)$", c, re.IGNORECASE) for c in profile.columns)
    if has_style:
        col_s = 1.0
        rules.append("style_transfer_column_names")

    if len(text_cols) >= 2:
        struct_s = 0.8

    if len(text_cols) >= 2 and sample_rows:
        t1, t2 = text_cols[0], text_cols[1]
        l1 = profile.column_profiles[t1].avg_length
        l2 = profile.column_profiles[t2].avg_length
        if abs(l1 - l2) / max(l1, l2, 1) < 0.25:
            sims = [_string_similarity(str(r.get(t1) or ""), str(r.get(t2) or "")) for r in sample_rows]
            style_sims = sum(0.4 < s < 0.8 for s in sims)
            if sims and style_sims / len(sims) > 0.35:
                samp_s = 1.0
                rules.append("similar_length_moderate_overlap")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_prompt_completion(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    has_prompt = any(re.search(r"^(prompt)$", c, re.IGNORECASE) for c in profile.columns)
    has_comp = any(re.search(r"^(completion|continuation)$", c, re.IGNORECASE) for c in profile.columns)

    if has_prompt and has_comp:
        col_s = 1.0
        rules.append("prompt_and_completion_columns")
    elif has_prompt or has_comp:
        col_s = 0.5

    if len(text_cols) >= 2:
        struct_s = 0.9

    if len(text_cols) >= 2 and sample_rows:
        p_col = next((c for c in text_cols if re.search(r"prompt", c, re.IGNORECASE)), text_cols[0])
        c_col = next((c for c in text_cols if c != p_col), text_cols[1])
        hits = sum(
            1 for r in sample_rows
            if str(r.get(p_col) or "").strip()
            and str(r.get(c_col) or "").strip()
            and (str(r.get(p_col))[-1] not in {".", "?", "!"} or str(r.get(c_col))[0].islower())
        )
        if sample_rows and hits / len(sample_rows) > 0.35:
            samp_s = 1.0
            rules.append("completion_continues_prompt")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_autocomplete(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(prefix|autocomplete|partial|suffix)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("autocomplete_column_name")

    if len(text_cols) >= 2:
        struct_s = 0.9

    if len(text_cols) >= 2:
        short_col = min(text_cols, key=lambda c: profile.column_profiles[c].avg_tokens)
        if profile.column_profiles[short_col].avg_tokens < 6.0:
            samp_s = 1.0
            rules.append("very_short_prefix")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_keyword_generation(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(keywords|tags|keyphrases)s?$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("keyword_column_name")

    active_lists = [c for c in list_cols if c not in text_cols]
    active_labels = [c for c in label_cols if c not in text_cols]

    if text_cols and (active_lists or active_labels):
        struct_s = 1.0
        rules.append("text_and_list_or_label_output")

    t_col = text_cols[0] if text_cols else None
    kw_col = active_lists[0] if active_lists else (active_labels[0] if active_labels else None)
    if t_col and kw_col and sample_rows:
        hits = 0
        valid = 0
        for r in sample_rows:
            t_val = str(r.get(t_col) or "").lower()
            kw_val = r.get(kw_col)
            if not kw_val:
                continue
            valid += 1
            kws = kw_val if isinstance(kw_val, list) else [x.strip() for x in str(kw_val).split(",")]
            if kws and all(str(k).lower() in t_val for k in kws if k):
                hits += 1
        if valid > 0 and hits / valid > 0.45:
            samp_s = 1.0
            rules.append("keywords_are_substrings_of_input")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


def _eval_title_generation(
    profile: DatasetProfile,
    text_cols: List[str], label_cols: List[str], num_cols: List[str], list_cols: List[str],
    sample_rows: List[Dict[str, Any]], _tabular: List[str]
) -> TaskScore:
    rules: List[str] = []
    col_s = struct_s = samp_s = stat_s = 0.0

    for c in profile.columns:
        if re.search(r"^(title|headline|subject)$", c, re.IGNORECASE):
            col_s = 1.0
            rules.append("title_column_name")

    if len(text_cols) >= 2:
        struct_s = 0.8

    if len(text_cols) >= 2:
        title_col = next(
            (c for c in text_cols if re.search(r"title|headline|subject", c, re.IGNORECASE)),
            min(text_cols, key=lambda c: profile.column_profiles[c].avg_length)
        )
        body_col = next((c for c in text_cols if c != title_col), text_cols[0])
        t_prof = profile.column_profiles[title_col]
        b_prof = profile.column_profiles[body_col]
        if t_prof.avg_tokens < 15 and b_prof.avg_length > t_prof.avg_length * 5:
            samp_s = 1.0
            rules.append("title_short_body_much_longer")

    if samp_s == 1.0:
        stat_s = 1.0

    return col_s, struct_s, samp_s, stat_s, rules


# ---------------------------------------------------------------------------
# Main TaskDetector
# ---------------------------------------------------------------------------

class TaskDetector:
    """
    Automatic NLP + Tabular task detector for HuggingFace datasets.
    Supports 26 tasks with improved scoring, tabular detection, and priority resolution.
    """

    # Scoring weights
    W_COL = 0.20
    W_STRUCT = 0.30
    W_SAMPLE = 0.30
    W_STAT = 0.20

    def detect(self, dataset: Any, cfg: DictConfig) -> Dict[str, Any]:
        try:
            from datasets import DatasetDict, Dataset
            if isinstance(dataset, DatasetDict):
                split_name = "train" if "train" in dataset else list(dataset.keys())[0]
                ds = dataset[split_name]
            else:
                ds = dataset

            if len(ds) == 0:
                logger.warning("Empty dataset — falling back to default.")
                return self._fallback_result()

            # ---- Phase 1: Profile ----
            profile = _generate_dataset_profile(ds)

            # ---- Phase 2: Column role detection ----
            metadata_cols: List[str] = []
            text_cols: List[str] = []
            label_cols: List[str] = []
            num_cols: List[str] = []
            list_cols: List[str] = []
            tabular_feat_cols: List[str] = []

            for col in profile.columns:
                cp = profile.column_profiles[col]

                if _is_metadata_column(col, cp, profile.rows):
                    metadata_cols.append(col)
                    continue

                is_text_name = bool(TEXT_COLUMN_NAME_PATTERN.match(col))
                is_long_text = cp.text_ratio > 0.75 and cp.avg_tokens > 3.0
                is_label_name = bool(re.search(r"^(label|sentiment|class|category|target|intent|y|split|fold)s?$", col, re.IGNORECASE))
                approx_unique = cp.unique_ratio * min(profile.rows, 1000)

                # Text columns
                if is_text_name or is_long_text:
                    text_cols.append(col)

                # Label columns
                if is_label_name or (not (is_text_name or is_long_text) and approx_unique <= 300 and cp.unique_ratio < 0.5):
                    label_cols.append(col)

                # Numeric columns
                if cp.numeric_ratio > 0.85 and not (is_text_name or is_long_text):
                    num_cols.append(col)

                # List columns
                if cp.contains_lists or cp.dtype == "list":
                    list_cols.append(col)

                # Tabular feature columns (not free text)
                if _is_tabular_feature(col, cp) and not (is_text_name or is_long_text):
                    tabular_feat_cols.append(col)

            # Remove duplicates while preserving order
            def _dedup(lst: List[str]) -> List[str]:
                seen: Set[str] = set()
                return [x for x in lst if not (x in seen or seen.add(x))]

            text_cols = _dedup(text_cols)
            label_cols = _dedup(label_cols)
            num_cols = _dedup(num_cols)
            list_cols = _dedup(list_cols)
            tabular_feat_cols = _dedup(tabular_feat_cols)

            # ---- Phase 3: Determine primary input & label columns ----
            # Primary input column
            scored_inputs = [
                (col, _score_text_input(col, profile.column_profiles[col]))
                for col in profile.columns
            ]
            scored_inputs = [(c, s) for c, s in scored_inputs if s > 0.0]
            input_col = max(scored_inputs, key=lambda x: x[1])[0] if scored_inputs else (
                next((c for c in profile.columns if c not in metadata_cols), profile.columns[0])
            )

            if input_col not in text_cols and input_col not in metadata_cols:
                text_cols.append(input_col)

            # Primary label column
            scored_labels = [(c, _score_label_candidate(c, profile.column_profiles[c], input_col)) for c in profile.columns if c != input_col]
            scored_nums = [(c, _score_numeric_target(c, profile.column_profiles[c], input_col)) for c in profile.columns if c != input_col]
            all_targets = [(c, s) for c, s in scored_labels + scored_nums if s > 0.0]
            label_col = max(all_targets, key=lambda x: x[1])[0] if all_targets else None

            # ---- Config overrides ----
            manual_text = getattr(cfg.dataset, "text_column", None)
            manual_label = getattr(cfg.dataset, "label_column", None)
            if manual_text and manual_text in profile.columns:
                input_col = manual_text
            if manual_label and manual_label in profile.columns:
                label_col = manual_label

            # ---- Phase 4: Sample rows ----
            sample_rows = ds.select(range(min(len(ds), 50))).to_list()

            # ---- Phase 5: Task scoring ----
            eval_args = (profile, text_cols, label_cols, num_cols, list_cols, sample_rows, tabular_feat_cols)

            evaluators: Dict[str, Any] = {
                "tabular_classification": _eval_tabular_classification,
                "tabular_regression": _eval_tabular_regression,
                "classification": _eval_classification,
                "multi_label_classification": _eval_multi_label_classification,
                "regression": _eval_regression,
                "token_classification": _eval_token_classification,
                "question_answering": _eval_question_answering,
                "summarization": _eval_summarization,
                "translation": _eval_translation,
                "text_generation": _eval_text_generation,
                "instruction_tuning": _eval_instruction_tuning,
                "chat": _eval_chat,
                "retrieval": _eval_retrieval,
                "text_similarity": _eval_text_similarity,
                "ranking": _eval_ranking,
                "information_extraction": _eval_information_extraction,
                "text_pair_classification": _eval_text_pair_classification,
                "natural_language_inference": _eval_nli,
                "dialogue": _eval_dialogue,
                "grammar_correction": _eval_grammar_correction,
                "text_simplification": _eval_text_simplification,
                "style_transfer": _eval_style_transfer,
                "prompt_completion": _eval_prompt_completion,
                "autocomplete": _eval_autocomplete,
                "keyword_generation": _eval_keyword_generation,
                "title_generation": _eval_title_generation,
            }

            task_scores: Dict[str, float] = {}
            task_rules: Dict[str, List[str]] = {}

            for task_name, fn in evaluators.items():
                col_s, str_s, samp_s, stat_s, rules = fn(*eval_args)

                # ---- Validation gates ----
                # Tasks requiring paired text inputs
                if task_name in ("text_pair_classification", "natural_language_inference", "text_similarity") and len(text_cols) < 2:
                    col_s = str_s = samp_s = stat_s = 0.0

                # Tasks requiring a label column
                if task_name in ("classification", "multi_label_classification", "regression",
                                 "text_pair_classification", "natural_language_inference") and not label_col:
                    str_s *= 0.1
                    samp_s *= 0.1

                # Tabular tasks require tabular features; suppress if mostly text
                if task_name in ("tabular_classification", "tabular_regression"):
                    if len(tabular_feat_cols) < 2 or len(text_cols) >= 2:
                        col_s = str_s = samp_s = stat_s = 0.0

                # text_generation should not win over specific tasks with clear signals
                if task_name == "text_generation" and len(profile.columns) > 1:
                    col_s *= 0.5
                    str_s *= 0.5

                score = self.W_COL * col_s + self.W_STRUCT * str_s + self.W_SAMPLE * samp_s + self.W_STAT * stat_s
                task_scores[task_name] = score
                task_rules[task_name] = rules

            # ---- Phase 6: Rank + tie-break ----
            # Primary sort: score DESC; secondary sort: priority DESC (more specific wins)
            sorted_tasks = sorted(
                task_scores.items(),
                key=lambda x: (round(x[1], 4), TASK_PRIORITY.get(x[0], 0)),
                reverse=True,
            )

            top_task, top_score = sorted_tasks[0]
            alt_tasks = [t for t, s in sorted_tasks[1:] if s > 0.08][:3]

            # ---- Phase 7: Build result ----
            result: Dict[str, Any] = {
                "task": top_task,
                "confidence": round(float(top_score), 4),
                "matched_rules": task_rules[top_task],
                "alternative_tasks": alt_tasks,
                "input_column": input_col,
                "label_column": label_col,
                "metadata_columns": [],
                # Legacy keys
                "sub_task": top_task,
                "text_column": input_col,
                "output_column": None,
                "instruction_column": None,
                "context_column": None,
                "problem_type": None,
                "num_labels": None,
                "label2id": None,
                "id2label": None,
                "detected_columns": {
                    "text": text_cols,
                    "label": label_cols,
                    "tabular": tabular_feat_cols,
                    "all": profile.columns,
                },
            }

            # ---- Phase 8: Task-specific column assignment ----
            self._assign_task_columns(result, top_task, text_cols, label_cols, input_col, profile)

            # ---- Phase 9: Enrichment (classification/token cls) ----
            if top_task in ("classification", "multi_label_classification", "token_classification"):
                self._enrich_classification_info(ds, result)
                if top_task == "classification":
                    n = result.get("num_labels")
                    result["problem_type"] = "single_label_classification"
                    result["sub_task"] = "binary_classification" if n == 2 else "multi_class_classification"
                elif top_task == "multi_label_classification":
                    result["problem_type"] = "multi_label_classification"
            elif top_task in ("regression", "tabular_regression"):
                result["num_labels"] = 1
                result["problem_type"] = "regression"
            elif top_task == "tabular_classification":
                result["problem_type"] = "single_label_classification"

            # ---- Compute metadata_columns ----
            exclude = {result.get("input_column"), result.get("label_column"),
                       result.get("context_column"), result.get("instruction_column"),
                       result.get("output_column")} - {None}
            result["metadata_columns"] = [c for c in profile.columns if c not in exclude]

            # ---- Config forced task override ----
            forced_task = getattr(cfg.dataset, "task", None) or getattr(getattr(cfg, "task", None), "name", None)
            if forced_task and forced_task != "auto":
                logger.info(f"Task forced by config: {forced_task}")
                result["task"] = forced_task
                result["sub_task"] = forced_task

            logger.info(
                f"Detected task: {result['task']} (confidence={result['confidence']:.3f}), "
                f"input={result['input_column']}, label={result['label_column']}"
            )
            return result

        except Exception as e:
            logger.exception(f"TaskDetector error: {e}")
            return self._fallback_result()

    def _assign_task_columns(
        self,
        result: Dict[str, Any],
        top_task: str,
        text_cols: List[str],
        label_cols: List[str],
        input_col: str,
        profile: DatasetProfile,
    ) -> None:
        other_text = [c for c in text_cols if c != input_col]

        if top_task == "question_answering":
            q_cols = [c for c in text_cols if re.search(r"question|query", c, re.IGNORECASE)]
            a_cols = [c for c in text_cols if re.search(r"answer", c, re.IGNORECASE)]
            c_cols = [c for c in text_cols if re.search(r"context|passage", c, re.IGNORECASE)]
            result["input_column"] = q_cols[0] if q_cols else input_col
            result["output_column"] = a_cols[0] if a_cols else (other_text[0] if other_text else None)
            result["context_column"] = c_cols[0] if c_cols else (other_text[1] if len(other_text) > 1 else None)

        elif top_task in ("summarization", "translation", "grammar_correction",
                          "text_simplification", "style_transfer", "prompt_completion", "autocomplete"):
            result["input_column"] = input_col
            result["output_column"] = other_text[0] if other_text else None

        elif top_task == "instruction_tuning":
            inst = next((c for c in text_cols if re.search(r"instruction|prompt", c, re.IGNORECASE)), input_col)
            out = next((c for c in text_cols if re.search(r"output|response|completion", c, re.IGNORECASE)), None)
            result["instruction_column"] = inst
            result["output_column"] = out or (other_text[0] if other_text else None)

        elif top_task in ("classification", "multi_label_classification", "regression",
                          "tabular_classification", "tabular_regression",
                          "token_classification", "text_pair_classification",
                          "natural_language_inference"):
            # output_column is the label column for these tasks
            result["output_column"] = result.get("label_column")

        elif top_task in ("title_generation", "keyword_generation", "text_generation", "text_similarity"):
            result["output_column"] = other_text[0] if other_text else result.get("label_column")

        else:
            # Default: just set output to next text col
            result["output_column"] = other_text[0] if other_text else None

    def _enrich_classification_info(self, split: Any, task_info: Dict[str, Any]) -> None:
        label_col = task_info.get("label_column")
        if not label_col:
            return
        try:
            from datasets.features import ClassLabel, Sequence
            feat = split.features.get(label_col)
            if isinstance(feat, ClassLabel):
                names = feat.names
                task_info.update({"num_labels": len(names), "label2id": {n: i for i, n in enumerate(names)}, "id2label": {i: n for i, n in enumerate(names)}})
                return
            if isinstance(feat, Sequence) and isinstance(feat.feature, ClassLabel):
                names = feat.feature.names
                task_info.update({"num_labels": len(names), "label2id": {n: i for i, n in enumerate(names)}, "id2label": {i: n for i, n in enumerate(names)}})
                return
        except Exception:
            pass

        try:
            vals = split[label_col]
            flat = []
            for v in vals:
                flat.extend(v) if isinstance(v, (list, tuple)) else flat.append(v)
            unique_labels = sorted(set(str(x) for x in flat if x is not None))
            task_info.update({
                "num_labels": len(unique_labels),
                "label2id": {l: i for i, l in enumerate(unique_labels)},
                "id2label": {i: l for i, l in enumerate(unique_labels)},
            })
        except Exception as e:
            logger.warning(f"Failed to enrich classification info: {e}")

    def _fallback_result(self) -> Dict[str, Any]:
        return {
            "task": "causal_lm",
            "confidence": 0.0,
            "matched_rules": ["fallback_to_causal_lm"],
            "alternative_tasks": [],
            "input_column": "text",
            "label_column": None,
            "metadata_columns": [],
            "sub_task": "text_generation",
            "text_column": "text",
            "output_column": None,
            "instruction_column": None,
            "context_column": None,
            "num_labels": None,
            "label2id": None,
            "id2label": None,
            "problem_type": None,
            "detected_columns": {"text": [], "label": [], "tabular": [], "all": []},
        }