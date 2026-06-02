"""
Train runner and dataset persistence helper.

Usage example:
  python train.py dataset.name=dataset/twitter_training.csv dataset.save_after_load=true dataset.save_dir=dataset/my_dataset

This script merges a small default config with OmegaConf CLI overrides,
loads data using `src.data.loader.load_dataset`, detects task metadata,
and optionally saves splits + task info to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.data.loader import load_dataset
from src.tokenization.tokenizer import tokenize_dataset_dict
from src.utils.common import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)


DEFAULT_CFG = OmegaConf.create({
    "dataset": {
        "name": "dataset/twitter_training.csv",
        "cache_dir": "",
        "max_samples": None,
        "seed": 42,
        "streaming": False,
        "test_size": 0.1,
        "val_size": 0.1,
        "save_after_load": True,
        "save_dir": "dataset/processed",
        "text_column": None,
        "label_column": None,
        "input_column": None,
        "output_column": None,
        "instruction_column": None,
    },
    "task": {
        "name": None,
        "num_labels": None,
        "label2id": None,
        "id2label": None,
        "problem_type": None,
    },
    "tokenizer": {
        "model_name": None,
        "max_length": 512,
        "truncation": True,
        "padding": True,
    },
})


def _drop_none_values(value: Any) -> Any:
    """Recursively remove keys/items with None values."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            nested_clean = _drop_none_values(nested)
            if nested_clean is not None:
                cleaned[key] = nested_clean
        return cleaned
    if isinstance(value, list):
        return [_drop_none_values(v) for v in value if v is not None]
    return value


def _cfg_without_nones(cfg: DictConfig) -> DictConfig:
    """Return config with all None values removed to preserve defaults."""
    as_obj = OmegaConf.to_container(cfg, resolve=False)
    cleaned = _drop_none_values(as_obj)
    return OmegaConf.create(cleaned)


def _to_json_safe(value: Any) -> Any:
    """Convert OmegaConf and nested containers into JSON-serializable objects."""
    if isinstance(value, DictConfig):
        return _to_json_safe(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(key): _to_json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    return value


def _load_project_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Compose project config from Hydra-style defaults list."""
    root_path = Path(config_path)
    if not root_path.exists():
        return OmegaConf.create({})

    root_cfg = OmegaConf.load(root_path)
    defaults_obj = root_cfg.get("defaults", [])
    defaults = OmegaConf.to_container(defaults_obj, resolve=True) if defaults_obj else []
    base_dir = root_path.parent

    # Keep root-level overrides except defaults list.
    root_no_defaults = OmegaConf.create({k: v for k, v in root_cfg.items() if k != "defaults"})

    merged = OmegaConf.create({})
    self_merged = False

    for item in defaults:
        if isinstance(item, str):
            if item == "_self_":
                merged = OmegaConf.merge(merged, root_no_defaults)
                self_merged = True
            continue

        if not isinstance(item, dict) or len(item) != 1:
            continue

        group, name = next(iter(item.items()))
        if group == "_self_":
            merged = OmegaConf.merge(merged, root_no_defaults)
            self_merged = True
            continue

        part_path = base_dir / group / f"{name}.yaml"
        if part_path.exists():
            part_cfg = OmegaConf.load(part_path)
            wrapped_cfg = OmegaConf.create({group: part_cfg})
            merged = OmegaConf.merge(merged, wrapped_cfg)
        else:
            logger.warning(f"Config part not found: {part_path}")

    if not self_merged:
        merged = OmegaConf.merge(merged, root_no_defaults)

    return merged


def save_dataset_splits(ds: Any, save_dir: str) -> dict:
    """Save each split in `ds` to `save_dir` and return a report dict."""
    p = ensure_dir(save_dir)
    report: dict[str, Any] = {"splits": {}, "total_splits": 0}

    for split, d in ds.items():
        df = d.to_pandas()
        out_path = p / f"{split}.parquet"
        try:
            df.to_parquet(out_path, index=False)
            fmt = "parquet"
        except Exception:
            out_path = p / f"{split}.csv"
            df.to_csv(out_path, index=False)
            fmt = "csv"

        report["splits"][split] = {
            "path": str(out_path),
            "rows": len(df),
            "format": fmt,
            "num_columns": len(df.columns),
            "columns": list(df.columns),
        }
        report["total_splits"] += 1

    return report





def save_task_info(task_info: dict[str, Any], save_dir: str) -> dict[str, str]:
    """Save detected task metadata to dedicated files."""
    p = ensure_dir(save_dir)

    task_json_path = p / "task_info.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(task_info), f, indent=2, ensure_ascii=False)

    # Single-file quick view for "which task" requirement.
    task_name_path = p / "detected_task.txt"
    with open(task_name_path, "w", encoding="utf-8") as f:
        f.write(str(task_info.get("task", "unknown")))

    return {
        "task_info_json": str(task_json_path),
        "task_name_file": str(task_name_path),
    }


def save_load_report(report: dict[str, Any], save_dir: str) -> str:
    """Persist consolidated load report."""
    p = ensure_dir(save_dir)
    report_path = p / "load_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(report), f, indent=2)
    return str(report_path)


def main(cfg: DictConfig | None = None) -> int:
    cli_cfg = OmegaConf.from_cli()
    project_cfg = _cfg_without_nones(_load_project_config("configs/config.yaml"))
    cfg = (
        OmegaConf.merge(DEFAULT_CFG, project_cfg, cli_cfg)
        if cfg is None
        else OmegaConf.merge(DEFAULT_CFG, project_cfg, cfg)
    )

    set_seed(int(cfg.dataset.seed))

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    ds_raw = load_dataset(cfg)

    # Print loaded dataset columns
    print("\n" + "="*50)
    print("DATASET LOADED SUCCESSFULLY")
    print(f"File: {cfg.dataset.name}")
    print(f"Available Columns: {list(ds_raw.columns)}")
    print("="*50 + "\n")
    
    # 1. Ask user which columns to keep as input features
    while True:
        text_cols_input = input("Enter input text column(s) (comma-separated, e.g. text or review,title): ").strip()
        if not text_cols_input:
            print("Error: Input column name cannot be empty. Please try again.")
            continue
        text_columns = [col.strip() for col in text_cols_input.split(",") if col.strip()]
        invalid_cols = [col for col in text_columns if col not in ds_raw.columns]
        if invalid_cols:
            print(f"Error: Column(s) {invalid_cols} not found in dataset. Please try again.")
            continue
        break
        
    # 2. Ask user which column is the target/output column
    while True:
        label_column = input("Enter target/output column (e.g. label or rating): ").strip()
        if not label_column:
            print("Error: Target column name cannot be empty. Please try again.")
            continue
        if label_column not in ds_raw.columns:
            print(f"Error: Column '{label_column}' not found in dataset. Please try again.")
            continue
        break

    # 3. Ask user for task type: classification or regression
    while True:
        task_choice = input("Select NLP task type:\n  1. Classification\n  2. Regression\nEnter choice (1 or 2): ").strip()
        if task_choice == "1":
            task_name = "classification"
            break
        elif task_choice == "2":
            task_name = "regression"
            break
        else:
            print("Error: Invalid choice. Please enter 1 or 2.")

    # 4. Suggest models in lightweight to heavyweight order and ask user to select/type one
    print("\n" + "-"*50)
    print(f"MODEL SUGGESTIONS FOR {task_name.upper()} (Lightweight to Heavyweight):")
    suggestions = [
        "distilbert-base-uncased  (Lightweight, ~66M parameters)",
        "bert-base-uncased        (Medium, ~110M parameters)",
        "roberta-base             (Medium, ~125M parameters)",
        "microsoft/deberta-v3-base (Heavyweight, ~86M parameters but advanced architecture)"
    ]
    for idx, sug in enumerate(suggestions, 1):
        print(f"  {idx}. {sug}")
    print("-"*50)
    
    while True:
        model_choice = input("Select model (1-4) or type custom HF model path (e.g. google/electra-base): ").strip()
        if model_choice in ["1", "2", "3", "4"]:
            idx = int(model_choice) - 1
            model_name = suggestions[idx].split()[0]
            break
        elif model_choice:
            model_name = model_choice
            break
        else:
            print("Error: Model selection cannot be empty.")

    print(f"\nProceeding with model: '{model_name}' on task '{task_name}'...\n")

    # 5. Populate task_info
    task_info = {
        "task": task_name,
        "text_column": text_columns[0],
        "label_column": label_column,
        "text_columns": text_columns,
        "label_columns": [label_column],
        "sub_task": task_name,
        "problem_type": task_name,
        "detected_columns": {
            "text": text_columns,
            "label": [label_column]
        },
        "input_column": ", ".join(text_columns),
        "output_column": label_column,
        "instruction_column": None,
        "context_column": None,
    }

    # Calculate class details if classification
    num_labels, label2id, id2label = None, None, None
    if task_name == "classification":
        unique_labels = sorted(list(ds_raw[label_column].dropna().unique()))
        num_labels = len(unique_labels)
        label2id = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
        id2label = {idx: str(lbl) for idx, lbl in enumerate(unique_labels)}

    task_info["num_labels"] = num_labels
    task_info["label2id"] = label2id
    task_info["id2label"] = id2label

    # Update configs
    cfg.task.name = task_name
    cfg.task.num_labels = num_labels
    cfg.task.label2id = label2id
    cfg.task.id2label = id2label
    cfg.task.problem_type = task_name
    
    cfg.dataset.text_column = text_columns[0]
    cfg.dataset.label_column = label_column
    cfg.tokenizer.model_name = model_name

    # Connect to universal preprocessor and splits splitting
    from datasets import Dataset, DatasetDict
    from src.preprocessing.preprocessor import preprocess_dataset
    
    # Convert raw dataframe to HuggingFace Dataset
    hf_ds = Dataset.from_pandas(ds_raw, preserve_index=False)
    
    # Split dataset based on val_size and test_size
    test_size = float(cfg.dataset.test_size)
    val_size = float(cfg.dataset.val_size)
    total_eval_size = test_size + val_size
    
    if total_eval_size > 0 and total_eval_size < 1.0:
        split1 = hf_ds.train_test_split(test_size=total_eval_size, seed=int(cfg.dataset.seed))
        train_ds = split1["train"]
        eval_ds = split1["test"]
        
        if val_size > 0 and test_size > 0:
            val_ratio = val_size / total_eval_size
            split2 = eval_ds.train_test_split(test_size=1.0 - val_ratio, seed=int(cfg.dataset.seed))
            val_ds = split2["train"]
            test_ds = split2["test"]
        elif val_size > 0:
            val_ds = eval_ds
            test_ds = None
        else:
            val_ds = None
            test_ds = eval_ds
    else:
        train_ds = hf_ds
        val_ds = None
        test_ds = None
        
    ds_splits = {"train": train_ds}
    if val_ds is not None:
        ds_splits["validation"] = val_ds
    if test_ds is not None:
        ds_splits["test"] = test_ds
        
    ds = DatasetDict(ds_splits)
    
    # Run preprocessor
    cleaned_ds, cleaning_report, quality_report, row_level_audit, issue_summary, warnings, recommendations = preprocess_dataset(ds, cfg, task_info=task_info)
    
    # Attach reports to the DatasetDict object so train.py can write them to report
    cleaned_ds.cleaning_report = cleaning_report
    cleaned_ds.quality_report = quality_report
    cleaned_ds.row_level_audit = row_level_audit
    cleaned_ds.issue_summary = issue_summary
    cleaned_ds.warnings = warnings
    cleaned_ds.recommendations = recommendations
    
    ds = cleaned_ds

    if cfg.dataset.save_after_load:
        dataset_report = save_dataset_splits(ds, cfg.dataset.save_dir)
        task_paths = save_task_info(task_info, cfg.dataset.save_dir)

        report: dict[str, Any] = {
            **dataset_report,
            "detected_task": task_info.get("task"),
            "task_info": task_info,
            "artifacts": task_paths,
        }

        tokenizer_model_name = cfg.tokenizer.model_name or getattr(cfg.model, "name", None)
        if tokenizer_model_name:
            tokenizer_output_dir = Path(cfg.dataset.save_dir) / "tokenized"
            
            # Use all input columns specified by the user
            tok_text_cols = task_info.get("text_columns", [])
            if not tok_text_cols:
                if cfg.dataset.text_column:
                    tok_text_cols = [cfg.dataset.text_column]
                elif task_info.get("input_column"):
                    tok_text_cols = [task_info.get("input_column")]

            # Remove all other columns from ds before tokenizing to keep only input and output columns
            all_cols = ds["train"].column_names
            cols_to_keep = tok_text_cols + [label_column]
            cols_to_remove = [c for c in all_cols if c not in cols_to_keep]
            if cols_to_remove:
                old_ds = ds
                ds = ds.remove_columns(cols_to_remove)
                # Preserve cleaning/quality reports and metrics attributes
                if hasattr(old_ds, "cleaning_report"):
                    ds.cleaning_report = old_ds.cleaning_report
                    ds.quality_report = old_ds.quality_report
                    ds.row_level_audit = old_ds.row_level_audit
                    ds.issue_summary = old_ds.issue_summary
                    ds.warnings = old_ds.warnings
                    ds.recommendations = old_ds.recommendations

            logger.info(
                "Tokenizing dataset splits with tokenizer model=%s, text_columns=%s",
                tokenizer_model_name,
                tok_text_cols,
            )

            tokenized_ds = tokenize_dataset_dict(
                ds,
                model_name=tokenizer_model_name,
                text_columns=tok_text_cols or None,
                max_length=int(cfg.tokenizer.max_length) if cfg.tokenizer.max_length is not None else 512,
                truncation=bool(cfg.tokenizer.truncation),
                padding=bool(cfg.tokenizer.padding),
                output_dir=str(tokenizer_output_dir),
            )

            report["tokenized_dataset"] = {
                "path": str(tokenizer_output_dir),
                "splits": list(tokenized_ds.keys()),
            }

        if hasattr(ds, "cleaning_report"):
            report["cleaning_report"] = ds.cleaning_report
            report["quality_report"] = ds.quality_report
            report["row_level_audit"] = ds.row_level_audit
            report["issue_summary"] = ds.issue_summary
            report["warnings"] = ds.warnings
            report["recommendations"] = ds.recommendations
            
        report_path = save_load_report(report, cfg.dataset.save_dir)

        logger.info(f"Saved dataset splits to: {Path(cfg.dataset.save_dir)}")
        if cfg.tokenizer.model_name:
            logger.info(f"Saved tokenized dataset to: {tokenizer_output_dir}")
        logger.info(f"Task file: {task_paths['task_name_file']}")
        logger.info(f"Task metadata: {task_paths['task_info_json']}")
        logger.info(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
