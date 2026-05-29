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
from src.data.task_detector import TaskDetector
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


def _validate_manual_label_column(ds: Any, label_column: str | None) -> None:
    """Validate manual label column exists in loaded dataset."""
    if not label_column:
        return

    train_split = ds.get("train") if hasattr(ds, "get") else None
    if train_split is None:
        return

    columns = list(train_split.column_names)
    if label_column not in columns:
        raise ValueError(
            f"dataset.label_column='{label_column}' not found in dataset columns: {columns}"
        )


def save_task_info(task_info: dict[str, Any], save_dir: str) -> dict[str, str]:
    """Save detected task metadata to dedicated files."""
    p = ensure_dir(save_dir)

    task_json_path = p / "task_info.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=2, ensure_ascii=False)

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
        json.dump(report, f, indent=2)
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

    ds = load_dataset(cfg)

    task_info: dict[str, Any]
    if cfg.task.name:
        logger.info(f"Task provided in config: {cfg.task.name} (skipping task auto-detection)")
        _validate_manual_label_column(ds, cfg.dataset.label_column)
        logger.info("Manual task + label_column mode: preserving all columns in saved splits")
        task_info = {
            "task": cfg.task.name,
            "text_column": cfg.dataset.text_column,
            "label_column": cfg.dataset.label_column,
            "input_column": cfg.dataset.input_column,
            "output_column": cfg.dataset.output_column,
            "instruction_column": cfg.dataset.instruction_column,
            "num_labels": cfg.task.num_labels,
            "label2id": cfg.task.label2id,
            "id2label": cfg.task.id2label,
            "problem_type": cfg.task.problem_type or cfg.task.name,
        }
    else:
        detector = TaskDetector()
        task_info = detector.detect(ds, cfg)
        logger.info(f"Detected task: {task_info.get('task')}")

    # Keep final task details in config for downstream pipeline stages.
    cfg.task.name = task_info.get("task")
    cfg.task.num_labels = task_info.get("num_labels")
    cfg.task.label2id = task_info.get("label2id")
    cfg.task.id2label = task_info.get("id2label")
    cfg.task.problem_type = task_info.get("problem_type")

    if cfg.dataset.save_after_load:
        dataset_report = save_dataset_splits(ds, cfg.dataset.save_dir)
        task_paths = save_task_info(task_info, cfg.dataset.save_dir)

        report: dict[str, Any] = {
            **dataset_report,
            "detected_task": task_info.get("task"),
            "task_info": task_info,
            "artifacts": task_paths,
        }
        report_path = save_load_report(report, cfg.dataset.save_dir)

        logger.info(f"Saved dataset splits to: {Path(cfg.dataset.save_dir)}")
        logger.info(f"Task file: {task_paths['task_name_file']}")
        logger.info(f"Task metadata: {task_paths['task_info_json']}")
        logger.info(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
