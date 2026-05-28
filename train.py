"""
Train runner and dataset persistence helper.

Usage examples:
  python train.py dataset.name=dataset/twitter_training.csv dataset.save_after_load=true dataset.save_dir=dataset/my_dataset

This script merges a small default config with OmegaConf CLI overrides,
calls `src.data.loader.load_dataset`, and optionally saves splits to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig

from src.utils.common import get_logger, ensure_dir
from src.data.loader import load_dataset

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
        "save_after_load": False,
        "save_dir": "dataset/my_dataset",
    }
})


def save_dataset_splits(ds: Any, save_dir: str) -> dict:
    """Save each split in `ds` to `save_dir` and return a report dict."""
    p = ensure_dir(save_dir)
    report: dict = {"splits": {}, "total_splits": 0}

    for split, d in ds.items():
        df = d.to_pandas()
        # prefer parquet when available
        out_path = p / f"{split}.parquet"
        try:
            df.to_parquet(out_path, index=False)
            fmt = "parquet"
        except Exception:
            out_path = p / f"{split}.csv"
            df.to_csv(out_path, index=False)
            fmt = "csv"

        report["splits"][split] = {"path": str(out_path), "rows": len(df), "format": fmt}
        report["total_splits"] += 1

    # write a small report.json
    report_path = p / "load_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved dataset splits to: {p}")
    logger.info(f"Report: {report_path}")
    return report


def main(cfg: DictConfig | None = None) -> int:
    # Merge defaults with CLI overrides
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(DEFAULT_CFG, cli_cfg) if cfg is None else OmegaConf.merge(DEFAULT_CFG, cfg)

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    ds = load_dataset(cfg)

    if cfg.dataset.save_after_load:
        save_dataset_splits(ds, cfg.dataset.save_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from src.data.loader import load_dataset

def main():
    # Example usage with OmegaConf defaults and CLI overrides
    from omegaconf import OmegaConf
    import sys

    cfg = OmegaConf.create({
        "dataset": {
            "name": "dataset/twitter_training.csv",
            "cache_dir": "./cache"
        }
    })

    # Merge CLI overrides like `dataset.name=...`
    if len(sys.argv) > 1:
        try:
            cli_cfg = OmegaConf.from_cli()
            cfg = OmegaConf.merge(cfg, cli_cfg)
        except Exception:
            # ignore CLI parsing errors and continue with defaults
            pass

    dataset = load_dataset(cfg)
    print(dataset)


if __name__ == "__main__":
    main()