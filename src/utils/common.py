"""
Shared utilities: logging, seeding, device detection, type helpers.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device_info() -> dict[str, Any]:
    """Return GPU/CPU availability info."""
    info: dict[str, Any] = {"cuda_available": False, "gpu_count": 0, "vram_gb": 0.0}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except ImportError:
        pass
    return info


def get_optimal_dtype() -> str:
    """Return best floating-point dtype for current hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:  # Ampere+ supports bf16
                return "bfloat16"
            return "float16"
    except ImportError:
        pass
    return "float32"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_import(module_name: str) -> Any | None:
    """Try to import a module; return None if unavailable."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
