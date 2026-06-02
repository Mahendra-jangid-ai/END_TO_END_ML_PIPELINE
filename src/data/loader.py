"""
Simple dataset loader - loads dataset from pandas and returns dataframe.
Accepts configuration from Hydra.
"""
import pandas as pd
from omegaconf import DictConfig


def load_dataset(cfg: DictConfig):
    path = cfg.dataset.path if hasattr(cfg.dataset, 'path') else cfg.dataset.name
    
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.tsv'):
        return pd.read_csv(path, sep='\t')
    elif path.endswith('.json'):
        return pd.read_json(path)
    elif path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format. Supported: CSV, TSV, JSON, JSONL, Parquet, Excel")
