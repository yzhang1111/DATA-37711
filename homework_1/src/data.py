"""Data loaders shared by all problems."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALE_CSV = Path(os.environ.get("ALE_CSV", ROOT / "data" / "ale.csv"))
ALE_FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
SEED = 0


def _normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def load_ale_df():
    if ALE_CSV.exists():
        return _normalize_cols(pd.read_csv(ALE_CSV))
    from ucimlrepo import fetch_ucirepo  # network fallback
    df = _normalize_cols(fetch_ucirepo(id=844).data.original)
    ALE_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(ALE_CSV, index=False)
    return df


def load_ale():
    """X = first 4 variables, y = ale (sd_ale ignored)."""
    df = load_ale_df()
    return df[ALE_FEATURES].to_numpy(float), df["ale"].to_numpy(float)


def split_scale(X, y, seed=SEED, stratify=False):
    """80/20 split; standardize X using training statistics only."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if stratify else None)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def load_cancer_split(seed=SEED):
    d = load_breast_cancer()
    return split_scale(d.data, d.target.astype(float), seed, stratify=True)


def load_ale_split(seed=SEED):
    X, y = load_ale()
    return split_scale(X, y, seed)
