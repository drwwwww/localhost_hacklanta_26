"""Loading the raw postings and A's splits.

splits.json contract (keyed on EMSCAD `job_id`, so B never needs A's feature
matrices — features are built inside the model pipeline, per training fold):

    {
      "test":  [job_id, ...],            # held-out, never used for fitting/calibration
      "folds": [[job_id, ...], ...]      # k disjoint validation folds over the rest
    }

Fold i validates on folds[i] and trains on the other folds. Duplicate clusters
must not span folds or test. Rows A dropped during dedup are simply absent.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "fake_job_postings.csv"
SPLITS_JSON = ROOT / "data" / "processed" / "splits.json"

ID_COL = "job_id"
LABEL_COL = "fraudulent"


def load_postings(path=RAW_CSV):
    df = pd.read_csv(path)
    if df[ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicate {ID_COL} values")
    return df.set_index(ID_COL)


def load_splits(path, known_ids):
    with open(path) as f:
        splits = json.load(f)
    validate_splits(splits, known_ids)
    return splits


def validate_splits(splits, known_ids):
    parts = [splits["test"], *splits["folds"]]
    if len(splits["folds"]) < 2:
        raise ValueError("splits.json needs at least 2 folds")
    seen = set()
    for part in parts:
        ids = set(part)
        if len(ids) != len(part):
            raise ValueError("splits.json: repeated id inside one fold")
        if seen & ids:
            raise ValueError("splits.json: an id appears in more than one fold/test")
        seen |= ids
    unknown = seen - set(known_ids)
    if unknown:
        raise ValueError(f"splits.json: {len(unknown)} ids not in the raw csv, e.g. {sorted(unknown)[:5]}")


def dev_splits(df, n_folds=5, test_size=0.2, seed=0):
    """Stand-in for A's splits. NOT group-aware: near-duplicates leak across
    folds, so numbers from this are optimistic and must not be reported."""
    y = df[LABEL_COL]
    trainval, test = train_test_split(df.index.to_numpy(), test_size=test_size, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [trainval[val].tolist() for _, val in skf.split(trainval, y.loc[trainval])]
    return {"test": test.tolist(), "folds": folds}
