"""Feature frame + per-tier sklearn pipelines.

Everything that is fit (TF-IDF vocabulary, one-hot categories, scaling) lives
inside the pipeline so it is refit on each training fold and shipped with the
model for predict().
"""

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
BINARY = ["telecommuting", "has_company_logo", "has_questions"]
CATEGORICAL = ["employment_type", "required_experience", "required_education"]
MISSING_FIELDS = [
    "company_profile", "requirements", "benefits", "salary_range", "department", "location",
    "employment_type", "required_experience", "required_education", "industry", "function",
]
RAW_COLUMNS = list(dict.fromkeys(TEXT_FIELDS + BINARY + CATEGORICAL + MISSING_FIELDS))

FLAGS = BINARY + [f"{c}_missing" for c in MISSING_FIELDS]
NUMERIC = ["log_text_len"]
# Features that track how EMSCAD was collected more than whether a posting is a scam.
SHORTCUTS = ["has_company_logo", "has_questions", "company_profile_missing"]

# name -> (use metadata, use text, drop shortcut features)
TIERS = {
    "A": (True, False, False),
    "A-noshortcut": (True, False, True),
    "B": (False, True, False),
    "C": (True, True, False),
    "C-noshortcut": (True, True, True),
}

# EMSCAD masks contact details as #URL_<hash>#, #EMAIL_<hash>#, #PHONE_<hash>#.
# Real pasted text is mapped to the same tokens so train and inference agree.
_MASKS = [
    (re.compile(r"#URL_\w+#|https?://\S+|www\.\S+", re.I), " urltoken "),
    (re.compile(r"#EMAIL_\w+#|[\w.+-]+@[\w-]+\.[\w.-]+", re.I), " emailtoken "),
    (re.compile(r"#PHONE_\w+#|\+?\d[\d\s().-]{8,}\d", re.I), " phonetoken "),
]


def normalize_text(text):
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    return re.sub(r"\s+", " ", text).strip()


def build_frame(raw):
    """Raw EMSCAD-shaped rows (any subset of columns) -> model input frame."""
    raw = raw.reindex(columns=RAW_COLUMNS)
    as_text = raw.fillna("").astype(str)
    out = pd.DataFrame(index=raw.index)
    out["text"] = as_text[TEXT_FIELDS].agg(" ".join, axis=1).map(normalize_text)
    for c in BINARY:
        out[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(int)
    for c in MISSING_FIELDS:
        out[f"{c}_missing"] = (as_text[c].str.strip() == "").astype(int)
    for c in CATEGORICAL:
        out[c] = as_text[c].str.strip().replace("", "unknown")
    out["log_text_len"] = np.log1p(out["text"].str.len())
    return out


def make_pipeline(tier):
    use_meta, use_text, drop_shortcuts = TIERS[tier]
    transformers = []
    if use_meta:
        flags = [c for c in FLAGS if not (drop_shortcuts and c in SHORTCUTS)]
        transformers += [
            ("flags", "passthrough", flags),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    if use_text:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                                sublinear_tf=True, stop_words="english")
        transformers.append(("tfidf", tfidf, "text"))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        # C=10: PR-AUC plateaus there under duplicate-aware CV; C=1 underfits the text tiers.
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", C=10, max_iter=2000)),
    ])
