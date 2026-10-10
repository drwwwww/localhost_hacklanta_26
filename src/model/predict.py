"""Inference entry point for the app: predict(text, meta) -> (score, risk_label)."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model.features import build_frame

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method, raw_scores, y):
    if method == "isotonic":
        return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw_scores, y)
    return LogisticRegression().fit(_logit(raw_scores), y)  # Platt scaling


def apply_calibrator(calibrator, raw_scores):
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(raw_scores)
    return calibrator.predict_proba(_logit(raw_scores))[:, 1]


@dataclass
class FraudModel:
    tier: str
    pipeline: Pipeline
    calibrator: object
    threshold: float         # at/above -> "high" (chosen for the precision target)
    review_threshold: float  # at/above -> "medium"

    def score(self, frame):
        return apply_calibrator(self.calibrator, self.pipeline.predict_proba(frame)[:, 1])

    def risk_label(self, score):
        if score >= self.threshold:
            return "high"
        if score >= self.review_threshold:
            return "medium"
        return "low"


@lru_cache(maxsize=None)
def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(text, meta=None, model=None):
    """Score one posting or recruiter message.

    text: the pasted body. meta: optional dict of EMSCAD fields (title,
    company_profile, has_company_logo, employment_type, ...); only used by
    tiers that include metadata. Returns (calibrated fraud probability, "low" |
    "medium" | "high").
    """
    model = model or load_model()
    row = {**(meta or {}), "description": text}
    score = float(model.score(build_frame(pd.DataFrame([row])))[0])
    return score, model.risk_label(score)
