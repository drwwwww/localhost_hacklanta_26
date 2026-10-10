"""False alarms on live legitimate postings (src/realworld/live.py): model, rules, and the app's combined verdict.

    python -m src.model.live_eval                # tiers with a models/model_<tier>.joblib
    python -m src.model.live_eval --tiers B F

Every row is a real posting open on an employer's own board when it was fetched, so
anything above "low" is a false alarm. Scored twice: the whole posting, and one 17-word
cut of it, since people also paste fragments. Thresholds are the ones stored in each
artifact (chosen on EMSCAD), like src.model.gap and src.model.realworld_eval.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from src.model.features import build_frame
from src.model.realworld_eval import MODELS, to_markdown
from src.model.verdict import ORDER
from src.realworld.dataset import snippet
from src.realworld.live import load
from src.redflags import rule_risk

OUT = MODELS / "live.md"


def _rates(risk):
    risk = np.asarray(risk)
    return {"n": len(risk), "flagged_high": float((risk == "high").mean()), "flagged_medium_or_high": float((risk != "low").mean())}


def evaluate(tiers):
    postings = load()
    subsets = {"live posting": postings["text"], "live posting_snippet": postings["text"].map(snippet)}
    rows = []
    for name, texts in subsets.items():
        rules = texts.map(rule_risk).to_numpy()
        rows.append({"subset": name, "method": "rules only", **_rates(rules)})
        for tier in tiers:
            model = joblib.load(MODELS / f"model_{tier}.joblib")
            scores = model.score(build_frame(pd.DataFrame({"description": texts.to_numpy()})))
            m_risk = np.array([model.risk_label(s) for s in scores])
            combined = np.array([max(m, r, key=ORDER.__getitem__) for m, r in zip(m_risk, rules)])
            rows.append({"subset": name, "method": f"tier {tier}", **_rates(m_risk)})
            rows.append({"subset": name, "method": f"tier {tier} + rules", **_rates(combined)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    available = sorted(f.stem.removeprefix("model_") for f in MODELS.glob("model_*.joblib"))
    p.add_argument("--tiers", nargs="+", default=[t for t in available if t in ("B", "E", "F", "FE")])
    args = p.parse_args()
    table = evaluate(args.tiers)
    OUT.write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
