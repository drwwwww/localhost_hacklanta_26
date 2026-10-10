"""Score a labelled csv (C's challenge set) with a trained model.

    python -m src.model.evaluate data/challenge_set/challenge_set.csv
    python -m src.model.evaluate path.csv --text-col body --label-col label --positive-label scam

Thresholds come from the model artifact (chosen on EMSCAD), so this measures how
the deployed model transfers, not how well it could do if re-tuned on this set.
Compare against the EMSCAD test row in models/ablation.md.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.features import build_frame
from src.model.predict import MODEL_PATH


def score_csv(df, model, text_col, label_col, positive_label=None):
    """Adds score, risk and y columns. Columns named like EMSCAD fields are used as metadata."""
    out = df.copy()
    out["score"] = model.score(build_frame(df.rename(columns={text_col: "description"})))
    out["risk"] = out["score"].map(model.risk_label)
    if positive_label is None:
        labels = set(df[label_col].unique())
        if not labels <= {0, 1}:
            raise ValueError(f"{label_col} has values {sorted(map(str, labels))}; pass --positive-label")
        out["y"] = df[label_col].astype(int)
    else:
        out["y"] = (df[label_col].astype(str) == positive_label).astype(int)
    return out


def summarize(scored):
    y, high = scored["y"], scored["risk"] == "high"
    fraud, legit = y == 1, y == 0
    both = fraud.any() and legit.any()
    return {
        "n": len(y),
        "n_fraud": int(fraud.sum()),
        "pr_auc": average_precision_score(y, scored["score"]) if both else float("nan"),
        "roc_auc": roc_auc_score(y, scored["score"]) if both else float("nan"),
        "precision_high": y[high].mean() if high.any() else float("nan"),
        "recall_high": high[fraud].mean() if fraud.any() else float("nan"),
        "legit_flagged_high": high[legit].mean() if legit.any() else float("nan"),
        "fraud_rated_low": (scored["risk"][fraud] == "low").mean() if fraud.any() else float("nan"),
    }


def by_group(scored, group_col):
    rows = []
    for (group, y), part in scored.groupby([group_col, "y"]):
        rows.append({
            group_col: group,
            "label": "fraud" if y else "legit",
            "n": len(part),
            "mean_score": part["score"].mean(),
            "high": (part["risk"] == "high").mean(),
            "medium": (part["risk"] == "medium").mean(),
            "low": (part["risk"] == "low").mean(),
        })
    return pd.DataFrame(rows)


def worst(scored, text_col, n):
    """The most confident mistakes: legit rows scored highest, fraud rows scored lowest."""
    legit = scored[scored["y"] == 0].nlargest(n, "score")
    fraud = scored[scored["y"] == 1].nsmallest(n, "score")
    snippet = lambda part: part.assign(text=part[text_col].astype(str).str.slice(0, 100))[["score", "risk", "text"]]
    return snippet(legit), snippet(fraud)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--positive-label", help="value of the label column meaning fraud; omit if labels are 0/1")
    p.add_argument("--group-col", default="provenance", help="breakdown column; skipped if absent")
    p.add_argument("--show", type=int, default=5, help="how many worst mistakes to print per class")
    p.add_argument("--out", type=Path, help="write per-row scores to this csv")
    args = p.parse_args()

    model = joblib.load(args.model)
    scored = score_csv(pd.read_csv(args.csv), model, args.text_col, args.label_col, args.positive_label)

    print(f"model: tier {model.tier}, high >= {model.threshold:.3f}, medium >= {model.review_threshold:.3f}")
    for name, value in summarize(scored).items():
        print(f"  {name:20s} {value:.3f}" if isinstance(value, float) else f"  {name:20s} {value}")
    if args.group_col in scored.columns:
        print(f"\nby {args.group_col}:")
        print(by_group(scored, args.group_col).to_string(index=False, float_format="%.2f"))
    false_alarms, misses = worst(scored, args.text_col, args.show)
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print("\nlegit rows scored highest:")
        print(false_alarms.to_string(index=False, float_format="%.3f"))
        print("\nfraud rows scored lowest:")
        print(misses.to_string(index=False, float_format="%.3f"))
    if args.out:
        scored.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
