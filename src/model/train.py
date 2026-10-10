"""Train tiers A-C (+ shortcut ablations), calibrate, pick thresholds, write artifacts.

    python -m src.model.train                 # uses A's data/processed/splits.json
    python -m src.model.train --dev-split     # until splits.json exists; numbers not reportable

    python -m src.model.train --tiers D       # DistilBERT; needs requirements-bert.txt, ~35 min on an M2

Per tier: out-of-fold scores on the train/val folds drive calibration and
threshold selection; the test ids are only ever scored once, at the end.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.model.data import LABEL_COL, RAW_CSV, ROOT, SPLITS_JSON, dev_splits, load_postings, load_splits
from src.model.features import OPTIONAL_TIERS, TIERS, build_frame, make_pipeline
from src.model.predict import FraudModel, apply_calibrator, fit_calibrator


def oof_scores(tier, X, y, fold_positions):
    scores = np.empty(len(X))
    everything = np.arange(len(X))
    for val in fold_positions:
        train = np.setdiff1d(everything, val)
        pipe = make_pipeline(tier).fit(X.iloc[train], y.iloc[train])
        scores[val] = pipe.predict_proba(X.iloc[val])[:, 1]
    return scores


def pick_threshold(y, scores, precision_target):
    """Lowest threshold whose precision meets the target (i.e. max recall at that precision)."""
    precision, _, thresholds = precision_recall_curve(y, scores)
    ok = np.flatnonzero(precision[:-1] >= precision_target)
    if len(ok) == 0:
        best = int(np.argmax(precision[:-1]))
        print(f"  WARNING: precision {precision_target:.2f} unreachable; using best available {precision[best]:.2f}")
        return float(thresholds[best])
    return float(thresholds[ok[0]])


def train_tier(tier, X, y, fold_positions, X_test, y_test, args):
    if tier in OPTIONAL_TIERS:
        # Too slow for k+1 fits: train once on all folds but the last, calibrate on the last.
        val = fold_positions[-1]
        train = np.setdiff1d(np.arange(len(X)), val)
        pipeline = make_pipeline(tier).fit(X.iloc[train], y.iloc[train])
        raw_val, y_val = pipeline.predict_proba(X.iloc[val])[:, 1], y.iloc[val]
    else:
        raw_val, y_val = oof_scores(tier, X, y, fold_positions), y
        pipeline = make_pipeline(tier).fit(X, y)
    calibrator = fit_calibrator(args.calibration, raw_val, y_val)
    cal_val = apply_calibrator(calibrator, raw_val)
    threshold = pick_threshold(y_val, cal_val, args.precision_target)
    model = FraudModel(
        tier=tier,
        pipeline=pipeline,
        calibrator=calibrator,
        threshold=threshold,
        review_threshold=min(args.review_score, threshold),
    )
    test_scores = model.score(X_test)
    flagged = test_scores >= model.threshold
    row = {
        "tier": tier,
        "val_pr_auc": average_precision_score(y_val, raw_val),
        "test_pr_auc": average_precision_score(y_test, test_scores),
        "test_roc_auc": roc_auc_score(y_test, test_scores),
        "test_brier": brier_score_loss(y_test, test_scores),
        "threshold": model.threshold,
        "test_precision": precision_score(y_test, flagged, zero_division=0),
        "test_recall": recall_score(y_test, flagged),
    }
    return model, row


def to_markdown(table, note):
    lines = [note, "", "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=RAW_CSV)
    p.add_argument("--splits", type=Path, default=SPLITS_JSON)
    p.add_argument("--dev-split", action="store_true", help="ignore splits.json; random non-grouped split (leaky)")
    p.add_argument("--tiers", nargs="+", default=[t for t in TIERS if t not in OPTIONAL_TIERS], choices=list(TIERS))
    # Sigmoid (Platt) by default: same Brier score as isotonic here, without isotonic's tied step scores.
    p.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="sigmoid")
    p.add_argument("--precision-target", type=float, default=0.90, help="precision required for the 'high' label")
    # A fixed probability, not a precision target: cumulative precision of 50% is already met at a
    # ~3% score, which would label uninformative text (scores near the base rate) as medium.
    p.add_argument("--review-score", type=float, default=0.10, help="calibrated score at which 'medium' starts")
    # Text-only by default: pasted postings/DMs in the demo carry no EMSCAD metadata.
    p.add_argument("--deploy-tier", default="B", choices=list(TIERS), help="tier saved as models/model.joblib")
    p.add_argument("--out", type=Path, default=ROOT / "models")
    args = p.parse_args()

    df = load_postings(args.raw)
    if args.dev_split:
        print("WARNING: --dev-split is not group-aware; results are optimistic. Do not report them.")
        splits = dev_splits(df)
    else:
        splits = load_splits(args.splits, df.index)

    trainval_ids = [i for fold in splits["folds"] for i in fold]
    position = {job_id: pos for pos, job_id in enumerate(trainval_ids)}
    fold_positions = [np.array([position[i] for i in fold]) for fold in splits["folds"]]

    frame = build_frame(df)
    X, y = frame.loc[trainval_ids], df.loc[trainval_ids, LABEL_COL]
    X_test, y_test = frame.loc[splits["test"]], df.loc[splits["test"], LABEL_COL]
    print(f"train/val: {len(y)} rows, {int(y.sum())} fraud ({y.mean():.1%}) | "
          f"test: {len(y_test)} rows, {int(y_test.sum())} fraud ({y_test.mean():.1%})")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tier in args.tiers:
        print(f"tier {tier} ...")
        model, row = train_tier(tier, X, y, fold_positions, X_test, y_test, args)
        rows.append(row)
        joblib.dump(model, args.out / f"model_{tier}.joblib")
        if tier == args.deploy_tier:
            joblib.dump(model, args.out / "model.joblib")

    table = pd.DataFrame(rows)
    split_note = "**DEV SPLIT (not group-aware) — do not report.**" if args.dev_split else f"Splits: `{args.splits.name}`."
    note = (f"{split_note} Calibration: {args.calibration}. "
            f"Threshold = lowest with validation precision >= {args.precision_target:.2f} "
            f"(out-of-fold for A-C, last fold only for D).")
    (args.out / "ablation.md").write_text(to_markdown(table, note))
    print(table.to_string(index=False, float_format="%.3f"))
    if args.deploy_tier not in args.tiers:
        print(f"WARNING: deploy tier {args.deploy_tier} was not trained; models/model.joblib not updated")


if __name__ == "__main__":
    main()
