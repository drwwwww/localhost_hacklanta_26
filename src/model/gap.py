"""EMSCAD test set vs C's challenge set, side by side: how far the model's numbers drop on modern scams.

    python -m src.model.gap                  # deployed text tier (B)
    python -m src.model.gap --tiers B D E    # any tiers with a models/model_<tier>.joblib
    python -m src.model.gap --save-deployed  # refresh models/gap_deployed.json after changing the demo model

Both sets are scored with the thresholds stored in the artifact (chosen on EMSCAD
validation folds), so the challenge-set rows show transfer, not a re-tuned model.
Text tiers only: the challenge set has no EMSCAD metadata for tiers A/C to use.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.model.data import LABEL_COL, RAW_CSV, ROOT, SPLITS_JSON, load_postings, load_splits
from src.model.evaluate import score_csv, summarize
from src.model.features import build_frame
from src.model.verify_models import sha256

CHALLENGE_CSV = ROOT / "data" / "challenge_set" / "challenge_set.csv"
MODELS_DIR = ROOT / "models"
DEPLOYED_MODEL = MODELS_DIR / "model.joblib"
DEPLOYED_GAP = MODELS_DIR / "gap_deployed.json"


def emscad_test_scored(model, raw=RAW_CSV, splits=SPLITS_JSON):
    df = load_postings(raw)
    test_ids = load_splits(splits, df.index)["test"]
    scores = model.score(build_frame(df.loc[test_ids]))
    out = pd.DataFrame({"score": scores, "y": df.loc[test_ids, LABEL_COL].to_numpy()})
    out["risk"] = out["score"].map(model.risk_label)
    return out


def gap_table(tiers=("B",), models_dir=MODELS_DIR, challenge_csv=CHALLENGE_CSV, raw=RAW_CSV, splits=SPLITS_JSON):
    """One row per (tier, dataset) with the columns of evaluate.summarize."""
    challenge = pd.read_csv(challenge_csv)
    rows = []
    for tier in tiers:
        model = joblib.load(Path(models_dir) / f"model_{tier}.joblib")
        scored = {
            "EMSCAD test": emscad_test_scored(model, raw, splits),
            "Challenge set": score_csv(challenge, model, "text", "label"),
        }
        for dataset, part in scored.items():
            rows.append({"tier": tier, "dataset": dataset, **summarize(part)})
    return pd.DataFrame(rows)


def _inputs():
    return {"model_sha256": sha256(DEPLOYED_MODEL), "challenge_sha256": sha256(CHALLENGE_CSV)}


def save_deployed_gap():
    model = joblib.load(DEPLOYED_MODEL)
    rows = [{"tier": model.tier, "dataset": "EMSCAD test", **summarize(emscad_test_scored(model))},
            {"tier": model.tier, "dataset": "Challenge set",
             **summarize(score_csv(pd.read_csv(CHALLENGE_CSV), model, "text", "label"))}]
    DEPLOYED_GAP.write_text(json.dumps({**_inputs(), "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


def deployed_gap_table():
    """The gap table for models/model.joblib, read from the committed copy when it is current.

    Scoring the EMSCAD test set means loading the whole raw csv, about 600 MB at peak,
    which a small web server does not have. The saved copy records the hashes of the model
    and the challenge set it was computed from; if either has changed it is recomputed here.
    """
    if DEPLOYED_GAP.exists():
        saved = json.loads(DEPLOYED_GAP.read_text(encoding="utf-8"))
        if all(saved.get(k) == v for k, v in _inputs().items()):
            return pd.DataFrame(saved["rows"])
    return save_deployed_gap()


def to_markdown(table):
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=["B"])
    p.add_argument("--models", type=Path, default=MODELS_DIR)
    p.add_argument("--challenge", type=Path, default=CHALLENGE_CSV)
    p.add_argument("--raw", type=Path, default=RAW_CSV)
    p.add_argument("--splits", type=Path, default=SPLITS_JSON)
    p.add_argument("--save-deployed", action="store_true", help="recompute models/gap_deployed.json for models/model.joblib")
    args = p.parse_args()

    if args.save_deployed:
        print(save_deployed_gap().to_string(index=False, float_format="%.3f"))
        return
    table = gap_table(args.tiers, args.models, args.challenge, args.raw, args.splits)
    (args.models / "gap.md").write_text(to_markdown(table))
    print(table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
