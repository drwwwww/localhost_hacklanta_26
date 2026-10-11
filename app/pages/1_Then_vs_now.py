"""Demo beat 5: the same model on EMSCAD's held-out test set vs the modern challenge set."""

import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.gap import CHALLENGE_CSV, deployed_gap_table  # noqa: E402
from src.model.predict import MODEL_PATH, load_model  # noqa: E402
from background import render_background  # noqa: E402
from theme import apply_theme, render_nav  # noqa: E402

DATASETS = {"EMSCAD test": "2012-2014 postings", "Challenge set": "Modern scams"}
MEASURES = {
    "recall_high": "Scams caught as high risk",
    "fraud_rated_low": "Scams rated low risk",
    "legit_flagged_high": "Real jobs flagged high risk",
}
# Blue/orange pair, one step per theme; checked for colour-blind separation on both surfaces.
COLORS = {"light": ["#2a78d6", "#eb6834"], "dark": ["#7d9cc4", "#d0912f"]}
INK = {"light": "#0b0b0b", "dark": "#ece8df"}


@st.cache_data(show_spinner="Scoring both datasets...")
def load_gap(model_mtime):  # mtime: reload after a retrain
    return deployed_gap_table()


st.set_page_config(page_title="Then vs now", page_icon="🔎", layout="wide")
render_background()
apply_theme()
render_nav(skip_target="#main")
page = st.container(key="page")
page.markdown('<div id="main"></div>', unsafe_allow_html=True)
page.title("Trained on 2014, tested on today")

with page:

    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.model.train` from the repo root, then reload.")
        st.stop()

    model = load_model()
    try:
        table = load_gap(MODEL_PATH.stat().st_mtime).set_index("dataset")
    except FileNotFoundError as e:
        st.error(f"Missing data file: `{e.filename}`. This page needs the raw EMSCAD csv, splits.json and the challenge set.")
        st.stop()

    old, new = table.loc["EMSCAD test"], table.loc["Challenge set"]
    st.caption(f"The same model and the same thresholds, scored on {old['n']:,} held-out postings from 2012-2014 "
               f"and on {new['n']} postings and recruiter messages written from current scam alerts.")

    left, right = st.columns(2)
    left.metric("Scams caught, 2012-2014 postings", f"{old['recall_high']:.0%}")
    right.metric("Scams caught, modern scams", f"{new['recall_high']:.0%}",
                 delta=f"{(new['recall_high'] - old['recall_high']) * 100:+.0f} pts")

    long = (table[list(MEASURES)].rename(columns=MEASURES).rename(index=DATASETS)
            .rename_axis(index="dataset", columns="measure").stack().rename("share").reset_index())
    try:
        theme = st.context.theme.type or "light"
    except AttributeError:
        theme = "light"

    x = alt.X("measure:N", sort=list(MEASURES.values()), title=None, axis=alt.Axis(labelAngle=0, labelLimit=220))
    offset = alt.XOffset("dataset:N", sort=list(DATASETS.values()))
    bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x,
        xOffset=offset,
        y=alt.Y("share:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("dataset:N", sort=list(DATASETS.values()), scale=alt.Scale(range=COLORS[theme]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=300)),
        tooltip=[alt.Tooltip("dataset:N", title="Dataset"), alt.Tooltip("measure:N", title="Measure"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")],
    )
    labels = alt.Chart(long).mark_text(dy=-8, color=INK[theme]).encode(x=x, xOffset=offset, y="share:Q", text=alt.Text("share:Q", format=".0%"))
    st.altair_chart((bars + labels).properties(height=340), use_container_width=True)

    st.write(f"High risk means a score of {model.threshold:.0%} or more, a threshold picked for 90% precision on the "
             "2012-2014 data. On modern scams the model's ranking gets worse too: PR-AUC goes from "
             f"{old['pr_auc']:.2f} to {new['pr_auc']:.2f}.")

    with st.expander("Numbers behind the chart"):
        numbers = table.drop(columns="tier").rename(index=DATASETS)
        st.dataframe(numbers.apply(lambda col: col.map("{:,}".format if col.name.startswith("n") else "{:.3f}".format)).T)
    st.caption(f"Model: tier {model.tier}. Challenge set: `{CHALLENGE_CSV.name}`, written from public scam alerts, "
               "so it measures known modern patterns rather than a sample of real traffic.")
