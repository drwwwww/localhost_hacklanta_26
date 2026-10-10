"""Paste-and-check demo.    streamlit run app/main.py"""

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.predict import MODEL_PATH, load_model, predict  # noqa: E402

# Swap for C's rule module once it lands; see redflags_placeholder.py for the contract.
from redflags_placeholder import find_red_flags  # noqa: E402

# Below this the text model has too little to go on (it misses most fraud under ~15 words).
MIN_WORDS = 15

EXAMPLES = {
    "Scam posting": (
        "Data Entry Clerk - Work From Home. Earn $4,500 weekly processing simple forms from your own computer. "
        "No experience required and no interview needed, you can start today. We provide full training. To secure "
        "your position, pay a $60 registration fee for your starter kit, which is refunded with your first paycheck. "
        "Send your full name, address and bank details to our hiring manager on Telegram to get started immediately."
    ),
    "Real posting": (
        "Software Engineering Intern, Summer 2027 - Atlanta, GA. You will join our platform team and work with a "
        "mentor to build and test backend services in Python and Go, write unit tests, and take part in code "
        "reviews. Requirements: currently pursuing a BS or MS in Computer Science or a related field, coursework "
        "in data structures and algorithms, and familiarity with Git. This is a paid 12-week internship. Apply "
        "through our careers page; interviews include one technical screen and a virtual onsite."
    ),
    "Hard negative": (
        "Sales Representative (commission only) for a residential solar installer. There is no base salary; "
        "representatives earn 8% of each closed contract, paid monthly. You will canvass assigned neighborhoods, "
        "run in-home consultations and prepare quotes. Two weeks of paid product training are provided at our "
        "Marietta office. A valid driver's license and reliable vehicle are required. Interviews are held in "
        "person with the regional sales manager."
    ),
    "Recruiter DM": (
        "Hello, I got your resume from the university career portal. I am a professor travelling abroad and need "
        "a student assistant to run errands, $400 weekly, flexible hours. I will mail you a check for supplies; "
        "deposit the check, keep your first week's pay and send the remainder to my vendor by Zelle. Reply with "
        "your full name, address and cell number and text me at this number, not by email."
    ),
}

VERDICTS = {
    "high": (st.error, "High risk", "This reads like known job scams. Don't send money or personal details."),
    "medium": (st.warning, "Some risk", "Parts of this resemble scam postings. Verify the employer independently before replying."),
    "low": (st.success, "Low risk", "Nothing here matches the scam patterns the model knows. That is not proof it's safe."),
}


def highlight(text, flags):
    parts, pos = [], 0
    for f in flags:
        if f["start"] < pos:  # overlapping match; the earlier one already covers it
            continue
        parts.append(html.escape(text[pos:f["start"]]))
        parts.append(f'<mark title="{html.escape(f["label"])}">{html.escape(text[f["start"]:f["end"]])}</mark>')
        pos = f["end"]
    parts.append(html.escape(text[pos:]))
    return "".join(parts).replace("\n", "<br>")


st.set_page_config(page_title="Fake Job Detector", page_icon="🔎")
st.title("Is this job real?")
st.caption("Paste a job posting or a recruiter's message to check it for signs of a scam.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.model.train` from the repo root, then reload.")
    st.stop()

st.write("Try an example:")
for col, name in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    if col.button(name, use_container_width=True):
        st.session_state["text"] = EXAMPLES[name]

text = st.text_area("Posting or message", key="text", height=220, placeholder="Paste the full text here...")

if text.strip():
    score, risk = predict(text)
    flags = find_red_flags(text)
    banner, headline, advice = VERDICTS[risk]
    banner(f"**{headline}.** {advice}")

    left, right = st.columns(2)
    left.metric("Model's fraud estimate", f"{score:.0%}")
    right.metric("Red flags found", len(flags))
    if len(text.split()) < MIN_WORDS:
        st.info(f"This is under {MIN_WORDS} words, which is too little text for the model to judge reliably. "
                "Paste the full posting or the whole conversation if you can.")

    if flags:
        st.subheader("Red flags")
        st.markdown(f'<div style="line-height:1.7">{highlight(text, flags)}</div>', unsafe_allow_html=True)
        st.write("")
        seen = set()
        for f in flags:
            if f["label"] not in seen:
                seen.add(f["label"])
                st.markdown(f"- **{f['label']}:** {f['why']}")

    model = load_model()
    st.caption(f"Model: tier {model.tier}, trained on EMSCAD job postings from 2012-2014. High risk starts at "
               f"{model.threshold:.0%}, some risk at {model.review_threshold:.0%}. Modern scams may look different.")
