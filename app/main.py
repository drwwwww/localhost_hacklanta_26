"""Paste-and-check demo.    streamlit run app/main.py"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from background import render_background  # noqa: E402
from theme import BRAND, apply_theme, brand_html, page_html, render_nav  # noqa: E402
from src.model.predict import MODEL_PATH, load_model, predict  # noqa: E402
from src.model.verdict import combined_risk  # noqa: E402
from src.redflags import MAX_CHARS, find_red_flags  # noqa: E402
from src.redflags.render import highlight_html  # noqa: E402

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

# headline, advice, stamp word, ink color, stamp tilt (degrees)
VERDICTS = {
    "high": ("High risk", "This reads like known job scams. Don't send money or personal details.",
              "Flagged", "#c1443c", -3),
    "medium": ("Some risk", "Parts of this resemble scam postings. Verify the employer independently before replying.",
               "Verify further", "#d0912f", 2),
    "low": ("Low risk", "Nothing here matches the scam patterns the model knows. That is not proof it's safe.",
            "Looks clear", "#5fa37a", -1.5),
}


def render_verdict(risk):
    headline, advice, stamp, ink, tilt = VERDICTS[risk]
    st.markdown(f"""
    <div class="verdict">
        <div class="stamp" style="--stamp:{ink}; --stamp-tilt:{tilt}deg">{stamp}</div>
        <div>
            <p class="verdict-headline">{headline}.</p>
            <p class="verdict-advice">{advice}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Layout and type follow the Orchid reference: a serif display face at regular weight,
# a quiet sans for everything else, a tall hero with copy anchored bottom-left, then
# centered section heads over card grids, one inverted band with alternating feature
# rows, an FAQ accordion, and a closing CTA band. Colors stay dark. GhostFibers fills
# the page behind the hero and the closing band; every section between them is solid.
st.set_page_config(page_title=BRAND, page_icon="🔎", layout="wide")
render_background()
apply_theme()
render_nav()
st.markdown("""
<style>
/* Hero: tall, copy anchored bottom-left like the reference, evidence card right */
.hero {
    min-height: 100vh;
    display: flex;
    align-items: flex-end;
    padding: 8rem 0 5.5rem;
    /* Scrim: keeps the copy at AA contrast over the brightest fibers, and fades the
       hero into the solid section below instead of ending on a hard edge. */
    background:
        linear-gradient(180deg, transparent 65%, var(--ink)),
        linear-gradient(90deg, rgba(8,11,19,0.7), rgba(8,11,19,0.15) 60%);
}
.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr);
    gap: 4rem;
    align-items: end;
}
.hero-sub { color: #c4c9d4; font-size: 1.04rem; line-height: 1.6; max-width: 31rem; margin: 1.5rem 0 2rem !important; }
.hero-actions { display: flex; gap: 0.75rem; flex-wrap: wrap; }

.evidence { position: relative; margin-bottom: 0.4rem; }
.evidence-card {
    background: rgba(15,19,32,0.82);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.4rem 1.5rem;
    transform: rotate(-1.6deg);
    box-shadow: 0 24px 60px rgba(0,0,0,0.45);
}
.evidence-label { font-family: var(--mono); font-size: 0.72rem; color: var(--dim); margin: 0 0 0.8rem; }
.evidence-text { font-family: var(--mono); font-size: 0.84rem; line-height: 1.75; color: #c7cbd6; margin: 0; }
.evidence-tag {
    margin: 1rem 0 0;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-size: 0.84rem;
    color: #e3ad5b;
    line-height: 1.5;
}
.stamp-chip {
    position: absolute;
    top: -0.9rem;
    right: 0.8rem;
    transform: rotate(7deg);
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--red);
    border: 2px double var(--red);
    border-radius: 4px;
    padding: 0.3rem 0.55rem;
    background: var(--ink);
    user-select: none;
}

/* Sources strip, standing in for the reference's logo row */
.sources { background: var(--ink); padding: 3.2rem 0 0; text-align: center; }
.sources-label { color: var(--dim); font-size: 0.8rem; margin: 0 0 1.2rem; }
.sources-row { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.6rem 3rem; }
.sources-row span { font-family: var(--serif); font-size: 1.15rem; color: var(--muted); }

/* Red-flag cards: 2x2 grid of tall tiles, label bottom-left like the reference */
.card-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.flag-card {
    position: relative;
    min-height: 19rem;
    border-radius: 10px;
    overflow: hidden;
    padding: 1.6rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border: 1px solid var(--line);
    background:
        radial-gradient(circle at 75% 20%, var(--glow), transparent 60%),
        radial-gradient(circle, rgba(236,232,223,0.06) 1px, transparent 1.3px) 0 0 / 14px 14px,
        var(--card);
}
.flag-snippet { font-family: var(--mono); font-size: 0.82rem; line-height: 1.7; color: #c7cbd6; max-width: 22rem; margin: 0; }
.flag-title { font-family: var(--serif); font-size: 1.6rem; color: var(--paper); margin: 0; }
.flag-why { color: var(--muted); font-size: 0.88rem; line-height: 1.5; margin: 0.35rem 0 0; max-width: 26rem; }

/* Stat statement, standing in for the reference's testimonial */
.statement { background: var(--ink); padding: 2rem 0 7rem; text-align: center; }
.statement .serif-lg { max-width: 46rem; margin: 0 auto; }
.statement-source { color: var(--dim); font-size: 0.82rem; margin: 1.2rem 0 0; }
.statement-link { margin: 1.8rem 0 0 !important; }

/* Inverted band with alternating feature rows */
.band { background: var(--panel); padding: 7rem 0; scroll-margin-top: 5rem; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.band-title { margin-bottom: 4rem; }
.row { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: 4rem; align-items: center; }
.row + .row { margin-top: 2.5rem; padding-top: 2.5rem; border-top: 1px dashed var(--line); }
.row-title { font-family: var(--serif); font-size: 1.55rem; color: var(--paper); margin: 0; font-weight: 400; }
.row-copy { color: var(--muted); font-size: 0.95rem; line-height: 1.6; margin: 0.7rem 0 1.2rem; }
.checks { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.55rem; }
.checks li { display: flex; gap: 0.65rem; align-items: baseline; color: #c4c9d4; font-size: 0.9rem; }
.checks li::before { content: ""; flex-shrink: 0; width: 0.5rem; height: 0.5rem; border: 1px solid var(--steel); border-radius: 2px; transform: translateY(-1px); }
.mock {
    background: var(--ink);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.6rem;
    min-height: 15rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.9rem;
}
.mock-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.mock-chips span { border: 1px solid var(--line); border-radius: 6px 6px 0 0; padding: 0.35rem 0.7rem; font-size: 0.76rem; color: var(--muted); }
.mock-field { border: 1px solid var(--line); border-radius: 6px; padding: 0.9rem; font-family: var(--mono); font-size: 0.78rem; line-height: 1.7; color: #aeb4c2; }
.mock-verdict { display: flex; align-items: center; gap: 0.9rem; }
.mock-verdict .stamp-inline {
    font-family: var(--mono); font-weight: 700; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--red); border: 2px double var(--red); border-radius: 4px; padding: 0.3rem 0.55rem; transform: rotate(-3deg);
}
.mock-verdict p { margin: 0; color: var(--paper); font-size: 0.88rem; }
.mock-step { display: flex; gap: 0.75rem; align-items: center; color: #c4c9d4; font-size: 0.88rem; }
.mock-step .tick { width: 1.1rem; height: 1.1rem; border-radius: 50%; border: 1px solid var(--steel); flex-shrink: 0; }

/* Tool */
.tool-head { background: var(--ink); padding: 7rem 0 2rem; text-align: center; scroll-margin-top: 4rem; }
.st-key-tool { background: var(--ink); padding: 0 max(1.5rem, calc((100% - 760px) / 2)) 6rem; }

/* Where-scams-find-you cards */
.where-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
.where-card {
    min-height: 22rem;
    border-radius: 10px;
    border: 1px solid var(--line);
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: radial-gradient(circle at 50% 110%, var(--glow), transparent 65%), var(--card);
}
.where-tag { align-self: flex-start; font-size: 0.78rem; color: var(--paper); border: 1px solid var(--line); border-radius: 999px; padding: 0.25rem 0.7rem; }
.where-card p { color: #c4c9d4; font-size: 0.92rem; line-height: 1.55; margin: 0; }

/* About: visual left, statement right */
.about-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 4.5rem; align-items: center; }
.about-visual {
    aspect-ratio: 4 / 3;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: radial-gradient(circle at 30% 70%, rgba(208,145,47,0.28), transparent 60%), var(--card);
    display: flex;
    align-items: center;
    justify-content: center;
}
.about-copy p { color: var(--muted); font-size: 0.96rem; line-height: 1.65; margin: 1rem 0 0; }

/* FAQ */
.faq { max-width: 44rem; margin: 0 auto; }
.faq details { border-bottom: 1px solid var(--line); }
.faq summary {
    list-style: none;
    cursor: pointer;
    padding: 1.15rem 0;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: var(--paper);
    font-size: 1rem;
}
.faq summary::-webkit-details-marker { display: none; }
.faq summary::after { content: "+"; color: var(--muted); transition: transform 0.2s ease; }
.faq details[open] summary::after { transform: rotate(45deg); }
.faq details p { color: var(--muted); font-size: 0.94rem; line-height: 1.65; margin: 0 0 1.2rem; max-width: 38rem; }
.faq a { color: var(--paper) !important; }

/* Closing band: transparent so the fibers show through again */
.cta { padding: 9rem 0; text-align: center; background: linear-gradient(var(--ink), rgba(8,11,19,0.35) 30%, rgba(8,11,19,0.35) 70%, var(--ink)); }
.cta .hero-actions { justify-content: center; margin-top: 2rem; }

.footer { background: var(--ink); border-top: 1px solid var(--line); padding: 3.5rem 0 2.5rem; }
.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 2rem; }
.footer h4 { color: var(--paper); font-size: 0.86rem; font-weight: 500; margin: 0 0 0.9rem; }
.footer a { display: block; color: var(--muted) !important; text-decoration: none !important; font-size: 0.86rem; margin-bottom: 0.55rem; }
.footer a:hover { color: var(--paper) !important; }
.footer-note { color: var(--dim); font-size: 0.8rem; line-height: 1.6; max-width: 24rem; margin: 0.9rem 0 0; }
.footer-base { color: var(--dim); font-size: 0.78rem; margin-top: 3rem; }

@media (max-width: 860px) {
    .hero-grid, .row, .about-grid, .footer-grid { grid-template-columns: 1fr; gap: 2.5rem; }
    .card-grid, .where-grid { grid-template-columns: 1fr; }
    .evidence-card { transform: none; }
    .sec, .band { padding: 5rem 0; }
    .wrap { padding: 0 1.25rem; }
}
@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    * { transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# Motion and finish layer. One motif carries the whole page: the tool's own amber
# underline drawing in under a suspicious phrase, and its ink stamp landing. Each
# section performs that mechanism a different way instead of sharing one generic
# entrance. Shape rule: pills for actions, 12px for surfaces, 6px for fields and
# chips, 4px for stamps. Scroll reveals need app/frontend's enhancer (it adds
# html.js-reveal); without it every element simply renders in its final state.
st.markdown("""
<style>
/* The underline motif (C's renderer emits plain <mark> in the tool): a background stroke (not a border) so it can draw across
   line wraps, which a transform on a pseudo-element can't do for inline text. */
mark.flag, .st-key-tool mark {
    border-bottom: none !important;
    padding-bottom: 1px !important;
    background-image: linear-gradient(var(--amber), var(--amber)) !important;
    background-repeat: no-repeat !important;
    background-position: 0 100% !important;
    background-size: 100% 2px !important;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
}
html.js-reveal [data-reveal]:not(.in) mark.flag { background-size: 0% 2px !important; }
[data-reveal] mark.flag { transition: background-size 650ms var(--ease-out) 250ms; }

@keyframes rise { from { opacity: 0; transform: translateY(0.4em); filter: blur(8px); } }
@keyframes card-in { from { opacity: 0; transform: translateY(18px) rotate(-1.6deg); } }
@keyframes draw { from { background-size: 0% 2px; } }
@keyframes thump {
    0% { opacity: 0; transform: scale(1.45) rotate(var(--tilt-from)); }
    100% { opacity: 1; transform: scale(1) rotate(var(--tilt)); }
}
@keyframes blink { 50% { opacity: 0; } }
@keyframes marquee { to { transform: translateX(-50%); } }

/* Hero: a single load sequence. Lines rise out of blur, the card settles, the
   phrase gets underlined, then the stamp lands. Runs once per visit. */
.h-line { display: block; animation: rise 800ms var(--ease-out) both; }
.h-line:nth-child(2) { animation-delay: 90ms; }
.hero-sub { animation: rise 800ms var(--ease-out) 200ms both; }
.hero .hero-actions { animation: rise 800ms var(--ease-out) 300ms both; }
.evidence-card { animation: card-in 900ms var(--ease-out) 380ms both; }
.evidence-card mark.flag { animation: draw 700ms var(--ease-out) 1050ms both; }
.stamp-chip { --tilt: 7deg; --tilt-from: 16deg; animation: thump 260ms var(--ease-out) 1650ms both; }

/* Two-line headline at desktop (taste-skill hero rule): wider copy column, capped size */
.hero .serif-xl { font-size: clamp(2.4rem, 4.6vw, 3.6rem) !important; }
@media (min-width: 861px) {
    .hero-grid { grid-template-columns: minmax(0, 1.5fr) minmax(0, 0.75fr); }
    .h-line { white-space: nowrap; }
}

/* Pointer spotlight on cards (position set by the enhancer), fine pointers only */
.spot { position: relative; isolation: isolate; }
.spot::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    z-index: -1;
    opacity: 0;
    background: radial-gradient(420px circle at var(--mx, 50%) var(--my, 50%), rgba(125,156,196,0.16), transparent 45%);
    transition: opacity 300ms ease;
}
@media (hover: hover) and (pointer: fine) {
    .spot:hover::before { opacity: 1; }
    .spot { transition: transform 300ms var(--ease-out), border-color 300ms ease; }
    .spot:hover { transform: translateY(-3px); border-color: rgba(236,232,223,0.2); }
}

/* Staggered reveals for groups, once */
html.js-reveal [data-reveal] > .stagger {
    transition: opacity 500ms var(--ease-out), transform 500ms var(--ease-out);
    transition-delay: calc(var(--i, 0) * 70ms);
}
html.js-reveal [data-reveal]:not(.in) > .stagger { opacity: 0; transform: translateY(14px); }

/* Red-flag cards: asymmetric rhythm instead of four identical tiles */
.card-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.card-grid > :nth-child(1), .card-grid > :nth-child(4) { grid-column: span 3; }
.card-grid > :nth-child(2), .card-grid > :nth-child(3) { grid-column: span 2; }
.flag-card { border-radius: 12px; }
.flag-card .flag-snippet { font-size: 0.9rem; }
.flag-card:nth-child(1) .flag-snippet, .flag-card:nth-child(4) .flag-snippet { font-size: 1.05rem; max-width: 28rem; }

/* Statement: words brighten as it scrolls through, like it's being read out */
.statement .serif-lg em { font-style: italic; color: var(--paper); }
.statement .w { display: inline; }
@supports (animation-timeline: view()) {
    @media (prefers-reduced-motion: no-preference) {
        .statement .w {
            animation: word-on linear both;
            animation-timeline: view();
            animation-range: cover calc(18% + var(--i) * 2.2%) cover calc(30% + var(--i) * 2.2%);
        }
    }
}
@keyframes word-on { from { opacity: 0.16; } to { opacity: 1; } }

/* Example-message marquee (the page's only marquee) */
.wild { background: var(--ink); padding: 0 0 6.5rem; overflow: hidden; }
.wild-note { text-align: center; color: var(--dim); font-size: 0.8rem; margin: 0 0 1.6rem !important; }
.marquee {
    display: flex;
    overflow: hidden;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent);
}
.marquee + .marquee { margin-top: 0.9rem; }
.marquee-track { display: flex; gap: 0.9rem; padding-right: 0.9rem; flex-shrink: 0; animation: marquee 60s linear infinite; }
.marquee.reverse .marquee-track { animation-direction: reverse; animation-duration: 70s; }
@media (hover: hover) { .marquee:hover .marquee-track { animation-play-state: paused; } }
.msg {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.75rem 1rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel);
    white-space: nowrap;
}
.msg-src { font-size: 0.72rem; color: var(--muted); border: 1px solid var(--line); border-radius: 6px; padding: 0.15rem 0.5rem; }
.msg-text { font-family: var(--mono); font-size: 0.82rem; color: #c7cbd6; }

/* How-it-works mocks perform their step when they scroll in */
.mock { border-radius: 12px; }
.mock-field .caret { display: inline-block; width: 1px; height: 1em; background: var(--amber); margin-left: 2px; vertical-align: -0.15em; animation: blink 1.1s steps(1) infinite; }
.mock-verdict .stamp-inline { --tilt: -3deg; --tilt-from: -14deg; }
html.js-reveal [data-reveal]:not(.in) .stamp-inline { opacity: 0; transform: scale(1.45) rotate(-14deg); }
html.js-reveal [data-reveal].in .stamp-inline { animation: thump 260ms var(--ease-out) 900ms both; }
.mock-step .tick { position: relative; transition: background 300ms ease, border-color 300ms ease; }
.mock-step .tick::after {
    content: ""; position: absolute; left: 0.32rem; top: 0.14rem; width: 0.3rem; height: 0.55rem;
    border: solid var(--ink); border-width: 0 2px 2px 0; transform: rotate(45deg);
}
[data-reveal].in .mock-step .tick, html:not(.js-reveal) .mock-step .tick { background: var(--steel); border-color: var(--steel); transition-delay: calc(400ms + var(--i, 0) * 350ms); }

/* Where-scams cards get a real example inside each */
.where-card { border-radius: 12px; }
.where-quote { font-family: var(--mono); font-size: 0.8rem; line-height: 1.65; color: #aeb4c2; border-left: 1px solid var(--line); padding-left: 0.9rem; margin: 0 0 1.1rem !important; }

/* About: the three verdict stamps, dropped like a pile of case notes */
.about-visual { position: relative; overflow: hidden; border-radius: 12px; }
.pile { position: absolute; white-space: nowrap; font-family: var(--mono); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; border-radius: 4px; padding: 0.55rem 0.9rem; background: rgba(8,11,19,0.7); }
.pile.p1 { --tilt: -9deg; --tilt-from: -22deg; top: 16%; left: 10%; color: var(--red); border: 3px double var(--red); font-size: clamp(0.8rem, 1.5vw, 1.05rem); }
.pile.p2 { --tilt: 5deg; --tilt-from: 18deg; top: 40%; left: 22%; color: var(--amber); border: 4px double var(--amber); font-size: clamp(1rem, 2.1vw, 1.45rem); }
.pile.p3 { --tilt: -3deg; --tilt-from: -16deg; top: 70%; left: 12%; color: #5fa37a; border: 3px double #5fa37a; font-size: clamp(0.78rem, 1.4vw, 1rem); }
.pile { transform: rotate(var(--tilt)); }
html.js-reveal [data-reveal]:not(.in) .pile { opacity: 0; }
html.js-reveal [data-reveal].in .pile { animation: thump 280ms var(--ease-out) both; }
html.js-reveal [data-reveal].in .pile.p1 { animation-delay: 200ms; }
html.js-reveal [data-reveal].in .pile.p2 { animation-delay: 520ms; }
html.js-reveal [data-reveal].in .pile.p3 { animation-delay: 840ms; }

/* FAQ opens smoothly where the browser supports animating to auto height */
.faq details::details-content {
    height: 0;
    overflow: hidden;
    transition: height 320ms var(--ease-out), content-visibility 320ms allow-discrete;
}
.faq details[open]::details-content { height: auto; }
.faq summary { transition: color 150ms ease; }
@media (hover: hover) { .faq summary:hover { color: #fff; } }

/* Verdict stamp lands every time a result renders */
.verdict .stamp {
    --tilt: var(--stamp-tilt, -3deg);
    --tilt-from: 14deg;
    transform: rotate(var(--tilt));
    animation: thump 260ms var(--ease-out) both;
}

@media (max-width: 860px) {
    .card-grid > * { grid-column: auto !important; }
    .card-grid { grid-template-columns: 1fr; }
    .flag-card:nth-child(1) .flag-snippet, .flag-card:nth-child(4) .flag-snippet { font-size: 0.9rem; }
}

/* Reduced motion: keep the content and its colors, drop movement */
@media (prefers-reduced-motion: reduce) {
    .h-line, .hero-sub, .hero .hero-actions, .evidence-card, .stamp-chip, .verdict .stamp,
    html.js-reveal [data-reveal].in .pile, html.js-reveal [data-reveal].in .stamp-inline {
        animation: none !important; opacity: 1 !important;
    }
    html.js-reveal [data-reveal] > .stagger { transform: none !important; }
    .marquee { overflow-x: auto; }
    .marquee-track { animation: none; }
    .marquee-track[aria-hidden] { display: none; }
    .mock-field .caret { animation: none; }
    .spot:hover { transform: none; }
}
</style>
""", unsafe_allow_html=True)

page_html("""


<section class="hero" id="top">
    <div class="wrap hero-grid">
        <div>
            <h1 class="serif-xl"><span class="h-line">Is this job real?</span><span class="h-line">Find out before you reply.</span></h1>
            <p class="hero-sub">Paste a job posting or recruiter's message. We flag phrases that match known scams
            and tell you what to check.</p>
            <div class="hero-actions">
                <a class="btn btn-solid" href="#check">Check a posting</a>
                <a class="btn btn-line" href="#how-it-works">See how it works</a>
            </div>
        </div>
        <div class="evidence" aria-label="Example of a flagged phrase">
            <div class="evidence-card">
                <p class="evidence-label">From a flagged posting</p>
                <p class="evidence-text">…To secure your position, pay a $60 <mark class="flag">registration
                fee</mark> for your starter kit, which is refunded with your first paycheck…</p>
                <p class="evidence-tag">Upfront payment: real employers don't charge you to get hired.</p>
            </div>
            <div class="stamp-chip" aria-hidden="true">Flagged</div>
        </div>
    </div>
</section>

<section class="sources">
    <div class="wrap">
        <p class="sources-label">Built on public data and guidance from</p>
        <div class="sources-row">
            <span>EMSCAD dataset</span>
            <span>University of the Aegean</span>
            <span>FTC consumer alerts</span>
            <span>BBB Scam Tracker</span>
        </div>
    </div>
</section>

<section class="sec" id="red-flags">
    <div class="wrap">
        <div class="sec-head">
            <h2 class="serif-lg">Scammers repeat themselves. We know the lines.</h2>
            <p class="lede">Most job scams reuse a handful of moves. These four show up again and again.</p>
        </div>
        <div class="card-grid" data-reveal>
            <div class="flag-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 0">
                <p class="flag-snippet">pay a $60 <mark class="flag">registration fee</mark> for your starter kit</p>
                <div><h3 class="flag-title">Upfront payment</h3>
                <p class="flag-why">Real employers don't charge you to start working.</p></div>
            </div>
            <div class="flag-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 1">
                <p class="flag-snippet"><mark class="flag">deposit the check</mark>, keep your first week's pay and send the remainder</p>
                <div><h3 class="flag-title">The check that bounces</h3>
                <p class="flag-why">The check clears, then reverses. The money you sent on is gone.</p></div>
            </div>
            <div class="flag-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 2">
                <p class="flag-snippet">send your details to our hiring manager on <mark class="flag">Telegram</mark></p>
                <div><h3 class="flag-title">Off-platform chat</h3>
                <p class="flag-why">Moving you to a chat app skips the job site's fraud checks.</p></div>
            </div>
            <div class="flag-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 3">
                <p class="flag-snippet"><mark class="flag">No experience required</mark> and no interview needed</p>
                <div><h3 class="flag-title">Too easy, too fast</h3>
                <p class="flag-why">High pay with no interview is the lure, not the job.</p></div>
            </div>
        </div>
    </div>
</section>

<section class="statement">
    <div class="wrap">
        <p class="serif-lg"><span class="w" style="--i: 0">Reported</span> <span class="w" style="--i: 1">losses</span> <span class="w" style="--i: 2">to</span> <span class="w" style="--i: 3">job</span> <span class="w" style="--i: 4">scams</span> <span class="w" style="--i: 5"><em>tripled</em></span> <span class="w" style="--i: 6">from</span> <span class="w" style="--i: 7">2020</span> <span class="w" style="--i: 8">to</span> <span class="w" style="--i: 9">2023.</span></p>
        <p class="statement-source">Federal Trade Commission, Data Spotlight, December 2024</p>
        <p class="statement-link"><a class="btn btn-line btn-sm" href="./Then_vs_now" target="_self">See how our model holds up</a></p>
    </div>
</section>

<section class="wild" aria-label="Example scam messages">
<p class="wild-note">Examples written from common scam patterns</p>
<div class="marquee"><div class="marquee-track"><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Data entry, &#36;45/hr, <mark class="flag">no interview needed</mark>. Start Monday.</span></div><div class="msg"><span class="msg-src">School inbox</span><span class="msg-text">Professor here, I need an assistant. <mark class="flag">I&#8217;ll mail you a check</mark> for supplies.</span></div><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Let&#8217;s continue on <mark class="flag">Telegram</mark>, my recruiter account is limited.</span></div><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Remote package handler. <mark class="flag">Receive parcels at home</mark> and reship them.</span></div></div><div class="marquee-track" aria-hidden="true"><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Data entry, &#36;45/hr, <mark class="flag">no interview needed</mark>. Start Monday.</span></div><div class="msg"><span class="msg-src">School inbox</span><span class="msg-text">Professor here, I need an assistant. <mark class="flag">I&#8217;ll mail you a check</mark> for supplies.</span></div><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Let&#8217;s continue on <mark class="flag">Telegram</mark>, my recruiter account is limited.</span></div><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Remote package handler. <mark class="flag">Receive parcels at home</mark> and reship them.</span></div></div></div>
<div class="marquee reverse"><div class="marquee-track"><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Congrats, you&#8217;re hired! Send a photo of your <mark class="flag">driver&#8217;s license</mark> to start.</span></div><div class="msg"><span class="msg-src">School inbox</span><span class="msg-text">Campus job, &#36;400 a week. <mark class="flag">Reply with your bank details</mark>.</span></div><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Buy your laptop from our <mark class="flag">approved vendor</mark>, we&#8217;ll reimburse you.</span></div><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Task-based earning. <mark class="flag">Deposit 50 USDT</mark> to unlock withdrawals.</span></div></div><div class="marquee-track" aria-hidden="true"><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Congrats, you&#8217;re hired! Send a photo of your <mark class="flag">driver&#8217;s license</mark> to start.</span></div><div class="msg"><span class="msg-src">School inbox</span><span class="msg-text">Campus job, &#36;400 a week. <mark class="flag">Reply with your bank details</mark>.</span></div><div class="msg"><span class="msg-src">Job board</span><span class="msg-text">Buy your laptop from our <mark class="flag">approved vendor</mark>, we&#8217;ll reimburse you.</span></div><div class="msg"><span class="msg-src">DM</span><span class="msg-text">Task-based earning. <mark class="flag">Deposit 50 USDT</mark> to unlock withdrawals.</span></div></div></div>
</section>

<section class="band" id="how-it-works">
    <div class="wrap">
        <h2 class="serif-lg band-title">Hand it a posting.</h2>
        <div class="row" data-reveal>
            <div>
                <h3 class="row-title">Paste the whole thing</h3>
                <p class="row-copy">A job ad, an email, a text from someone who says they're a recruiter. The more
                text you give it, the better it can judge.</p>
                <ul class="checks"><li>Job board listings</li><li>Recruiter emails and DMs</li><li>Texts and chat messages</li></ul>
            </div>
            <div class="mock" aria-hidden="true">
                <div class="mock-chips"><span>Scam posting</span><span>Real posting</span><span>Recruiter DM</span></div>
                <div class="mock-field">Data Entry Clerk - Work From Home. Earn $4,500 weekly processing simple forms…<span class="caret" aria-hidden="true"></span></div>
            </div>
        </div>
        <div class="row" data-reveal>
            <div>
                <h3 class="row-title">See exactly what's flagged</h3>
                <p class="row-copy">A model trained on more than 14,000 real job postings scores it. Each suspicious
                phrase is underlined in place, with the reason next to it.</p>
                <ul class="checks"><li>A risk level: low, some, or high</li><li>Every red flag, underlined where it appears</li><li>A plain reason for each one</li></ul>
            </div>
            <div class="mock" aria-hidden="true">
                <div class="mock-verdict"><span class="stamp-inline">Flagged</span><p>High risk. Don't send money or personal details.</p></div>
                <div class="mock-field">…no interview needed, you can <mark class="flag">start today</mark>. Send your
                <mark class="flag">bank details</mark> to our hiring manager…</div>
            </div>
        </div>
        <div class="row" data-reveal>
            <div>
                <h3 class="row-title">Check it yourself before you reply</h3>
                <p class="row-copy">A low score isn't a guarantee. Use the result to decide what to check, then
                check it.</p>
                <ul class="checks"><li>Find the same listing on the company's own site</li><li>Match the recruiter's email to the company's domain</li><li>Ask your campus career center</li></ul>
            </div>
            <div class="mock" aria-hidden="true">
                <div class="mock-step" style="--i: 0"><span class="tick"></span>Listing found on the company's careers page</div>
                <div class="mock-step" style="--i: 1"><span class="tick"></span>Email domain matches the company</div>
                <div class="mock-step" style="--i: 2"><span class="tick"></span>Career center has heard of them</div>
            </div>
        </div>
    </div>
</section>

<section class="tool-head" id="check">
    <div class="wrap">
        <h2 class="serif-lg">Check a posting</h2>
        <p class="lede">Paste the full text, or try one of the examples.</p>
    </div>
</section>
""")

with st.container(key="tool"):
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.model.train` from the repo root, then reload.")
        st.stop()

    for col, name in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if col.button(name, use_container_width=True):
            st.session_state["text"] = EXAMPLES[name]

    text = st.text_area("Posting or message", key="text", height=220, placeholder="Paste the full text here…", max_chars=MAX_CHARS)
    st.caption("Nothing you paste is saved or logged by this app. Still, don't paste real SSNs, bank details or IDs.")

    if text.strip():
        score, model_risk = predict(text)
        risk = combined_risk(model_risk, text)
        flags = find_red_flags(text)
        render_verdict(risk)

        left, right = st.columns(2)
        left.metric("Model's fraud estimate", f"{score:.0%}")
        right.metric("Red flags found", len(flags))
        if len(text.split()) < MIN_WORDS:
            st.info(f"This is under {MIN_WORDS} words, which is too little text for the model to judge reliably. "
                    "Paste the full posting or the whole conversation if you can.")

        if flags:
            st.subheader("Red flags")
            st.markdown(f'<div style="line-height:1.7">{highlight_html(text, flags).replace("$", "&#36;")}</div>', unsafe_allow_html=True)
            st.write("")
            seen = set()
            for f in flags:
                if f["label"] not in seen:
                    seen.add(f["label"])
                    st.markdown(f"- **{f['label']}:** {f['why']}")

        model = load_model()
        st.caption(f"Model: tier {model.tier}, trained on EMSCAD job postings from 2012-2014. High risk starts at "
                   f"{model.threshold:.0%}, some risk at {model.review_threshold:.0%}. Modern scams may look different.")

page_html("""
<section class="sec">
    <div class="wrap">
        <div class="sec-head">
            <h2 class="serif-lg">Built for where scams find students.</h2>
        </div>
        <div class="where-grid" data-reveal>
            <div class="where-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 0">
                <span class="where-tag">Job boards</span>
                <div><p class="where-quote">Data Entry Clerk, &#36;4,500 weekly, <mark class="flag">no experience required</mark></p>
                <p>Fake listings borrow real company names and pay well above what the work is worth.</p></div>
            </div>
            <div class="where-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 1">
                <span class="where-tag">Your school inbox</span>
                <div><p class="where-quote">I need a student assistant. <mark class="flag">I will mail you a check</mark></p>
                <p>Emails to your .edu address from a professor or recruiter you've never met, offering easy money.</p></div>
            </div>
            <div class="where-card spot stagger" style="--glow: rgba(125,156,196,0.3); --i: 2">
                <span class="where-tag">Your DMs</span>
                <div><p class="where-quote">Text me on <mark class="flag">WhatsApp</mark>, not by email</p>
                <p>Recruiters who move you to Telegram or WhatsApp after one message and skip the interview.</p></div>
            </div>
        </div>
    </div>
</section>

<section class="sec" id="about" style="padding-top: 2rem">
    <div class="wrap about-grid">
        <div class="about-visual" data-reveal aria-hidden="true"><span class="pile p1">Flagged</span><span class="pile p2">Verify further</span><span class="pile p3">Looks clear</span></div>
        <div class="about-copy">
            <h2 class="serif-lg">A second opinion, not a verdict.</h2>
            <p>BogusAI is a student project from HackLanta 2026. It points out what looks wrong so you
            know what to check. It never decides for you whether an employer is real.</p>
            <p>The model learned from job postings collected between 2012 and 2014. Scams have changed since then,
            so we also test it against recent scam patterns written from public warnings. That's why every result
            shows its reasons instead of just a score.</p>
            <p>Nothing you paste is saved. It's scored in memory and discarded when you close the tab.</p>
        </div>
    </div>
</section>

<section class="sec" id="faq" style="padding-top: 2rem">
    <div class="wrap">
        <div class="sec-head"><h2 class="serif-lg">FAQ</h2></div>
        <div class="faq">
            <details>
                <summary>How accurate is it?</summary>
                <p>On job postings from 2012 to 2014 it hadn't seen, the model marked about 4 in 5 scams as high
                risk. On recent scam patterns it marks only about 1 in 10, and rates half of them low risk. The
                red flags are what catch most modern scams, so read each one rather than relying on the score. <a href="./Then_vs_now" target="_self">See the comparison</a>.</p>
            </details>
            <details>
                <summary>It said low risk. Am I safe?</summary>
                <p>Not necessarily. Low risk means nothing matched the patterns it knows. Still find the listing on
                the company's own site and confirm the recruiter's email domain before you share anything.</p>
            </details>
            <details>
                <summary>What does it look for?</summary>
                <p>Requests for money up front, checks you're asked to deposit and send back, moves to chat apps,
                early requests for bank or ID details, and pay that's too good for the work described.</p>
            </details>
            <details>
                <summary>Does it save what I paste?</summary>
                <p>No. Your text is scored in memory and isn't written anywhere.</p>
            </details>
            <details>
                <summary>I think I've been scammed. What now?</summary>
                <p>Stop sending money or information. Call your bank to flag any payments, then report it at
                <a href="https://reportfraud.ftc.gov" target="_blank" rel="noopener">ReportFraud.ftc.gov</a> and
                tell your campus career center.</p>
            </details>
        </div>
    </div>
</section>

<section class="cta">
    <div class="wrap">
        <h2 class="serif-xl" data-reveal>Got an offer that <mark class="flag">feels off</mark>?</h2>
        <div class="hero-actions"><a class="btn btn-solid" href="#check">Check a posting</a></div>
    </div>
</section>

<footer class="footer">
    <div class="wrap">
        <div class="footer-grid">
            <div>
                <div class="brand" translate="no">""" + brand_html() + """</div>
                <p class="footer-note">A warning to verify, never an accusation. Results are based on patterns,
                not on any knowledge of a specific employer.</p>
            </div>
            <div>
                <h4>Product</h4>
                <a href="#check">Check a posting</a>
                <a href="#how-it-works">How it works</a>
                <a href="#red-flags">Red flags</a>
            </div>
            <div>
                <h4>Help</h4>
                <a href="#faq">FAQ</a>
                <a href="https://reportfraud.ftc.gov" target="_blank" rel="noopener">Report a scam to the FTC</a>
            </div>
        </div>
        <p class="footer-base">HackLanta 2026</p>
    </div>
</footer>
""")
