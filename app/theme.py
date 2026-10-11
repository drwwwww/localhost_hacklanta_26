"""Shared look for every page: tokens, fonts, background pinning, nav, buttons and
Streamlit widget styling. Page-specific sections and motion stay in each page."""

import base64
import re
from pathlib import Path

import streamlit as st

BRAND = "BogusAI"
LOGO = Path(__file__).parent / "assets" / "BogusAI-2.svg"

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..500;1,6..72,300..500&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;700&display=swap');

:root {
    --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    interpolate-size: allow-keywords;
}


:root {
    --ink: #080b13;
    --panel: #0f1320;
    --card: #141a29;
    --line: rgba(236,232,223,0.12);
    --paper: #ece8df;
    --muted: #9aa1b2;
    --dim: #6f7688;
    --amber: #d0912f;
    --red: #c1443c;
    --steel: #7d9cc4;
    --serif: 'Newsreader', Georgia, serif;
    --sans: 'Geist', system-ui, sans-serif;
    --mono: 'Geist Mono', ui-monospace, monospace;
}

html {
    background: var(--ink);
    color-scheme: dark;
    scroll-behavior: smooth;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
html, body, [class*="css"] { font-family: var(--sans); }
/* The theme's fill lands on stApp, which paints above a negative-z-index iframe
   regardless of DOM nesting, so every ancestor of the background has to be clear. */
body, [data-testid="stApp"], [data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: none; padding: 0 !important; }
/* Streamlit adds hover-anchor icons to every markdown heading */
[data-testid="stHeaderActionElements"] { display: none; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0; }

h1, h2, h3 { text-wrap: balance; }
p, li { text-wrap: pretty; }

/* Pin the GhostFibers iframe (app/background.py) full-bleed behind everything */
[data-testid="stElementContainer"]:has(> iframe) {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -1 !important;
}
[data-testid="stElementContainer"]:has(> iframe) iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    pointer-events: none !important;
}

a, button, summary { touch-action: manipulation; }
.skip {
    position: fixed; top: 0.6rem; left: 0.6rem; z-index: 1000000;
    padding: 0.5rem 0.9rem; border-radius: 6px; background: var(--paper); color: var(--ink) !important;
    transform: translateY(-200%); transition: transform 0.15s ease;
}
.skip:focus-visible { transform: none; }
a:focus-visible, summary:focus-visible, button:focus-visible {
    outline: 2px solid var(--amber);
    outline-offset: 3px;
}

/* Opaque scrollbar track, otherwise the fixed fiber canvas shows through the gutter */
* { scrollbar-color: #2a3142 var(--ink); }

/* Sections overlap by 1px: fractional section heights left hairline seams where the
   fiber glow showed through between two solid blocks. */
.sources, .sec, .statement, .band, .tool-head, .st-key-tool, .cta, .footer { margin-top: -1px !important; }

/* Streamlit's markdown paragraph margins outrank class selectors */
.lede { margin: 1rem 0 0 !important; }
.row-copy { margin: 0.7rem 0 1.2rem !important; }
.about-copy p { margin: 1rem 0 0 !important; }
.flag-why { margin: 0.35rem 0 0 !important; }
.statement-source { margin: 1.2rem 0 0 !important; }

.wrap { max-width: 1120px; margin: 0 auto; padding: 0 2rem; }
.sec { background: var(--ink); padding: 7rem 0; scroll-margin-top: 5rem; }
.sec-head { text-align: center; max-width: 40rem; margin: 0 auto 3.5rem; }
/* Streamlit's markdown heading rules outrank plain class selectors, hence !important */
.serif-xl, .serif-lg, .flag-title, .row-title {
    font-family: var(--serif) !important;
    font-weight: 400 !important;
    letter-spacing: -0.015em;
    color: var(--paper) !important;
    margin: 0;
    padding: 0 !important;
}
.serif-xl { font-size: clamp(2.6rem, 5.4vw, 4.4rem) !important; line-height: 1.04 !important; }
.serif-lg { font-size: clamp(1.9rem, 3.4vw, 2.7rem) !important; line-height: 1.12 !important; }
.flag-title { font-size: 1.6rem !important; }
.row-title { font-size: 1.55rem !important; }
.lede { color: var(--muted); font-size: 1rem; line-height: 1.6; margin: 1rem 0 0; }

/* Nav: one glass bar, brand left, links centered, action right */
.nav {
    position: fixed;
    top: 0.9rem;
    left: 50%;
    transform: translateX(-50%);
    width: min(1120px, calc(100% - 1.5rem));
    z-index: 999999;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding: 0.6rem 0.6rem 0.6rem 1.8rem;
    border-radius: 999px;
    background: rgba(15,19,32,0.55);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid var(--line);
}
.brand { display: flex; align-items: center; gap: 0.55rem; color: var(--paper) !important; text-decoration: none !important; font-weight: 500; font-size: 0.9rem; }
/* Shared page frame for secondary pages (the home page lays out its own sections) */
.st-key-page { max-width: 880px; margin: 0 auto; padding: 7.5rem 2rem 5rem; background: var(--ink); }
.st-key-page h1 { font-family: var(--serif) !important; font-weight: 400 !important; letter-spacing: -0.015em; color: var(--paper) !important; font-size: clamp(2.2rem, 4.4vw, 3.4rem) !important; line-height: 1.08 !important; }
.st-key-page [data-testid="stMarkdownContainer"] p { color: #c4c9d4; line-height: 1.6; }
.st-key-page [data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: 12px !important; background: var(--panel); }
.brand-logo { height: 1.45rem; width: auto; display: block; }
/* The file's box includes room for the g's tail, so the capitals sit high; nudge them onto the bar's centre line. */
.nav .brand-logo { height: 2.15rem; transform: translateY(0.2rem); }
.nav-links { display: flex; gap: 2rem; }
.nav-links a, .nav-actions .ghost {
    color: var(--muted) !important;
    text-decoration: none !important;
    font-size: 0.94rem;
    transition: color 0.15s ease;
}
.nav-links a:hover { color: var(--paper) !important; }
.nav-actions { display: flex; justify-content: flex-end; }
.nav .btn-sm { min-height: 2.45rem; padding: 0 1.15rem; font-size: 0.9rem; }

.btn {
    display: inline-flex;
    align-items: center;
    min-height: 2.5rem;
    padding: 0 1.15rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 500;
    text-decoration: none !important;
    white-space: nowrap;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.btn-solid { background: var(--paper); color: var(--ink) !important; }
.btn-solid:hover { background: #fff; }
.btn-line { border: 1px solid rgba(236,232,223,0.32); color: var(--paper) !important; }
.btn-line:hover { border-color: var(--paper); }
.btn-sm { min-height: 2.2rem; padding: 0 1rem; font-size: 0.84rem; }

/* Streamlit widgets inside the tool */
[data-testid="stButton"] button {
    background: rgba(236,232,223,0.04) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px 6px 0 0 !important;
    color: #c7cbd6 !important;
    font-size: 0.84rem !important;
    min-height: 2.5rem;
    transition: border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stButton"] button:hover { border-color: var(--steel) !important; color: var(--paper) !important; }
[data-testid="stTextAreaRootElement"] { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 6px !important; }
[data-testid="stTextAreaRootElement"] textarea { color: var(--paper) !important; font-size: 0.96rem !important; }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 0.88rem !important; }
.st-key-tool [data-testid="stMarkdownContainer"] p, .st-key-tool [data-testid="stMarkdownContainer"] li { color: #c7cbd6; }
.st-key-tool [data-testid="stMarkdownContainer"] strong { color: var(--paper); }
.st-key-tool h3 { font-family: var(--serif) !important; font-weight: 400 !important; color: var(--paper) !important; font-size: 1.6rem !important; }
[data-testid="stCaptionContainer"] { color: var(--dim) !important; }

.verdict {
    display: flex; align-items: center; gap: 1.1rem;
    padding: 1rem 0.2rem; margin: 0.4rem 0 1.2rem;
    border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
}
.verdict .stamp {
    flex-shrink: 0; font-family: var(--mono); font-weight: 700; font-size: 0.9rem;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--stamp);
    border: 3px double var(--stamp); border-radius: 5px; padding: 0.45rem 0.8rem;
    white-space: nowrap; user-select: none;
}
.verdict-headline { color: var(--paper); font-weight: 500; font-size: 1.05rem; margin: 0; }
.verdict-advice { color: var(--muted); font-size: 0.92rem; margin: 0.15rem 0 0; }
[data-testid="stAlertContainer"] { border-radius: 6px !important; border: 1px solid rgba(125,156,196,0.35) !important; background: rgba(125,156,196,0.08) !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: var(--paper) !important; font-family: var(--serif) !important; font-variant-numeric: tabular-nums; }
mark.flag, .st-key-tool mark { background: transparent; color: inherit; border-bottom: 2px solid var(--amber); padding: 0; }

/* Browser surfaces carry the palette too */
::selection { background: rgba(208,145,47,0.35); color: var(--paper); }
textarea, input { caret-color: var(--amber); }
a { text-underline-offset: 0.2em; }

/* Buttons answer the press */
.btn { transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease, transform 160ms var(--ease-out); }
.btn:active, [data-testid="stButton"] button:active { transform: scale(0.97); }
[data-testid="stTextAreaRootElement"]:focus-within { border-color: rgba(208,145,47,0.6) !important; }


@media (max-width: 860px) {
    .nav { grid-template-columns: 1fr auto; }
    .nav-links { display: none; }
}
"""

# Section anchors live on the home page; "./" resolves to the app root from any page.
NAV_LINKS = [("How it works", "./#how-it-works"), ("Red flags", "./#red-flags"),
             ("Then vs now", "./Then_vs_now"), ("FAQ", "./#faq")]


def page_html(markup):
    # Streamlit's markdown parser turns indented lines into code blocks and $...$ into math.
    # Curly apostrophes too: page copy never uses single-quoted HTML attributes.
    markup = re.sub(r"\n\s*", "\n", markup).replace("$", "&#36;").replace("'", "’")
    st.markdown(markup, unsafe_allow_html=True)


def apply_theme():
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)


def brand_html():
    """The wordmark as an inline image, so it needs no static file serving."""
    data = base64.b64encode(LOGO.read_bytes()).decode()
    return f'<img class="brand-logo" src="data:image/svg+xml;base64,{data}" alt="{BRAND}">'


def render_nav(skip_target="#check"):
    links = "".join(f'<a href="{href}" target="_self">{label}</a>' for label, href in NAV_LINKS)
    page_html(f"""
    <a class="skip" href="{skip_target}">Skip to main content</a>
    <nav class="nav" aria-label="Main">
        <a class="brand" href="./" target="_self" translate="no">{brand_html()}</a>
        <div class="nav-links">{links}</div>
        <div class="nav-actions"><a class="btn btn-solid btn-sm" href="./#check" target="_self">Check a posting</a></div>
    </nav>
    """)
