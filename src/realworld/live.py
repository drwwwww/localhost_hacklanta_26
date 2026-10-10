"""Live legitimate postings: what real employers have open right now, from their own career boards.

    python -m src.realworld.live            # fetch (cached), sample, write the id list and the text cache
    python -m src.realworld.live --refresh  # drop the cached boards first, to take a new snapshot

Greenhouse and Lever host company career pages and publish each board as a public JSON
feed, no login, meant for outside use. These are the same postings those companies list
on LinkedIn and Indeed; LinkedIn and Indeed themselves are not scraped (their terms forbid it).

Used only to measure false alarms on current text (src/model/live_eval.py), never for
training. Postings are assumed legitimate: they come from the employer's own board.

Posting text belongs to the employers, so it is not committed: boards are cached under
data/external/live/ and only ids are kept in data/realworld/live_ids.csv. Boards change
daily, so a later fetch gives a different snapshot; the id list records which one was scored.
"""

import hashlib
import html
import json
import re
import shutil
import ssl
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external" / "live"
OUT_CSV = CACHE / "live_postings.csv"
IDS = ROOT / "data" / "realworld" / "live_ids.csv"
HEADERS = {"User-Agent": "hacklanta-fake-job-detector (student research, non-commercial)"}
PER_BOARD = 12  # so a 2,000-posting board counts no more than a 20-posting one
MIN_WORDS = 40

# Chosen for spread, not size: food and retail, health, finance, media, nonprofits, logistics,
# hardware and crowdwork next to software, since students apply well beyond tech.
BOARDS = {
    "greenhouse": [
        "sweetgreen", "hellofresh", "doordashusa", "toast", "peloton", "glossier", "faire",
        "onemedical", "oscar", "zocdoc", "flatironhealth",
        "sofi", "betterment", "chime", "affirm", "robinhood", "point72", "janestreet", "akunacapital", "justworks", "gusto",
        "aclu", "wikimedia", "khanacademy", "coursera", "axios",
        "flexport", "samsara", "lucidmotors", "chargepoint", "spacex", "relativity", "waymo",
        "airbnb", "lyft", "pinterest", "reddit", "discord", "duolingo", "roblox", "epicgames", "riotgames",
        "stripe", "gitlab", "cloudflare", "datadog", "mongodb", "okta", "anthropic", "scaleai", "figma", "asana",
    ],
    "lever": ["spotify", "palantir", "zoox", "wealthfront", "ro", "octoenergy", "matchgroup", "gopuff", "appen"],
}
URLS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true",
    "lever": "https://api.lever.co/v0/postings/{board}?mode=json",
}


def _ssl_context():
    try:  # python.org builds on macOS ship without root certificates
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch_board(provider, board):
    """Raw feed for one board, cached. A board that fails is skipped and retried on the next run."""
    path = CACHE / "boards" / f"{provider}_{board}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(URLS[provider].format(board=board), headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as r:
            data = json.load(r)
    except Exception as e:
        print(f"WARNING: {provider} {board} unavailable ({type(e).__name__}); re-run to retry")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched": date.today().isoformat(), "data": data}), encoding="utf-8")
    time.sleep(0.5)
    return json.loads(path.read_text(encoding="utf-8"))


def html_to_text(markup):
    """Greenhouse sends escaped HTML. Block tags become line breaks, the rest is dropped."""
    text = html.unescape(markup or "")
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<\s*(br|/p|/div|/li|/h[1-6]|/tr)\s*/?>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def _rows(provider, board, payload):
    fetched, data = payload["fetched"], payload["data"]
    if provider == "greenhouse":
        for job in data.get("jobs", []):
            yield dict(source_id=str(job["id"]), title=job.get("title", ""), url=job.get("absolute_url", ""),
                       location=(job.get("location") or {}).get("name", ""), fetched=fetched,
                       body=html_to_text(job.get("content")))
    else:
        for job in data:
            parts = [job.get("descriptionPlain", "")]
            parts += [f"{section.get('text', '')}\n{html_to_text(section.get('content'))}" for section in job.get("lists", [])]
            parts.append(job.get("additionalPlain", ""))
            yield dict(source_id=str(job["id"]), title=job.get("text", ""), url=job.get("hostedUrl", ""),
                       location=(job.get("categories") or {}).get("location", ""), fetched=fetched,
                       body="\n".join(p.strip() for p in parts if p and p.strip()))


def _rank(source_id):
    """Stable pseudo-random order, so the per-board sample does not depend on feed order."""
    return hashlib.md5(source_id.encode()).hexdigest()


def build():
    rows = []
    for provider, boards in BOARDS.items():
        for board in boards:
            payload = _fetch_board(provider, board)
            if payload is None:
                continue
            for row in _rows(provider, board, payload):
                rows.append({"source": provider, "board": board, **row})
    df = pd.DataFrame(rows)
    df["text"] = df["title"].str.strip() + "\n" + df["body"]
    ascii_share = df["body"].map(lambda t: sum(c.isascii() for c in t) / max(len(t), 1))
    df = df[(df["body"].str.split().str.len() >= MIN_WORDS) & (ascii_share > 0.95)]
    # Companies post one role per city; keep one copy.
    df = df[~df.duplicated(subset=["board", "title"]) & ~df["body"].str[:300].duplicated()]
    df = df.assign(rank=df["source_id"].map(_rank)).sort_values(["source", "board", "rank"])
    df = df.groupby(["source", "board"], sort=False).head(PER_BOARD).drop(columns=["rank", "body"])
    return df.reset_index(drop=True)


def load():
    """One row per sampled posting: source, board, source_id, title, url, location, fetched, text."""
    if not OUT_CSV.exists():
        main()
    return pd.read_csv(OUT_CSV, dtype={"source_id": str})


def main():
    if "--refresh" in sys.argv and (CACHE / "boards").exists():
        shutil.rmtree(CACHE / "boards")
    df = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    IDS.parent.mkdir(parents=True, exist_ok=True)
    df[["source", "board", "source_id", "url", "fetched"]].to_csv(IDS, index=False)
    words = df["text"].str.split().str.len()
    print(f"{len(df)} postings from {df['board'].nunique()} boards, fetched {df['fetched'].min()} to {df['fetched'].max()}; "
          f"median {int(words.median())} words")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
