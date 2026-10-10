# Real-world data

Why this exists: the model is trained on EMSCAD (2012-2014) and evaluated on a modern
challenge set that is synthetic, written by one author, and was used to write the
red-flag rules. Neither tells us how the app does on real scams people get today.

## What the gap is made of

Measured on tier B with EMSCAD thresholds:

| Test | Scams marked high risk | Scams rated low |
|---|---|---|
| EMSCAD test, full postings | 80% | 8% |
| EMSCAD's own scams cut to 17-word snippets | 17% | 48% |
| Modern challenge set (about 17 words each) | 10% | 51% |

Most of the drop is format: the model only ever saw whole postings. Training on short
windows of EMSCAD fixes part of that (snippet PR-AUC 0.21 to 0.32) but does not move the
challenge set, because modern scam mechanics (task/crypto deposits, gift cards,
reshipping) are barely present in 2012-2014 data. That part needs modern data.

## Sources

| Source | What it gives | License / terms | In git? |
|---|---|---|---|
| r/Scams via [Arctic Shift](https://arctic-shift.photon-reddit.com) (2024-01 onward) | Real job-scam messages people received, quoted in their posts, plus whole posts | Reddit user content; used for non-commercial research | IDs and labels only. Text cached in `data/external/reddit/` |
| [LinkedIn job postings, mid-2023](https://huggingface.co/datasets/xanderios/linkedin-job-postings) | 7,500 real modern legitimate postings (6,000 train, 1,500 eval, disjoint companies) | Uploader says MIT, but it is scraped LinkedIn content | Job IDs only (`data/realworld/linkedin_ids.csv`) |
| Employer career boards on Greenhouse and Lever (snapshot of 2026-10-10) | 413 postings open right now at 61 employers, for false alarms on current text. Evaluation only, never training | Public JSON feeds the boards publish for outside use; the text belongs to the employers | Posting IDs only (`data/realworld/live_ids.csv`). Text cached in `data/external/live/` |

Labels for Reddit come from Reddit, not from the rule author: "Scam report" / "Victim of
a scam" flairs are scams; "Is this a scam?" posts take the community verdict from the
highest-scored top-level comments (mixed threads dropped). See `src/realworld/reddit.py`.

LinkedIn postings are assumed legitimate. LinkedIn has some scams, so expect a little
label noise in the false-alarm numbers.

## Considered and not used

- **CFPB complaint narratives.** Public domain and full of real job-scam accounts, but the
  CFPB stopped publishing narratives on 2026-08-14; the bulk file no longer has the
  column and the search API blocks scripted clients. A FOIA request is the remaining route.
- **NextGig job postings (June 2026, CC BY 4.0).** Real postings, but every text field is
  an LLM-written summary. As legitimate examples they would teach the model that LLM prose
  means legitimate, so they are out.
- **Hugging Face "job scam" datasets.** The ones checked are EMSCAD rebalanced or generated text.
- **Unreleased:** a Reddit study of 175,430 scam messages ([arXiv 2605.16656](https://arxiv.org/abs/2605.16656))
  and a carrier study of 1.35M reports ([arXiv 2508.05276](https://arxiv.org/abs/2508.05276)). Worth emailing the authors.

## Splits

- Reddit: posts before 2025-07-01 train, later posts evaluate (scams from after the cutoff).
- LinkedIn: by company hash, 20% of companies held out.
- Eval also includes one 17-word snippet per held-out posting, to measure false alarms on
  short legitimate text.

## Commands

```bash
python -m src.realworld.reddit          # slow: the archive is rate-limited and often degraded; cached, re-run to resume
python -m src.realworld.dataset         # LinkedIn sample + committed index, prints counts
python -m src.model.train --tiers F FE  # augmented tiers (short windows + real modern data)
python -m src.model.realworld_eval      # model vs rules vs combined, per kind of text
python -m src.realworld.live            # live postings from employer boards (cached; --refresh for a new snapshot)
python -m src.model.live_eval           # false alarms on those, full postings and 17-word cuts
```
