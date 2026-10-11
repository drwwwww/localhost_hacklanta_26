"""Headless check of the Streamlit app against attack inputs. Needs a trained model.

    python -m src.model.train          # once
    python tests/app_smoke.py          # prints PASS/FAIL per check, exit code 1 on failure
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.model.predict import predict  # noqa: E402

APP = str(ROOT / "app" / "main.py")
BASE = ("Earn $900 weekly from home, no interview needed. Pay a $60 registration fee for your starter kit "
        "and send your bank details on Telegram to start today.")
EVADED = ("Earn $900 weekly from home, no interview needed. Pay a $60 registr\u0430tion fee for your starter kit "
          "and send your b\u200bank details on T e l e g r a m to start today.")
# A modern scam the EMSCAD-trained model rates below high risk; the upfront-payment rule catches it.
QUIET_SCAM = "Remote records clerk: pay a refundable $45 activation fee before the hiring manager releases your first assignment."
HOSTILE = '<script>alert(1)</script> Contact me on Telegram "><img src=x onerror=alert(1)>'


def run(text):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value(text).run()
    markdown = " ".join(m.value for m in at.markdown)
    return at, markdown, {m.label: m.value for m in at.metric}


failures = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failures.append(name)


at, md, base_metrics = run(BASE)
check("base scam text raises red flags", int(base_metrics["Red flags found"]) >= 4)

_, evaded_md, evaded_metrics = run(EVADED)
check("evasion text raises the same number of flags", evaded_metrics["Red flags found"] == base_metrics["Red flags found"])

_, quiet_md, _ = run(QUIET_SCAM)
check("model alone does not rate the quiet scam high", predict(QUIET_SCAM)[1] != "high")
check("red-flag rules still raise the quiet scam to a high-risk verdict",
      '<p class="verdict-headline">High risk.</p>' in quiet_md)

_, hostile_md, _ = run(HOSTILE)
check("script tag is escaped, not rendered", "<script" not in hostile_md and "&lt;script&gt;" in hostile_md)
check("img tag is escaped, not rendered", "<img src=x" not in hostile_md and "&lt;img src=x" in hostile_md)
check("text box has a length cap", at.text_area[0].max_chars == 20000)
check("privacy notice is shown", any("Nothing you paste is saved" in c.value for c in at.caption))
cfg = (ROOT / ".streamlit" / "config.toml").read_text()
check("usage telemetry is off", "gatherUsageStats = false" in cfg)

sys.exit(1 if failures else 0)
