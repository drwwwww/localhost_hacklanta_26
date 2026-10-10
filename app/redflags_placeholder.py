"""PLACEHOLDER red-flag rules so the UI can be built before C's module exists.

Swap the import in main.py for C's real rules. The contract the UI expects:

    find_red_flags(text) -> [{"start": int, "end": int, "label": str, "why": str}, ...]

where start/end are character offsets into `text`.
"""

import re

_RULES = [
    ("Upfront payment",
     r"(registration|training|application|processing|starter kit|equipment)\s+fee|pay\s+(a|an|the)?\s*\$?\d+.{0,20}(to (start|secure|begin)|upfront)",
     "Real employers don't charge you to get hired."),
    ("Check or money forwarding",
     r"(deposit|cash)\s+(the|a|this|it)?\s*check|send\s+(back\s+)?the\s+(remainder|rest|difference)|western union|moneygram|gift cards?|zelle|wire transfer",
     "Being sent a check and asked to forward part of it is the classic overpayment scam."),
    ("Off-platform chat",
     r"telegram|whatsapp|google (chat|hangouts)|signal app|text me (at|on)",
     "Moving the conversation to a chat app early avoids the job site's fraud checks."),
    ("Sensitive details too early",
     r"bank (details|account|information)|social security|\bssn\b|routing number|copy of your (id|passport|driver)",
     "Bank or ID details should only be requested after a formal offer, through HR systems."),
    ("Too easy, too fast",
     r"no (interview|experience)\s+(is\s+)?(required|needed|necessary)|hired immediately|start (today|immediately)|earn \$?\d[\d,]*\s*(\+)?\s*(a|per)?\s*(day|daily|week|weekly)",
     "High pay with no interview or experience is a common lure."),
]
_COMPILED = [(label, re.compile(pattern, re.I), why) for label, pattern, why in _RULES]


def find_red_flags(text):
    flags = []
    for label, pattern, why in _COMPILED:
        for m in pattern.finditer(text):
            flags.append({"start": m.start(), "end": m.end(), "label": label, "why": why})
    return sorted(flags, key=lambda f: f["start"])
