# ai/model/analyze_utils.py
import re
import json
from datetime import datetime
from typing import Any, Dict

# Heuristic patterns for redaction – not perfect, good for PoC
PATTERNS = [
    re.compile(r"arn:aws:[^\s\"']+"),
    re.compile(r"\b[0-9]{12}\b"),
    re.compile(r"\b[A-Z0-9]{16,40}\b")
]

def scrub(text: str) -> str:
    s = text
    for p in PATTERNS:
        s = p.sub("[REDACTED]", s)
    return s

def redact_metadata(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: redact_metadata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_metadata(v) for v in obj]
    if isinstance(obj, str):
        return scrub(obj)
    return obj

def save_report(report: Dict, prefix: str = "aws_report") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"{prefix}_{ts}.json"
    with open(name, "w") as f:
        json.dump(report, f, indent=2)
    return name
