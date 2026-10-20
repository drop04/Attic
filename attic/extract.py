"""Document field extraction.

Tier 1 (always, fully local): regex + dateparser rules.
Tier 2 (optional, see cloud.py): an LLM on Bedrock, fed *redacted* text only.

Every extracted field carries a confidence and the source span it came from,
so the assistant can cite evidence and the UI can ask a human to confirm.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

import dateparser
from dateutil.relativedelta import relativedelta

DOC_TYPES = ("receipt", "warranty", "manual", "insurance", "prescription", "notice", "other")
FIELDS = (
    "doc_type", "vendor", "product", "model_no", "serial_no",
    "purchase_date", "amount", "currency", "warranty_months",
)
# Identifiers are only ever taken from the local pass: they are exactly what
# redaction removes before text is sent to the cloud, and an LLM may invent them.
LOCAL_ONLY = {"model_no", "serial_no"}

_DATE = (
    r"(\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})"
)
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
_NUM = r"(\d{1,3}|" + "|".join(_NUM_WORDS) + r")"


def parse_date(s: str) -> date | None:
    """Parse a date string. Day-first (DD/MM/YYYY), matching Indian receipts."""
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None
    dt = dateparser.parse(s, settings={"DATE_ORDER": "DMY", "PREFER_DAY_OF_MONTH": "first"})
    return dt.date() if dt else None


def _field(value: Any, conf: float, span: str) -> dict[str, Any]:
    return {"value": value, "confidence": conf, "span": span.strip()[:100]}


def _months(n: str, unit: str) -> int:
    count = int(n) if n.isdigit() else _NUM_WORDS[n.lower()]
    return count * 12 if unit.lower().startswith(("y",)) else count


def _detect_doc_type(low: str) -> str:
    rules = (
        ("insurance", ("policy no", "premium", "insured", "sum assured")),
        ("prescription", ("prescription", "dosage", "tablet", "rx ", "mg ")),
        ("manual", ("user manual", "owner's manual", "operating instructions", "user guide")),
        ("receipt", ("invoice", "receipt", "gstin", "grand total", "cash memo", "bill no")),
        ("warranty", ("warranty",)),
        ("notice", ("circular", "notice", "dear parent")),
    )
    for name, words in rules:
        if any(w in low for w in words):
            return name
    return "other"


def extract_local(text: str) -> dict[str, dict[str, Any]]:
    """Rule-based extraction. Returns {field: {value, confidence, span}}."""
    out: dict[str, dict[str, Any]] = {}
    low = text.lower()
    out["doc_type"] = _field(_detect_doc_type(low), 0.6, "")

    # purchase date: a labelled date beats any date
    m = re.search(
        r"(?:invoice|purchase|bill|order|sale|billing)?\s*date\s*(?:of\s*purchase)?\s*[:\-]?\s*" + _DATE,
        text, re.I,
    )
    if m and (d := parse_date(m.group(1))):
        out["purchase_date"] = _field(d.isoformat(), 0.9, m.group(0))
    else:
        for m in re.finditer(_DATE, text):
            if d := parse_date(m.group(1)):
                out["purchase_date"] = _field(d.isoformat(), 0.5, m.group(0))
                break

    # amount: prefer grand total, then total (never "subtotal")
    for pat, conf in (
        (r"grand\s*total", 0.9), (r"total\s*amount", 0.85), (r"amount\s*paid", 0.85),
        (r"net\s*(?:amount|payable)", 0.85), (r"(?<![A-Za-z])total", 0.7),
    ):
        m = re.search(
            pat + r"\s*(?:\(.*?\))?\s*[:\-]?\s*(₹|rs\.?|inr|\$|usd)?\s*([\d,]+(?:\.\d{1,2})?)",
            text, re.I,
        )
        if m:
            sym = (m.group(1) or "").lower()
            cur = {"₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "$": "USD", "usd": "USD"}.get(sym)
            out["amount"] = _field(float(m.group(2).replace(",", "")), conf, m.group(0))
            if cur:
                out["currency"] = _field(cur, conf, m.group(0))
            break

    # warranty: "Warranty: 1 year" (labelled) beats "1 year warranty"
    for pat in (
        r"warranty(?:\s*period)?\s*(?:of|:|-)?\s*" + _NUM + r"\s*[- ]?\s*(year|yr|month|mth)s?",
        _NUM + r"\s*[- ]?\s*(year|yr|month|mth)s?\s*(?:limited\s*|comprehensive\s*|standard\s*)?warranty",
    ):
        m = re.search(pat, text, re.I)
        if m:
            out["warranty_months"] = _field(_months(m.group(1), m.group(2)), 0.85, m.group(0))
            break

    # identifiers (must contain a digit, to skip "Model: Samsung")
    for key, pat in (
        ("model_no", r"\b(?:model|mod\.?)\s*(?:no\.?|number|#)?\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,24})"),
        ("serial_no", r"\b(?:serial|s/n|sr\.?\s*no\.?|imei)\s*(?:no\.?|number|#)?\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{5,24})"),
    ):
        for m in re.finditer(pat, text, re.I):
            if re.search(r"\d", m.group(1)):
                out[key] = _field(m.group(1).upper(), 0.85, m.group(0))
                break

    m = re.search(r"^\s*(?:product|item|description|article)\s*[:\-]\s*(.{3,80})$", text, re.I | re.M)
    if m:
        out["product"] = _field(m.group(1).strip(), 0.8, m.group(0))

    m = re.search(r"(?:sold\s*by|seller|store|dealer|vendor|merchant)\s*[:\-]\s*(.{3,60})", text, re.I)
    if m:
        out["vendor"] = _field(m.group(1).strip(), 0.85, m.group(0))
    else:
        for line in text.splitlines():
            s = line.strip()
            if 3 <= len(s) <= 60 and re.search(r"[A-Za-z]{3}", s) and not re.search(
                r"invoice|receipt|tax|bill|warranty|gstin|\d{4}", s, re.I
            ):
                out["vendor"] = _field(s, 0.5, s)
                break
    return out


# ---- merge, validate, finalise ------------------------------------------------

def merge(local: dict[str, dict], cloud: dict[str, dict] | None) -> dict[str, dict]:
    """Per field, keep the higher-confidence value. Identifiers stay local-only."""
    merged = dict(local)
    for key, cf in (cloud or {}).items():
        if key in LOCAL_ONLY:
            continue
        if key not in merged or cf["confidence"] > merged[key]["confidence"]:
            merged[key] = cf
    return merged


def compute_expiry(purchase_date: str | None, warranty_months: int | None) -> str | None:
    if not purchase_date or not warranty_months:
        return None
    return (date.fromisoformat(purchase_date) + relativedelta(months=int(warranty_months))).isoformat()


def clean_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce untrusted field values (from the UI, or from an LLM)."""
    out: dict[str, Any] = {}
    for k in ("title", "vendor", "product", "model_no", "serial_no", "currency"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = re.sub(r"\s+", " ", v).strip()[:120]
    dt = raw.get("doc_type")
    out["doc_type"] = dt if dt in DOC_TYPES else "other"
    for k in ("purchase_date", "expires_on"):
        v = raw.get(k)
        if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            try:
                out[k] = date.fromisoformat(v).isoformat()
            except ValueError:
                pass
    try:
        amt = float(raw["amount"])
        if 0 <= amt < 1e9:
            out["amount"] = amt
    except (KeyError, TypeError, ValueError):
        pass
    try:
        wm = int(raw["warranty_months"])
        if 0 < wm <= 240:
            out["warranty_months"] = wm
    except (KeyError, TypeError, ValueError):
        pass
    exp = compute_expiry(out.get("purchase_date"), out.get("warranty_months"))
    if exp:
        out["expires_on"] = exp
    return out


def finalize(merged: dict[str, dict]) -> tuple[dict[str, Any], float, dict[str, str]]:
    """-> (flat validated fields, overall confidence, source spans)."""
    flat = clean_fields({k: v["value"] for k, v in merged.items()})
    key_fields = [merged[k]["confidence"] for k in ("purchase_date", "warranty_months", "product", "vendor") if k in merged]
    confidence = round(sum(key_fields) / len(key_fields), 2) if key_fields else 0.0
    spans = {k: v["span"] for k, v in merged.items() if v.get("span") and k in flat}
    return flat, confidence, spans


# ---- redaction (applied before anything leaves the device) --------------------

_REDACTIONS = (
    (r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b", "[GSTIN]"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),
    (r"\b\d{4}\s\d{4}\s\d{4}\b", "[ID]"),
    (r"(?<![\w-])(?:\d[ -]?){13,19}(?!\w)", "[NUMBER]"),
    (r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)", "[PHONE]"),
    (r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]"),
)
_PII_LINE = re.compile(
    r"^(\s*(?:name|customer(?: name)?|bill(?:ed)? to|ship(?:ped)? to|deliver(?:y)? to|address|contact)\s*[:\-]).*$",
    re.I | re.M,
)


def redact(text: str) -> str:
    """Best-effort PII removal. Names/addresses are only caught when labelled."""
    text = _PII_LINE.sub(r"\1 [REDACTED]", text)
    for pat, repl in _REDACTIONS:
        text = re.sub(pat, repl, text)
    return text
