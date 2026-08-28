"""
Profile Agent -> Input Processor.

First stage of the deck's Profile Agent pipeline. Normalises whatever the user
typed or said before anything tries to understand it.

The work here is unglamorous but load-bearing, because the input arrives from a
voice transcript as often as a keyboard. Indian financial conversation mixes
scripts and number systems freely -- "मेरी income 15 हज़ार है", "2.5 lakh",
"Rs. 1,20,000/-", "fifteen thousand rupees" -- and an extractor fed raw text
either misses those or hallucinates around them.

So this stage converts numbers to a canonical form and records what it found,
rather than leaving it to the language model to guess.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: Indian numbering multipliers, including common transliterations.
MULTIPLIERS: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "hazaar": 1_000,
    "hazar": 1_000,
    "हज़ार": 1_000,
    "हजार": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lakhs": 100_000,
    "लाख": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "cr": 10_000_000,
    "करोड़": 10_000_000,
}

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

CURRENCY_TOKENS = re.compile(r"(?:₹|rs\.?|inr|rupees?|/-)", re.IGNORECASE)

#: "15k", "2.5 lakh", "1.2 crore"
_AMOUNT_WITH_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(sorted(MULTIPLIERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
#: "1,20,000" or "120000"
_PLAIN_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d{4,}\b")


def _to_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def normalise_text(text: str) -> str:
    """Unicode-normalise, strip currency noise, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = CURRENCY_TOKENS.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_amounts(text: str) -> list[dict[str, Any]]:
    """
    Every monetary amount found, with the surface form that produced it.

    Keeping the surface form matters: it lets the response generator echo the
    user's own words back ("you said 15 thousand") instead of a bare integer,
    which is what makes a confirmation feel like being heard.
    """
    found: list[dict[str, Any]] = []
    consumed: list[tuple[int, int]] = []

    for match in _AMOUNT_WITH_UNIT.finditer(text):
        value = _to_number(match.group(1)) * MULTIPLIERS[match.group(2).lower()]
        found.append({
            "value": round(value, 2),
            "surface": match.group(0).strip(),
            "unit": match.group(2).lower(),
        })
        consumed.append(match.span())

    for match in _PLAIN_AMOUNT.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        found.append({
            "value": _to_number(match.group(0)),
            "surface": match.group(0),
            "unit": None,
        })

    return sorted(found, key=lambda a: a["value"], reverse=True)


def detect_language(text: str) -> str:
    """Rough script detection, enough to pick a reply language."""
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"          # Devanagari: Hindi or Marathi
    if re.search(r"[ಀ-೿]", text):
        return "kn"
    if re.search(r"[஀-௿]", text):
        return "ta"
    if re.search(r"[ঀ-৿]", text):
        return "bn"
    return "en"


def process_input(text: str) -> dict[str, Any]:
    """
    Clean the input and surface what can be read without a model.

    Anything found deterministically here is a fact the extractor does not have
    to infer, which is the cheapest way to reduce hallucination.
    """
    raw = text or ""
    cleaned = normalise_text(raw)
    amounts = extract_amounts(cleaned)

    return {
        "raw": raw,
        "cleaned": cleaned,
        "language": detect_language(raw),
        "amounts": amounts,
        "largest_amount": amounts[0]["value"] if amounts else None,
        "word_count": len(cleaned.split()),
        "is_empty": not cleaned,
        "mentions_currency": bool(CURRENCY_TOKENS.search(raw)),
    }
