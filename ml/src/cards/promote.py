"""
Promote raw extracted card profiles into the engine's scoring shape.

`card_attributes/` holds 148 cards as the Tier-0 pipeline extracted them:
prose bullet lists, with reward rates buried in sentences like "1.5% cashback
on all other spends". `final_decision/` holds the shape the recommendation
engine actually scores: numeric rates, fees and eligibility flags.

The gap between them is real, and pretending otherwise would be worse than
leaving the four curated cards alone. Measured over all 148:

    121  cannot be scored: no line states a rate on ORDINARY spend, only
         category and merchant rates ("5% on fuel", "10% on Samsung")
     37  state the annual fee as prose rather than a number
      0  state employment eligibility (the `eligibility` field in the raw
         extraction actually contains card NETWORKS -- extraction noise)

So this module promotes only what it can derive with confidence -- 27 cards --
and records what it could not. A card missing a base reward rate is skipped
rather than defaulted, because a wrong rate does not fail loudly: it silently
mis-ranks every recommendation it touches. Two earlier versions of this parser
proved the point, promoting Axis Magnus at a 15% base earn rate and a further
twelve cards at 5% on all spend -- rates no Indian card pays, each of which
put those cards above every hand-checked one.

Employment eligibility is absent from the source entirely. Rather than invent
it, promoted cards are marked `eligibility_confirmed: False` and left
permissive, mirroring how the Benefits Council treats an unevaluable rule:
surfaced as unconfirmed, never guessed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..common import config

#: "1.5% cashback on all other spends" -- the base earn rate.
_BASE_RATE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%[^.;\n]{0,60}?\b(?:all other|every other|other spend|"
    r"all spend|all transaction|all retail|all purchase|all eligible|"
    r"unlimited cashback|other eligible|other domestic|every spend|across all)",
    re.IGNORECASE,
)
_ANY_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_FEE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+)", re.IGNORECASE)
_LOUNGE = re.compile(r"(\d+)\s+(?:complimentary\s+)?(?:domestic\s+)?lounge", re.IGNORECASE)

#: Plausibility bounds. A parsed number that falls outside these is not a
#: conservative estimate -- it is a misread, and it does not fail loudly. The
#: first version promoted Axis Magnus with a 15% base earn rate (the real rate
#: is nearer 1.2%), which would have made it dominate every ranking it appeared
#: in. Anything implausible is now a reason to skip the card, not to guess.
MIN_BASE_RATE, MAX_BASE_RATE = 0.2, 6.0
MAX_CATEGORY_RATE = 12.0
MIN_FEE_WAIVER = 10_000.0
MAX_ANNUAL_FEE = 100_000.0

#: The engine scores against these types; anything else is unusable to it.
CARD_TYPES = {"CASHBACK", "TRAVEL", "REWARDS", "FUEL", "SHOPPING", "LIFESTYLE"}

#: Spend category -> words that identify it in a reward sentence.
CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "utility": ("utility", "bill payment", "dth", "recharge", "electricity", "broadband"),
    "dining": ("dining", "restaurant", "swiggy", "zomato", "food"),
    "fuel": ("fuel", "petrol", "diesel"),
    "travel": ("travel", "flight", "hotel", "airline", "makemytrip", "booking"),
    "grocery": ("grocery", "supermarket", "bigbasket", "blinkit"),
    "online": ("online", "e-commerce", "amazon", "flipkart", "myntra"),
    "international": ("international", "forex", "overseas", "foreign currency"),
}


def _number(value: Any) -> float | None:
    """Coerce a fee that may be a number, 'Nil', or a sentence containing one."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    if re.search(r"\b(nil|free|zero|lifetime free|no annual fee)\b", value, re.I):
        return 0.0
    match = _FEE.search(value) or re.search(r"\b(\d[\d,]{2,})\b", value)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


#: Milestone and welcome benefits are bonuses by definition, so they may raise
#: a category rate but must never establish the BASE earn rate.
BASE_RATE_KEYS = ("reward_structure", "cashback_benefits")
CATEGORY_RATE_KEYS = BASE_RATE_KEYS + ("milestone_benefits",)


def _reward_lines(raw: dict[str, Any], keys: tuple[str, ...] = CATEGORY_RATE_KEYS) -> list[str]:
    lines: list[str] = []
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            lines.extend(str(v) for v in value)
        elif isinstance(value, str):
            lines.append(value)
    return lines


def _base_rate(lines: list[str]) -> float | None:
    """
    The earn rate on ordinary spend, which anchors every valuation.

    Only a line that states a rate on ALL spend counts. An earlier version fell
    back to the lowest rate mentioned anywhere, which quietly promoted category
    and merchant rates into the base: twelve cards claimed 5% on everything --
    a rate no Indian card pays -- purely because "5% cashback on Samsung
    purchases" was the only rate their profile stated. Every one of them then
    outranked the curated cards. No enumeration of categories can catch this,
    because the categories are brand names; the only sound test is whether the
    source says the rate applies to everything.

    Sources also contradict each other. HDFC Pixel carries both "1% CashBack on
    all transactions" and a marketing "5% cashback on all spends". A base rate
    is a floor by definition, so where several are stated, the lowest is the
    only defensible reading.
    """
    stated = []
    for line in lines:
        m = _BASE_RATE.search(line)
        if m:
            try:
                rate = float(m.group(1))
            except ValueError:
                continue
            # "100% fee waiver on all spends" is a waiver, not an earn rate.
            if MIN_BASE_RATE <= rate <= MAX_BASE_RATE:
                stated.append(rate)
    return min(stated) if stated else None


def _category_rates(lines: list[str], base: float) -> dict[str, float]:
    """Highest stated rate per spend category."""
    out: dict[str, float] = {}
    for line in lines:
        low = line.lower()
        rates = []
        for m in _ANY_RATE.finditer(line):
            try:
                r = float(m.group(1))
            except ValueError:
                continue
            if 0 < r <= MAX_CATEGORY_RATE:
                rates.append(r)
        if not rates:
            continue
        best = max(rates)
        for category, words in CATEGORY_WORDS.items():
            if any(w in low for w in words):
                out[category] = max(out.get(category, 0.0), best)
    return {k: v for k, v in out.items() if v > base}


def _lounge_visits(raw: dict[str, Any]) -> int:
    entries = raw.get("lounge_access") or []
    if isinstance(entries, str):
        entries = [entries]
    best = 0
    for entry in entries:
        m = _LOUNGE.search(str(entry))
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                pass
    return best


def promote_card(raw: dict[str, Any], issuer: str) -> dict[str, Any] | None:
    """
    Convert one raw record to the engine shape, or None if it cannot be scored.

    Returning None is the point: a card without a derivable earn rate cannot be
    valued, and inventing one would silently corrupt every ranking it appears in.
    """
    name = raw.get("card_name")
    if not name:
        return None

    lines = _reward_lines(raw)
    base = _base_rate(_reward_lines(raw, BASE_RATE_KEYS))
    if base is None or not (MIN_BASE_RATE <= base <= MAX_BASE_RATE):
        return None

    annual_fee = _number(raw.get("annual_fee"))
    if annual_fee is None or not (0 <= annual_fee <= MAX_ANNUAL_FEE):
        return None

    card_type = str(raw.get("card_type") or "REWARDS").upper().replace(" ", "_")
    if card_type not in CARD_TYPES:
        # "Credit Card" and similar carry no signal for the scorer; REWARDS is
        # the neutral default rather than a guess at the card's character.
        card_type = "REWARDS"

    card = {
        "card_name": str(name).strip(),
        "issuer": issuer,
        "network": raw.get("card_network"),
        "card_type": card_type,
        "annual_fee": annual_fee,
        "joining_fee": _number(raw.get("joining_fee")) or 0.0,
        "base_reward_rate": base,
        "domestic_lounge_visits": _lounge_visits(raw),
        "international_lounge_visits": 0,
        "invite_only": False,
        "age_min": 18,
        "age_max": 70,
        # Absent from the source. Left permissive and flagged rather than
        # invented -- see the module docstring.
        "self_employed_eligible": True,
        "unsalaried_eligible": True,
        "student_eligible": False,
        "income_requirement": 0,
        "eligibility_confirmed": False,
        "source": "promoted_from_card_attributes",
        "benefits": [],
    }

    for category, rate in _category_rates(lines, base).items():
        card[f"{category}_reward_rate"] = rate

    fee_waiver = _number(raw.get("fee_waiver_condition"))
    if fee_waiver and fee_waiver >= MIN_FEE_WAIVER:
        card["fee_waiver_spend"] = fee_waiver

    return card


def promote_all(
    source: Path | None = None, destination: Path | None = None
) -> dict[str, Any]:
    """
    Promote every raw card that can be scored. Returns a report.

    Cards already curated in `final_decision/` are never overwritten -- a
    hand-checked record beats a parsed one.
    """
    source = Path(source or config.CARD_ATTRIBUTES_DIR)
    destination = Path(destination or (config.CARD_PIPELINE_DIR / "card_pool"))
    destination.mkdir(parents=True, exist_ok=True)

    curated = {
        json.loads(p.read_text(encoding="utf-8")).get("card_name")
        for p in Path(config.CARD_FINAL_DECISION_DIR).rglob("*.json")
        if p.name != "cards_master.xlsx"
    }

    promoted, skipped = [], []
    for path in sorted(source.rglob("*.json")):
        issuer = path.parent.name
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped.append({"file": path.name, "reason": "unreadable"})
            continue

        if raw.get("card_name") in curated:
            skipped.append({"file": path.name, "reason": "already hand-curated"})
            continue

        card = promote_card(raw, issuer)
        if card is None:
            skipped.append({
                "file": path.name,
                "reason": "no plausible reward rate or annual fee",
            })
            continue

        out = destination / issuer
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{card['card_name']}.json").write_text(
            json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        promoted.append(card["card_name"])

    return {
        "source_files": len(list(source.rglob("*.json"))),
        "promoted": len(promoted),
        "skipped": len(skipped),
        "skip_reasons": {
            reason: sum(1 for s in skipped if s["reason"] == reason)
            for reason in {s["reason"] for s in skipped}
        },
        "destination": str(destination),
    }


if __name__ == "__main__":
    report = promote_all()
    print(json.dumps(report, indent=2))
