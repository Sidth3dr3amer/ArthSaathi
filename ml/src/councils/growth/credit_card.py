"""
Growth Council -> Credit Card Agent  (also Tier 2: Card Evaluation Engine).

Migrated verbatim from
`CreditCardDataMaker_Final/credit_card_recommendation_engine.ipynb`.

  analyze_spend_profile -> weights the user's spend into card-type affinity
  filter_eligible_cards -> hard eligibility gate (age, income, employment)
  calculate_card_value  -> net annual value: rewards earned minus fees
  build_spend_routing   -> which card to use for which category
  build_llm_context     -> compact JSON handed to the LLM for the final report

The card database is loaded from `CreditCardDataMaker_Final/final_decision/`,
resolved via `common.config` so it works from any working directory. During the
Day-1 audit this load was dead code -- a hardcoded literal shadowed it and the
cell raised `SyntaxError`. That cell was removed; this is now the only path.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...common import config
from ...schemas.profile import UserProfile
from ...schemas.state import FinancialState


def load_card_database(directory: Path | None = None) -> list[dict[str, Any]]:
    """
    Load every curated card JSON from `final_decision/`.

    Returns an empty list rather than raising when the directory is absent, so
    the agent degrades to "no cards available" instead of breaking a workflow.
    """
    directory = Path(directory or config.CARD_FINAL_DECISION_DIR)
    if not directory.exists():
        return []

    cards: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            cards.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return cards


def profile_to_engine_dict(profile: UserProfile) -> dict[str, Any]:
    """
    Adapt the unified `UserProfile` to the flat dict the engine expects.

    Field names here match the notebook's USER_PROFILE literal exactly.
    """
    return {
        "name": profile.name,
        "age": profile.age,
        "employment_type": (
            "salaried" if profile.job_type in ("salaried", "govt")
            else "self_employed" if profile.job_type in ("business", "freelancer")
            else profile.job_type
        ),
        "monthly_income": profile.monthly_income,
        "monthly_spend": dict(profile.monthly_spend),
        "max_annual_fee": profile.max_annual_fee,
        "prefer_cashback": profile.prefer_cashback,
        "prefer_travel_perks": profile.prefer_travel_perks,
        "travel_frequency": profile.travel_frequency,
        "preferred_airlines": [],
        "preferred_hotels": [],
        **profile.lifestyle_flags,
    }


def analyze_spend_profile(profile: dict) -> dict:
    """
    Compute spend ratios and determine priority card categories.
    Returns a dict with ratios, dominant category, and recommended card types.
    """
    spend = profile["monthly_spend"]
    total = sum(spend.values()) or 1

    # Group into macro-categories
    macro = {
        "cashback_general": (
            spend.get("online_shopping", 0) +
            spend.get("offline_retail", 0) +
            spend.get("groceries", 0)
        ),
        "travel": (
            spend.get("travel", 0) +
            spend.get("international", 0)
        ),
        "lifestyle": (
            spend.get("dining", 0) +
            spend.get("entertainment", 0)
        ),
        "utility": spend.get("utility_bills", 0),
        "fuel":    spend.get("fuel", 0),
        "others":  spend.get("others", 0),
    }

    ratios = {k: v / total for k, v in macro.items()}

    # Travel gets a 1.4× alpha boost (per spec) because travel benefits
    # are typically high-value (lounge access, miles, hotel status)
    TRAVEL_ALPHA = 1.4
    weighted = {
        "CASHBACK": ratios["cashback_general"] + ratios["utility"] * 0.6,
        "TRAVEL":   (ratios["travel"] + ratios["fuel"] * 0.3) * TRAVEL_ALPHA,
        "LIFESTYLE":ratios["lifestyle"] + ratios["utility"] * 0.3,
        "REWARDS":  ratios["others"] * 0.5,
    }

    # Sort by weighted importance
    ranked = sorted(weighted.items(), key=lambda x: -x[1])
    dominant = ranked[0][0]

    # User adjustments
    if profile.get("prefer_travel_perks") and profile.get("travel_frequency") == "frequent":
        dominant = "TRAVEL"
    if profile.get("prefer_cashback"):
        # nudge cashback up if user prefers it
        weighted["CASHBACK"] *= 1.15

    return {
        "macro_spend":   macro,
        "ratios":        ratios,
        "weighted":      weighted,
        "ranked_types":  ranked,
        "dominant_type": dominant,
        "total_monthly": total,
        "total_annual":  total * 12,
    }


def filter_eligible_cards(profile: dict, cards: list) -> tuple[list, list]:
    """
    Returns (eligible_cards, rejected_cards_with_reason).
    """
    eligible, rejected = [], []

    age  = profile["age"]
    emp  = profile["employment_type"]          # salaried | self_employed | student | unsalaried
    fee  = profile.get("max_annual_fee", 99999)

    for card in cards:
        reasons = []

        # Age check
        if card.get("age_min") is None or card.get("age_max") is None:
           continue  # skip cards with missing age criteria
        elif age < card.get("age_min", 0):
            reasons.append(f"Age {age} below minimum {card['age_min']}")
        elif age > card.get("age_max", 100):
            reasons.append(f"Age {age} above maximum {card['age_max']}")

        # Employment check
        if emp == "student"       and not card.get("student_eligible",      False):
            reasons.append("Card not available for students")
        if emp == "self_employed" and not card.get("self_employed_eligible", True):
            reasons.append("Card not available for self-employed")
        if emp == "unsalaried"    and not card.get("unsalaried_eligible",    False):
            reasons.append("Card not available for unsalaried individuals")

        # Invite-only
        if card.get("invite_only", False):
            reasons.append("Invite-only card")

        # Annual fee preference
        eff_fee = card.get("annual_fee", 0)
        if eff_fee > fee:
            reasons.append(f"Annual fee ₹{eff_fee} exceeds user max ₹{fee}")

        if reasons:
            rejected.append({"card": card["card_name"], "reasons": reasons})
        else:
            eligible.append(card)

    return eligible, rejected


def calculate_card_value(card: dict, profile: dict, spend_analysis: dict) -> dict:
    """
    Estimate annual monetary value a user gets from a card.
    Returns a breakdown dict with per-category savings and a total.
    """
    spend  = profile["monthly_spend"]
    is_airtel = profile.get("is_airtel_user", False)
    is_prime  = profile.get("is_amazon_prime", False)

    breakdown = {}

    # ── 1. Base cashback / rewards on all spend ──────────────────────
    base_rate = card.get("base_reward_rate", 1) / 100   # to decimal
    all_spend_annual = spend_analysis["total_annual"]
    breakdown["base_rewards"] = all_spend_annual * base_rate

    # ── 2. Category-specific rates ──────────────────────────────────
    # Utility
    util_rate = card.get("utility_reward_rate")
    if util_rate:
        util_annual = spend.get("utility_bills", 0) * 12
        if not is_airtel and "Airtel" in card["card_name"]:
            util_rate = 2   # non-Airtel users get reduced rate on Airtel card
        extra = util_annual * (util_rate - card.get("base_reward_rate", 1)) / 100
        breakdown["utility_bonus"] = max(extra, 0)

    # Travel
    travel_rate = card.get("travel_reward_rate")
    if travel_rate:
        travel_annual = spend.get("travel", 0) * 12
        extra = travel_annual * (travel_rate - card.get("base_reward_rate", 1)) / 100
        breakdown["travel_bonus"] = max(extra, 0)

    # ── 3. Named benefit valuation ───────────────────────────────────
    named_value = 0
    named_detail = []

    for b in card.get("benefits", []):
        bcat  = b.get("benefit_category", "")
        bval  = b.get("value", 0)
        bunit = b.get("value_unit", "")
        blim  = b.get("max_limit")
        bper  = b.get("limit_period", "")
        periods_per_year = 12 if "Monthly" in (bper or "") else 1

        # Skip non-percentage benefits from value calc (eVouchers handled separately)
        if bunit == "INR":
            named_value += bval        # one-time welcome vouchers etc.
            named_detail.append(f"{b['company_name']}: ₹{bval} voucher")
            continue

        if bunit != "%":
            continue

        # Map benefit to user spend
        company = b.get("company_name", "").lower()
        user_monthly_spend = 0

        if "zomato" in company or "swiggy" in company:
            user_monthly_spend = spend.get("dining", 0) * 0.5
        elif "blinkit" in company:
            user_monthly_spend = spend.get("groceries", 0) * 0.4
        elif "airtel" in company:
            user_monthly_spend = spend.get("utility_bills", 0) * (0.8 if is_airtel else 0)
        elif "utility" in company:
            user_monthly_spend = spend.get("utility_bills", 0) * (0.7 if is_airtel else 0)
        elif "amazon" in company:
            user_monthly_spend = spend.get("online_shopping", 0) * 0.5
        elif "goibibo" in company or "makemytrip" in company:
            user_monthly_spend = spend.get("travel", 0) * 0.4
        elif "fuel" in company:
            user_monthly_spend = spend.get("fuel", 0)
        elif "movie" in company or "district" in company:
            user_monthly_spend = spend.get("entertainment", 0) * 0.4
        elif "tira" in company:
            user_monthly_spend = spend.get("offline_retail", 0) * 0.1
        elif "online" in company:
            user_monthly_spend = spend.get("online_shopping", 0)
        elif "offline" in company:
            user_monthly_spend = spend.get("offline_retail", 0)
        elif "dining" in company:
            user_monthly_spend = spend.get("dining", 0)
        elif "travel" in company:
            user_monthly_spend = spend.get("travel", 0)

        if user_monthly_spend == 0:
            continue

        raw_monthly = user_monthly_spend * bval / 100
        if blim is not None:
            raw_monthly = min(raw_monthly, blim)

        annual = raw_monthly * periods_per_year
        named_value += annual
        named_detail.append(f"{b['company_name']} {bval}%: ₹{annual:,.0f}/yr")

    breakdown["named_benefits"] = named_value
    breakdown["named_detail"]   = named_detail

    # ── 4. Lounge access value ────────────────────────────────────────

    dom_lounges = card.get("domestic_lounge_visits") or 0
    intl_lounges = card.get("international_lounge_visits") or 0
    lounge_value = dom_lounges * 500 + intl_lounges * 2000   # approx market value
    breakdown["lounge_value"] = lounge_value

    # ── 5. Welcome bonus (one-time, amortise over 1 year) ────────────
    breakdown["welcome_bonus"] = card.get("welcome_bonus_value", 0)

    # ── 6. Forex savings ─────────────────────────────────────────────
    intl_spend_annual = spend.get("international", 0) * 12
    # Compare to avg 3.5% markup; card's markup is forex_markup
    avg_markup = 3.5
    card_markup = card.get("forex_markup", 3.5)
    forex_saving = intl_spend_annual * (avg_markup - card_markup) / 100
    breakdown["forex_savings"] = max(forex_saving, 0)

    # ── 7. Fee waiver check ──────────────────────────────────────────
    annual_fee = card.get("annual_fee", 0)
    waiver_threshold = card.get("fee_waiver_spend", None)
    if waiver_threshold and spend_analysis["total_annual"] >= waiver_threshold:
        breakdown["fee_waived"] = True
        effective_fee = 0
    else:
        breakdown["fee_waived"] = False
        effective_fee = annual_fee

    # ── Total ────────────────────────────────────────────────────────
    gross = (
        breakdown.get("base_rewards") or 0 +
        breakdown.get("utility_bonus") or 0 +
        breakdown.get("travel_bonus") or 0 +
        breakdown.get("named_benefits") or 0 +
        breakdown.get("lounge_value") or 0 +
        breakdown.get("welcome_bonus") or 0 +
        breakdown.get("forex_savings") or 0
    )

    # Dedup with base (named benefits may overlap)
    gross = gross * 0.75   # conservative overlap discount

    net = gross - effective_fee
    breakdown["gross_value"]   = gross
    breakdown["annual_fee"]    = annual_fee
    breakdown["effective_fee"] = effective_fee
    breakdown["net_value"]     = net

    return breakdown


def build_spend_routing(card_scores: list, profile: dict) -> dict:
    """
    For each spend category, recommend the best eligible card.
    Applies travel alpha boost (1.4×) when comparing cards for travel spends.
    """
    TRAVEL_ALPHA = 1.4

    spend_categories = [
        "online_shopping", "groceries", "dining", "fuel",
        "utility_bills", "travel", "entertainment", "international",
        "offline_retail", "others",
    ]

    # Build a quick lookup: card_name -> benefit rates
    routing = {}

    for cat in spend_categories:
        best_card  = None
        best_score = -999
        best_reason = ""

        for cs in card_scores:
            card = cs["card"]
            val  = cs["valuation"]
            score = 0
            reason_parts = []

            # Category-specific scoring
            if cat == "utility_bills":
                rate = card.get("utility_reward_rate", card.get("base_reward_rate", 1))
                score = rate
                reason_parts.append(f"{rate}% cashback")

            elif cat == "travel":
                rate = card.get("travel_reward_rate", card.get("base_reward_rate", 1))
                lounge_bonus = (card.get("domestic_lounge_visits", 0) * 500 +
                                card.get("international_lounge_visits", 0) * 2000) / 12
                score = (rate * TRAVEL_ALPHA) + lounge_bonus / 1000
                reason_parts.append(f"{rate}% rewards × {TRAVEL_ALPHA} alpha")
                if lounge_bonus > 0:
                    reason_parts.append(f"lounge access")

            elif cat == "international":
                rate = card.get("international_reward_rate", card.get("base_reward_rate", 1))
                forex_save = max(0, 3.5 - card.get("forex_markup", 3.5))
                score = rate + forex_save
                reason_parts.append(f"{rate}% rewards + {forex_save:.1f}% forex saving")

            elif cat == "fuel":
                rate = card.get("fuel_reward_rate", card.get("base_reward_rate", 1))
                score = rate
                reason_parts.append(f"{rate}% / fuel surcharge waiver")

            elif cat == "dining":
                rate = card.get("dining_reward_rate", card.get("base_reward_rate", 1))
                # Check named benefits for dining
                for b in card.get("benefits", []):
                    if b.get("value_unit") == "%" and ("zomato" in b.get("company_name","").lower() or "swiggy" in b.get("company_name","").lower()):
                        rate = max(rate, b.get("value", 0))
                score = rate
                reason_parts.append(f"up to {rate}% on dining/food apps")

            elif cat in ("online_shopping", "groceries", "entertainment", "offline_retail", "others"):
                rate = card.get("base_reward_rate", 1)
                # Check named online benefits
                for b in card.get("benefits", []):
                    bname = b.get("company_name","").lower()
                    if ("online" in bname or "amazon" in bname or "flipkart" in bname):
                        rate = max(rate, b.get("value", 0))
                score = rate
                reason_parts.append(f"{rate}% cashback/rewards")

            if score > best_score:
                best_score  = score
                best_card   = card["card_name"]
                best_reason = ", ".join(reason_parts) if reason_parts else f"{score:.1f}%"

        routing[cat] = {
            "card":   best_card,
            "score":  best_score,
            "reason": best_reason,
        }

    return routing


def build_llm_context(card_scores, routing, spend_analysis, profile, top_n=3) -> str:
    """Serialize the algorithmic outputs into a compact JSON payload for the LLM."""

    top_cards = []
    for cs in card_scores[:top_n]:
        card = cs["card"]
        v    = cs["valuation"]
        top_cards.append({
            "rank":          card_scores.index(cs) + 1,
            "card_name":     card["card_name"],
            "issuer":        card["issuer"],
            "card_type":     card["card_type"],
            "annual_fee":    card["annual_fee"],
            "fee_waived":    v.get("fee_waived", False),
            "effective_fee": v["effective_fee"],
            "gross_annual_value": round(v["gross_value"]),
            "net_annual_value":   round(v["net_value"]),
            "savings_breakdown": {
                "base_rewards":    round(v.get("base_rewards", 0)),
                "utility_bonus":   round(v.get("utility_bonus", 0)),
                "travel_bonus":    round(v.get("travel_bonus", 0)),
                "partner_benefits":round(v.get("named_benefits", 0)),
                "lounge_value":    round(v.get("lounge_value", 0)),
                "forex_savings":   round(v.get("forex_savings", 0)),
                "welcome_bonus":   round(v.get("welcome_bonus", 0)),
            },
            "partner_benefit_detail": v.get("named_detail", [])[:8],
            "best_for":    card.get("best_for", []),
            "avoid_for":   card.get("excluded_categories", [])[:5],
            "key_benefits": [
                {
                    "partner":    b["company_name"],
                    "rate":       f"{b['value']}{b['value_unit']}",
                    "type":       b["benefit_category"],
                    "cap":        f"max ₹{b['max_limit']}" if b.get("max_limit") else "uncapped",
                    "conditions": b.get("conditions", ""),
                }
                for b in card.get("benefits", [])
            ],
        })

    payload = {
        "user": {
            "name":             profile["name"],
            "age":              profile["age"],
            "employment":       profile["employment_type"],
            "monthly_spend":    profile["monthly_spend"],
            "total_monthly":    spend_analysis["total_monthly"],
            "total_annual":     spend_analysis["total_annual"],
            "is_airtel_user":   profile.get("is_airtel_user", False),
            "is_amazon_prime":  profile.get("is_amazon_prime", False),
            "travel_frequency": profile.get("travel_frequency", "occasional"),
            "max_annual_fee":   profile.get("max_annual_fee", 9999),
        },
        "spend_analysis": {
            "dominant_card_type": spend_analysis["dominant_type"],
            "ranked_types": [
                {"type": t, "weighted_score": round(s, 3)}
                for t, s in spend_analysis["ranked_types"]
            ],
        },
        "top_recommended_cards": top_cards,
        "spend_routing": {
            cat: {"card": r["card"], "reason": r["reason"]}
            for cat, r in routing.items()
            if profile["monthly_spend"].get(cat, 0) > 0
        },
    }

    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# LangGraph adapter (added during migration)
# --------------------------------------------------------------------------- #

def credit_card_node(
    state: FinancialState,
    *,
    cards: list[dict[str, Any]] | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    """
    Rank the card database against the user's spend profile.

    `cards` may be injected to bypass disk access; tests use that to stay
    hermetic and to exercise edge cases the four curated cards do not cover.
    """
    profile = state["profile"]
    engine_profile = profile_to_engine_dict(profile)
    database = cards if cards is not None else load_card_database()

    if not database:
        return {
            "credit_card_result": {
                "cards_considered": 0,
                "reason": "card database is empty",
                "recommendations": [],
                "routing": {},
            }
        }

    spend_analysis = analyze_spend_profile(engine_profile)
    eligible, rejected = filter_eligible_cards(engine_profile, database)

    # Shape matches the notebook's `card_scores` exactly -- build_spend_routing
    # and build_llm_context both index into "card" / "valuation" / "net_value".
    scored = []
    for card in eligible:
        valuation = calculate_card_value(card, engine_profile, spend_analysis)
        scored.append({
            "card": card,
            "valuation": valuation,
            "net_value": valuation["net_value"],
        })
    scored.sort(key=lambda x: -x["net_value"])

    routing = build_spend_routing(scored, engine_profile) if scored else {}

    return {
        "credit_card_result": {
            "cards_considered": len(database),
            "eligible_count": len(eligible),
            "rejected_count": len(rejected),
            "spend_analysis": spend_analysis,
            "recommendations": scored[:top_n],
            "routing": routing,
        }
    }
