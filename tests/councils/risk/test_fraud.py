"""
Risk Council -> Fraud Protection Agent.

Only the evidence layer touches the network. Everything asserted here is the
pure detection + scoring layer, plus the node driven with injected evidence, so
the whole file runs offline and deterministically.
"""

from __future__ import annotations

import json

import pytest

from ml.src.councils.risk.fraud import (
    MLM_TERMS,
    SCAM_TERMS,
    calculate_risk_score,
    detect_mlm_phrases,
    detect_scam_phrases,
    fraud_node,
    gather_evidence,
    normalize_company_name,
    score_evidence,
)
from ml.src.schemas.state import new_state

CLEAN_EVIDENCE = {
    "company_info": {"company_found": True},
    "domain_info": {"domain_age_days": 4_000},
    "rbi_check": {"rbi_warning_found": False},
    "sebi_check": {"enforcement_found": False, "sebi_registered": True},
    "complaints": {"complaint_count": 0},
    "scam_phrases": {"scam_term_count": 0},
    "mlm_phrases": {"mlm_count": 0},
    "web_rep": {"secure_connection": True},
}


def evidence(**overrides):
    """CLEAN_EVIDENCE with one or more sections replaced."""
    merged = {k: dict(v) for k, v in CLEAN_EVIDENCE.items()}
    for section, patch in overrides.items():
        merged[section] = {**merged.get(section, {}), **patch}
    return merged


# --------------------------------------------------------------------------- #
# Phrase detection (pure)
# --------------------------------------------------------------------------- #

def test_scam_phrases_detected_case_insensitively():
    out = detect_scam_phrases("RISK FREE and Double Your Money today")
    assert out["scam_term_count"] == 2
    assert set(out["matched_terms"]) == {"risk free", "double your money"}


def test_mlm_phrases_detected():
    out = detect_mlm_phrases("Build your downline and recruit members")
    assert out["mlm_count"] == 2


def test_clean_text_matches_nothing():
    text = "A regulated mutual fund with disclosed expense ratios."
    assert detect_scam_phrases(text)["scam_term_count"] == 0
    assert detect_mlm_phrases(text)["mlm_count"] == 0


def test_empty_text_is_safe():
    assert detect_scam_phrases("")["scam_term_count"] == 0
    assert detect_mlm_phrases("")["mlm_count"] == 0


def test_term_lists_are_populated_and_lowercase():
    assert len(SCAM_TERMS) > 10 and len(MLM_TERMS) > 10
    assert all(t == t.lower() for t in SCAM_TERMS)
    assert all(t == t.lower() for t in MLM_TERMS)


def test_company_name_normalisation_strips_suffixes():
    assert normalize_company_name("  Acme Pvt Ltd.  ").strip() != ""


# --------------------------------------------------------------------------- #
# Scoring rules — each weight asserted independently
# --------------------------------------------------------------------------- #

def test_fully_clean_entity_scores_zero():
    out = score_evidence(CLEAN_EVIDENCE)
    assert out["risk_score"] == 0
    assert out["triggered_rules"] == []
    assert "LOW" in out["risk_level"]


@pytest.mark.parametrize(
    "overrides,expected_delta",
    [
        ({"company_info": {"company_found": False}}, 40),
        ({"rbi_check": {"rbi_warning_found": True}}, 50),
        ({"sebi_check": {"enforcement_found": True}}, 30),
        ({"sebi_check": {"sebi_registered": False}}, 25),
        ({"domain_info": {"domain_age_days": 10}}, 30),
        ({"domain_info": {"domain_age_days": 100}}, 20),
        ({"complaints": {"complaint_count": 60}}, 40),
        ({"complaints": {"complaint_count": 20}}, 20),
        ({"scam_phrases": {"scam_term_count": 1}}, 15),
        ({"scam_phrases": {"scam_term_count": 6}}, 25),   # 15 + 10
        ({"mlm_phrases": {"mlm_count": 1}}, 20),
        ({"mlm_phrases": {"mlm_count": 4}}, 30),          # 20 + 10
        ({"web_rep": {"secure_connection": False}}, 10),
    ],
)
def test_each_rule_contributes_its_documented_weight(overrides, expected_delta):
    assert score_evidence(evidence(**overrides))["risk_score"] == expected_delta


def test_domain_age_bands_do_not_double_count():
    """A very new domain scores the 30 band, not 30 + 20."""
    assert score_evidence(evidence(domain_info={"domain_age_days": 10}))["risk_score"] == 30


def test_missing_domain_age_contributes_nothing():
    assert score_evidence(evidence(domain_info={"domain_age_days": None}))["risk_score"] == 0


def test_score_is_capped_at_100():
    worst = score_evidence(evidence(
        company_info={"company_found": False},
        rbi_check={"rbi_warning_found": True},
        sebi_check={"enforcement_found": True, "sebi_registered": False},
        domain_info={"domain_age_days": 5},
        complaints={"complaint_count": 500},
        scam_phrases={"scam_term_count": 20},
        mlm_phrases={"mlm_count": 20},
        web_rep={"secure_connection": False},
    ))
    assert worst["risk_score"] == 100
    assert "HIGH" in worst["risk_level"]


def test_missing_web_rep_does_not_raise():
    """`web_rep` was a notebook global; absent it must default, not crash."""
    payload = {k: v for k, v in CLEAN_EVIDENCE.items() if k != "web_rep"}
    assert score_evidence(payload)["risk_score"] == 0


def test_calculate_risk_score_contract():
    out = calculate_risk_score(
        company_info={}, domain_info={}, rbi_check={}, sebi_check={},
        complaints={}, scam_phrases={}, mlm_phrases={},
    )
    assert set(out) >= {"risk_score", "risk_level", "triggered_rules"}
    assert 0 <= out["risk_score"] <= 100


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_scores_injected_evidence_without_network(salaried_profile):
    state = new_state(salaried_profile, query="Doubler Capital guaranteed scheme")
    patch = fraud_node(state, evidence=evidence(
        company_info={"company_found": False},
        mlm_phrases={"mlm_count": 5},
    ))
    assert set(patch) == {"fraud_result"}
    result = patch["fraud_result"]
    assert result["checked"] is True
    assert result["risk_score"] == 70
    json.dumps(result)


def test_node_returns_unchecked_when_nothing_to_check(salaried_profile):
    patch = fraud_node(new_state(salaried_profile, query=""))
    result = patch["fraud_result"]
    assert result["checked"] is False
    assert result["risk_score"] == 0
    assert result["risk_level"] == "UNKNOWN"


def test_node_derives_company_name_from_the_query(salaried_profile):
    state = new_state(salaried_profile, query="Doubler Capital\nguaranteed returns")
    result = fraud_node(state, evidence=CLEAN_EVIDENCE)["fraud_result"]
    assert result["company_name"] == "Doubler Capital"


def test_node_explicit_company_name_wins(salaried_profile):
    state = new_state(salaried_profile, query="some scheme text")
    result = fraud_node(state, company_name="Acme Ltd", evidence=CLEAN_EVIDENCE)["fraud_result"]
    assert result["company_name"] == "Acme Ltd"


def test_gather_evidence_isolates_failures_per_probe(monkeypatch):
    """One failing source degrades that field; the rest still return."""
    import ml.src.councils.risk.fraud as mod

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "verify_company", boom)
    monkeypatch.setattr(mod, "check_rbi_alerts", lambda n: {"rbi_warning_found": True})
    monkeypatch.setattr(mod, "check_sebi", lambda n: {"sebi_registered": True})
    monkeypatch.setattr(mod, "search_complaints", lambda n: {"complaint_count": 0})

    out = gather_evidence("Acme", website_url=None, scheme_text="risk free")
    assert "error" in out["company_info"]
    assert out["rbi_check"]["rbi_warning_found"] is True
    assert out["scam_phrases"]["scam_term_count"] == 1
