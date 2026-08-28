"""
Risk Council -> Fraud Protection Agent.

Migrated verbatim from `FraudDetectionAdvisor/fraud_detection_mvp.ipynb`.

Three clearly separated layers:

  1. EVIDENCE (network)   analyze_domain, web_search, verify_company,
                          check_rbi_alerts, check_sebi, search_complaints,
                          search_news, website_reputation
  2. DETECTION (pure)     detect_scam_phrases, detect_mlm_phrases
  3. SCORING (pure)       calculate_risk_score -- deterministic rule weights,
                          documented in its own docstring, capped at 100

Only layer 1 touches the network, so layers 2 and 3 are unit-tested directly and
`fraud_node` is tested with injected evidence. Search degrades Tavily -> SerpAPI
-> DuckDuckGo depending on which keys are present.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests
import whois
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from ...common import config
from ...schemas.state import FinancialState

# Search-provider selection, resolved from config rather than notebook globals.
TAVILY_API_KEY = config.TAVILY_API_KEY
SERPAPI_KEY = config.SERPAPI_KEY
GROQ_API_KEY = config.GROQ_API_KEY
USE_TAVILY = bool(TAVILY_API_KEY)
USE_SERPAPI = bool(SERPAPI_KEY) and not USE_TAVILY
USE_GROQ = bool(GROQ_API_KEY)


SCAM_TERMS = [
    "guaranteed returns",
    "risk free",
    "risk-free",
    "double your money",
    "fixed income",
    "assured profit",
    "assured returns",
    "100% return",
    "earn daily",
    "daily income",
    "no risk",
    "zero risk",
    "safe investment",
    "100% safe",
    "get rich",
    "passive income guarantee",
    "instant profit",
    "quick money",
    "easy money",
    "limited slots",
    "limited time offer",
    "exclusive offer",
    "secret formula",
    "insider tips",
    "jackpot",
    "monthly assured",
    "weekly profit",
    "fixed monthly",
    "no loss",
]


MLM_TERMS = [
    "downline",
    "binary income",
    "team income",
    "referral income",
    "recruit members",
    "network marketing",
    "level income",
    "join our team",
    "direct selling",
    "matrix plan",
    "generation income",
    "sponsor",
    "recruitment bonus",
    "joining fee",
    "activation fee",
    "distributor",
    "upline",
    "pyramid",
    "chain marketing",
    "multi level",
    "multilevel",
    "downline bonus",
    "team bonus",
    "refer and earn",
]


COMPANY_REGISTRIES = ["mca.gov.in", "zaubacorp.com", "tofler.in", "quickcompany.in"]


def normalize_company_name(text):
    text = text.strip()

    if text.startswith(("http://", "https://")):
        domain = urlparse(text).netloc.lower()
        domain = domain.replace("www.", "")
        return domain.split(".")[0]

    return text


def detect_scam_phrases(text: str) -> dict:
    """Count and identify scam phrases in combined_text."""
    text_lower = text.lower()
    matched = [term for term in SCAM_TERMS if term in text_lower]
    return {
        "scam_term_count": len(matched),
        "matched_terms":   matched,
    }


def detect_mlm_phrases(text: str) -> dict:
    """Count and identify MLM/pyramid scheme phrases."""
    text_lower = text.lower()
    matched = [term for term in MLM_TERMS if term in text_lower]
    return {
        "mlm_count":    len(matched),
        "matched_terms": matched,
    }


def calculate_risk_score(
    company_info: dict,
    domain_info:  dict,
    rbi_check:    dict,
    sebi_check:   dict,
    complaints:   dict,
    scam_phrases: dict,
    mlm_phrases:  dict,
    web_rep:      dict | None = None,
) -> dict:
    """
    Deterministic risk scoring — no LLM involved.

    Rule weights
    ─────────────────────────────────────────────────────────────────────────
    Company not found in MCA / registries     +40
    RBI warning / caution notice found        +50
    SEBI enforcement action found             +30
    SEBI not registered (no positive hits)    +25
    Domain age < 180 days                     +20
    Domain age < 30  days                     +10 (additional)
    Complaint count > 10                      +20
    Complaint count > 50                      +40 (replaces +20)
    Scam phrase detected (any)                +15
    Scam phrase > 5 matches                   +10 (additional)
    MLM phrase detected (any)                 +20
    MLM phrase > 3 matches                    +10 (additional)
    SEBI enforcement found                    already counted above
    Website not HTTPS                         +10
    ─────────────────────────────────────────────────────────────────────────
    Score capped at 100.
    """
    score = 0
    rules = []

    # Company not found
    if not company_info.get("company_found"):
        score += 40
        rules.append(("Company not found in MCA/corporate registries", 40))

    # RBI warning
    if rbi_check.get("rbi_warning_found"):
        score += 50
        rules.append(("RBI warning / caution notice found", 50))

    # SEBI
    if sebi_check.get("enforcement_found"):
        score += 30
        rules.append(("SEBI enforcement action found", 30))
    if not sebi_check.get("sebi_registered"):
        score += 25
        rules.append(("Not found as SEBI-registered entity", 25))

    # Domain age
    age = domain_info.get("domain_age_days")
    if age is not None:
        if age < 30:
            score += 30
            rules.append((f"Domain extremely new ({age} days < 30 days)", 30))
        elif age < 180:
            score += 20
            rules.append((f"Domain relatively new ({age} days < 180 days)", 20))

    # Complaints
    cc = complaints.get("complaint_count", 0)
    if cc > 50:
        score += 40
        rules.append((f"High complaint volume: {cc} results found", 40))
    elif cc > 10:
        score += 20
        rules.append((f"Moderate complaint volume: {cc} results found", 20))

    # Scam phrases
    stc = scam_phrases.get("scam_term_count", 0)
    if stc > 0:
        score += 15
        rules.append((f"Scam phrases detected: {scam_phrases.get('matched_terms', [])}", 15))
    if stc > 5:
        score += 10
        rules.append((f"High scam phrase density ({stc} matches)", 10))

    # MLM phrases
    mlmc = mlm_phrases.get("mlm_count", 0)
    if mlmc > 0:
        score += 20
        rules.append((f"MLM/pyramid phrases detected: {mlm_phrases.get('matched_terms', [])}", 20))
    if mlmc > 3:
        score += 10
        rules.append((f"High MLM phrase density ({mlmc} matches)", 10))

    # Website not HTTPS
    if not (web_rep or {}).get("secure_connection", True):
        score += 10
        rules.append(("Website does not use HTTPS", 10))

    # Cap
    score = min(score, 100)

    # Risk level
    if score <= 30:
        level = "🟢 LOW"
    elif score <= 60:
        level = "🟡 MEDIUM"
    else:
        level = "🔴 HIGH"

    return {
        "risk_score":     score,
        "risk_level":     level,
        "triggered_rules": rules,
    }


def analyze_domain(url: str) -> dict:

    result = {
        "domain": None,
        "registrar": None,
        "creation_date": None,
        "expiration_date": None,
        "domain_age_days": None,
        "country": None,
        "name_servers": [],
        "status": None,
        "error": None,
    }

    try:

        # Extract domain
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)

        domain = parsed.netloc.lower().replace("www.", "")

        result["domain"] = domain

        # WHOIS lookup
        w = whois.whois(domain)

        def parse_date(value):
            if value is None:
                return None

            if isinstance(value, list):
                value = value[0]

            if isinstance(value, datetime):
                return value.replace(tzinfo=None)

            return None

        creation = parse_date(getattr(w, "creation_date", None))
        expiration = parse_date(getattr(w, "expiration_date", None))

        result["registrar"] = getattr(w, "registrar", None)

        result["country"] = getattr(w, "country", None)

        result["status"] = getattr(w, "status", None)

        name_servers = getattr(w, "name_servers", None)

        if name_servers:

            if isinstance(name_servers, set):
                name_servers = list(name_servers)

            if isinstance(name_servers, str):
                name_servers = [name_servers]

            result["name_servers"] = sorted(
                list(set(str(ns) for ns in name_servers))
            )

        if creation:

            result["creation_date"] = creation.isoformat()

            result["domain_age_days"] = (
                datetime.utcnow() - creation
            ).days

        if expiration:
            result["expiration_date"] = expiration.isoformat()

        # Fraud signal
        if result["domain_age_days"] is not None:

            age = result["domain_age_days"]

            if age < 30:
                result["risk_signal"] = "VERY_NEW_DOMAIN"

            elif age < 180:
                result["risk_signal"] = "NEW_DOMAIN"

            elif age < 365:
                result["risk_signal"] = "RECENT_DOMAIN"

            else:
                result["risk_signal"] = "ESTABLISHED_DOMAIN"

    except Exception as e:

        result["error"] = str(e)

    return result


def web_search(query: str, num: int = 5) -> list[dict]:
    """
    Returns list of {title, url, snippet}.
    Tries Tavily → SerpAPI → DuckDuckGo HTML scrape.
    """
    results = []

    # 1) Tavily
    if os.environ.get("TAVILY_API_KEY"):
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": os.environ["TAVILY_API_KEY"],
                      "query": query,
                      "max_results": num,
                      "search_depth": "basic"},
                timeout=15,
            )
            if resp.ok:
                for r in resp.json().get("results", []):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "snippet": r.get("content", "")[:300],
                    })
                if results:
                    return results
        except Exception:
            pass

    # 2) SerpAPI
    if os.environ.get("SERPAPI_KEY"):
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": os.environ["SERPAPI_KEY"],
                        "num": num, "hl": "en", "gl": "in"},
                timeout=15,
            )
            if resp.ok:
                for r in resp.json().get("organic_results", []):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("link", ""),
                        "snippet": r.get("snippet", "")[:300],
                    })
                if results:
                    return results
        except Exception:
            pass

    # 3) DuckDuckGo HTML (no API key needed, rate-limited)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (FraudCheck/1.0)"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=15,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select(".result__title a")[:num]:
            href = a.get("href", "")
            snippet_tag = a.find_parent(".result").find(class_="result__snippet") if a.find_parent(".result") else None
            results.append({
                "title":   a.get_text(strip=True),
                "url":     href,
                "snippet": snippet_tag.get_text(strip=True)[:300] if snippet_tag else "",
            })
        if results:
            return results
    except Exception:
        pass

    return []  # all backends failed


def verify_company(company_name: str) -> dict:
    """Search for company on Indian corporate registries."""
    results_all = []
    for site in COMPANY_REGISTRIES:
        query = f'site:{site} "{company_name}"'
        results_all.extend(web_search(query, num=3))
    # Also generic search
    results_all.extend(web_search(f'"{company_name}" MCA registered company India', num=5))

    company_found   = len(results_all) > 0
    evidence_links  = list({r["url"] for r in results_all if r["url"]})[:10]
    evidence_snips  = [r["snippet"] for r in results_all if r["snippet"]][:5]

    return {
        "company_found":      company_found,
        "evidence_links":     evidence_links,
        "evidence_snippets":  evidence_snips,
    }


def check_rbi_alerts(company_name: str) -> dict:
    """Search for RBI warnings, caution notices, and enforcement actions."""
    queries = [
        f'"{company_name}" RBI warning',
        f'"{company_name}" RBI caution notice',
        f'"{company_name}" RBI alert unauthorized',
        f'"{company_name}" Reserve Bank of India fraud',
        f'site:rbi.org.in "{company_name}"',
    ]
    results_all = []
    for q in queries:
        results_all.extend(web_search(q, num=3))

    rbi_keywords = ["rbi", "reserve bank", "warning", "caution", "notice",
                    "unauthorized", "alert", "penalty", "enforcement"]

    matching = [
        r for r in results_all
        if any(kw in (r["title"] + r["snippet"]).lower() for kw in rbi_keywords)
    ]

    return {
        "rbi_warning_found": len(matching) > 0,
        "supporting_links":  list({r["url"] for r in matching})[:10],
        "warning_snippets":  [r["snippet"] for r in matching if r["snippet"]][:3],
    }


def check_sebi(company_name: str) -> dict:
    """Check SEBI registration and any SEBI enforcement actions."""
    queries = [
        f'site:sebi.gov.in "{company_name}"',
        f'"{company_name}" SEBI registered investment advisor',
        f'"{company_name}" SEBI registration number',
        f'"{company_name}" SEBI order penalty',
        f'"{company_name}" SEBI fraud unregistered',
    ]
    results_all = []
    for q in queries:
        results_all.extend(web_search(q, num=3))

    positive_kw = ["registered", "registration", "sebi.gov.in", "ria", "portfolio manager"]
    negative_kw = ["unregistered", "illegal", "fraud", "penalty", "order", "violation",
                   "ban", "debarred", "warned"]

    positive = [r for r in results_all if any(kw in (r["title"] + r["snippet"]).lower() for kw in positive_kw)]
    negative = [r for r in results_all if any(kw in (r["title"] + r["snippet"]).lower() for kw in negative_kw)]

    sebi_registered  = len(positive) > 0 and len(negative) == 0
    enforcement_found = len(negative) > 0

    return {
        "sebi_registered":     sebi_registered,
        "enforcement_found":   enforcement_found,
        "supporting_links":    list({r["url"] for r in results_all})[:10],
        "positive_snippets":   [r["snippet"] for r in positive][:2],
        "negative_snippets":   [r["snippet"] for r in negative][:2],
    }


def search_complaints(company_name: str) -> dict:

    company_name = normalize_company_name(company_name)

    queries = [
        f'{company_name} complaint',
        f'{company_name} scam',
        f'{company_name} fraud',
        f'{company_name} review',
        f'site:consumercomplaints.in {company_name}',
        f'site:mouthshut.com {company_name}',
        f'site:reddit.com {company_name} scam',
        f'{company_name} cheated money',
    ]

    results_all = []

    for q in tqdm(queries, desc="Complaint queries"):

        try:
            results = web_search(q, num=10)

            print(f"\nQuery: {q}")
            print(f"Results: {len(results)}")

            results_all.extend(results)

            time.sleep(0.5)

        except Exception as e:
            print(f"Search failed: {q}")
            print(e)

    # Deduplicate
    unique = {}
    for r in results_all:

        url = r.get("url")

        if url:
            unique[url] = r

    unique_results = list(unique.values())

    return {
        "company_name": company_name,
        "complaint_count": len(unique_results),
        "complaint_results": unique_results[:20]
    }


def search_news(company_name: str) -> dict:
    """Search for investigative news articles about the company."""
    queries = [
        f'"{company_name}" investigation ED CBI police',
        f'"{company_name}" fraud arrested India',
        f'"{company_name}" Ponzi scheme India',
        f'"{company_name}" SFIO investigation',
        f'"{company_name}" scam news',
    ]
    results_all = []
    for q in tqdm(queries, desc="News queries", leave=False):
        results_all.extend(web_search(q, num=4))
        time.sleep(0.3)

    news_domains = [
        "economictimes", "ndtv", "thehindu", "hindustantimes",
        "livemint", "businessstandard", "moneycontrol", "timesofindia",
        "scroll.in", "thewire", "newslaundry",
    ]

    articles = []
    seen = set()
    for r in results_all:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        domain = urlparse(r["url"]).netloc
        is_news = any(nd in domain for nd in news_domains)
        articles.append({
            "title":    r["title"],
            "source":   domain,
            "url":      r["url"],
            "snippet":  r["snippet"],
            "is_news":  is_news,
        })

    articles.sort(key=lambda x: x["is_news"], reverse=True)

    return {
        "article_count": len(articles),
        "articles":      articles[:15],
    }


def website_reputation(url: str, domain_info: dict | None = None) -> dict:
    """
    Checks:
      - HTTPS presence
      - Reachability
      - Domain age from earlier WHOIS result
      - Basic header signals
    """
    result = {
        "secure_connection":   False,
        "reachable":           False,
        "status_code":         None,
        "server":              None,
        "domain_age_days":     (domain_info or {}).get("domain_age_days"),
        "suspicious":          False,
        "flags":               [],
    }

    try:
        full_url = url if url.startswith("http") else "https://" + url
        result["secure_connection"] = full_url.startswith("https://")

        if not result["secure_connection"]:
            result["flags"].append("No HTTPS — connection is unencrypted")

        resp = requests.get(full_url, timeout=10, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        result["reachable"]   = True
        result["status_code"] = resp.status_code
        result["server"]      = resp.headers.get("Server", "Unknown")

        age = result["domain_age_days"]
        if age is not None and age < 180:
            result["flags"].append(f"Domain is only {age} days old (< 180 days)")

    except requests.exceptions.SSLError:
        result["flags"].append("SSL certificate error")
    except requests.exceptions.ConnectionError:
        result["flags"].append("Website unreachable")
    except Exception as exc:
        result["flags"].append(f"Check error: {exc}")

    result["suspicious"] = len(result["flags"]) > 0
    return result


def build_prompt(evidence: dict) -> str:
    q    = evidence["query"]
    risk = evidence["risk"]
    rules_text = "\n".join(
        f"  • [{pts:+d} pts] {rule}"
        for rule, pts in risk["triggered_rules"]
    ) or "  None triggered."

    return f"""
You are a senior financial fraud investigator specialising in Indian retail investment scams.
You have been given a structured evidence package about a company / investment scheme.
Your task is to produce a comprehensive Fraud Analysis Report in Markdown.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Company Name      : {q['company_name']}
Scheme Name       : {q['scheme_name']}
Website           : {q['website_url']}
Scheme Description:
{q['scheme_description']}

WhatsApp Message  :
{q['whatsapp_message']}

SMS Message       :
{q['sms_message']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETERMINISTIC RISK SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score      : {risk['risk_score']} / 100
Risk Level : {risk['risk_level']}

Triggered Rules:
{rules_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Company found in registries : {evidence['company_info']['company_found']}
RBI warning found            : {evidence['rbi_check']['rbi_warning_found']}
SEBI registered              : {evidence['sebi_check']['sebi_registered']}
SEBI enforcement action      : {evidence['sebi_check']['enforcement_found']}
Domain age (days)            : {evidence['domain_info']['domain_age_days']}
Domain registrar             : {evidence['domain_info']['registrar']}
Website HTTPS                : {evidence['website_rep']['secure_connection']}
Complaint results found      : {evidence['complaints']['complaint_count']}
News articles found          : {evidence['news']['article_count']}
Scam phrases matched         : {evidence['scam_phrases']['matched_terms']}
MLM phrases matched          : {evidence['mlm_phrases']['matched_terms']}

Top complaint URLs:
{chr(10).join('  - ' + r['url'] for r in evidence['complaints']['top_results'][:3]) or '  None'}

Top news articles:
{chr(10).join('  - [' + a['source'] + '] ' + a['title'] for a in evidence['news']['top_articles'][:3]) or '  None'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR REPORT MUST INCLUDE (in order, all in Markdown):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Executive Summary** — 2-3 sentence scam likelihood verdict, written for a non-finance person.
2. **Why We Think This** — Explain the reasoning behind the risk score in plain language.
3. **Red Flags Identified** — Bullet list of every red flag found in the evidence AND in the scheme's language/promises. Be specific.
4. **Positive Signs (if any)** — Bullet list of any factors that reduce risk (e.g. genuine SEBI registration, long-standing domain, clean news history). Write "None found." if none.
5. **Detailed Analysis** — Sections covering: Company Legitimacy | Regulatory Standing (RBI + SEBI) | Digital Footprint | Complaint Pattern | Language Analysis (scam / MLM phrases).
6. **What This Means For You** — Written at Class-10 reading level in simple, direct language. Use an analogy if helpful.
7. **What To Do Now** — Numbered action steps: (a) immediate steps to protect yourself, (b) how to verify, (c) how to report.
8. **Verification Checklist** — A checklist (using - [ ] markdown) of steps to independently verify this entity.
9. **India-Specific Helplines & Portals** — List with names, numbers, and URLs for: SEBI SCORES, RBI Sachet, MCA21, Cyber Crime portal, NCLT, SFIO, local consumer forum.
10. **Disclaimer** — Short standard disclaimer that this is AI-assisted analysis and not legal advice.

Write clearly, compassionately, and without jargon. This report may be read by a first-time investor who is scared of losing their savings.
"""


# --------------------------------------------------------------------------- #
# Orchestration + LangGraph adapter (added during migration)
# --------------------------------------------------------------------------- #

def gather_evidence(
    company_name: str,
    website_url: str | None = None,
    scheme_text: str = "",
) -> dict[str, Any]:
    """
    Run the whole evidence layer. This is the only part that touches the network.

    Every probe is individually guarded: one failing source (WHOIS timeout, a
    rate-limited search) degrades that field rather than failing the check.
    """
    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return {"error": repr(exc)}

    domain_info = _safe(analyze_domain, website_url) if website_url else {}
    return {
        "company_info": _safe(verify_company, company_name),
        "domain_info": domain_info,
        "rbi_check": _safe(check_rbi_alerts, company_name),
        "sebi_check": _safe(check_sebi, company_name),
        "complaints": _safe(search_complaints, company_name),
        "web_rep": _safe(website_reputation, website_url, domain_info) if website_url else {},
        "scam_phrases": detect_scam_phrases(scheme_text),
        "mlm_phrases": detect_mlm_phrases(scheme_text),
    }


def score_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Pure scoring over gathered evidence. No network, no LLM."""
    return calculate_risk_score(
        company_info=evidence.get("company_info") or {},
        domain_info=evidence.get("domain_info") or {},
        rbi_check=evidence.get("rbi_check") or {},
        sebi_check=evidence.get("sebi_check") or {},
        complaints=evidence.get("complaints") or {},
        scam_phrases=evidence.get("scam_phrases") or {},
        mlm_phrases=evidence.get("mlm_phrases") or {},
        web_rep=evidence.get("web_rep") or {},
    )


#: Phrases that signal the user is actually asking us to vet something.
#: Without this guard the agent treated any query as a company name -- a routine
#: "give me a full financial review" triggered eight live complaint searches
#: taking 24 seconds and returning noise. Network probes are expensive and
#: rate-limited, so the bar to fire them is an explicit signal.
FRAUD_INTENT_PATTERNS = re.compile(
    r"\b(scam|fraud|fraudulent|genuine|legit|legitimate|trust(worthy)?|safe to invest|"
    r"is this real|too good to be true|ponzi|chit fund|mlm|multi.?level|"
    r"guaranteed returns?|double (my|your) money|should i invest in|"
    r"verify|check (this )?(company|scheme|offer)|received (a|an) (offer|message|call))\b",
    re.IGNORECASE,
)


def looks_like_a_fraud_check(text: str) -> bool:
    """Does this query actually ask us to vet an entity?"""
    return bool(text and FRAUD_INTENT_PATTERNS.search(text))


def fraud_node(
    state: FinancialState,
    *,
    company_name: str | None = None,
    website_url: str | None = None,
    scheme_text: str | None = None,
    evidence: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Assess whether an offer put to the user is fraudulent.

    Only runs the network evidence layer when there is an entity to check --
    an explicit `company_name`/`website_url`, injected `evidence`, `force=True`,
    or a query that reads like a fraud check. Otherwise it returns `checked:
    False` immediately and costs nothing.

    `evidence` may be injected to skip the network layer entirely -- that is how
    the tests exercise scoring deterministically.
    """
    text = scheme_text if scheme_text is not None else state.get("query", "")

    def _unchecked(reason: str) -> dict[str, Any]:
        return {
            "fraud_result": {
                "checked": False,
                "reason": reason,
                "risk_score": 0,
                "risk_level": "UNKNOWN",
                "triggered_rules": [],
            }
        }

    if evidence is None:
        has_explicit_target = bool(company_name or website_url)
        if not (has_explicit_target or force or looks_like_a_fraud_check(text)):
            return _unchecked(
                "no entity to verify -- supply a company name, or ask a question "
                "that names something to check"
            )

        name = company_name or (text.strip().split("\n")[0][:120] if text else "")
        if not name:
            return _unchecked("no company name or scheme text supplied")

        evidence = gather_evidence(name, website_url, text)
    else:
        name = company_name or (text.strip().split("\n")[0][:120] if text else "")

    scored = score_evidence(evidence)
    return {
        "fraud_result": {
            "checked": True,
            "company_name": name,
            "website_url": website_url,
            **scored,
            "evidence": evidence,
        }
    }
