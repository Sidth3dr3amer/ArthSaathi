/*
  The only place the frontend knows the API exists.

  Every call returns { ok, data, error } rather than throwing, because a
  dashboard that shows a blank screen when the backend hiccups is worse than
  one that says what went wrong. The backend degrades rather than 500s, and
  this mirrors that contract on the client.
*/

const BASE = import.meta.env.VITE_API_BASE || "/api";

async function call(path, { method = "GET", body } = {}) {
  try {
    const res = await fetch(BASE + path, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      return {
        ok: false,
        data,
        error: data?.detail
          ? typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail)
          : `${res.status} ${res.statusText}`,
      };
    }
    return { ok: true, data, error: null };
  } catch (e) {
    return {
      ok: false,
      data: null,
      error: "Cannot reach the API. Start it with: uvicorn server.main:app --port 8000",
    };
  }
}

export const api = {
  health: () => call("/health"),
  workflows: () => call("/workflows"),

  chat: (user_id, message, profile, use_llm_router = false) =>
    call("/chat", { method: "POST", body: { user_id, message, profile, use_llm_router } }),

  ask: (user_id, question, profile) =>
    call("/ask", { method: "POST", body: { user_id, question, profile } }),

  profile: (user_id) => call(`/profile/${encodeURIComponent(user_id)}`),
  saveProfile: (user_id, profile) =>
    call(`/profile/${encodeURIComponent(user_id)}`, { method: "PUT", body: profile }),
  questions: (user_id) => call(`/profile/${encodeURIComponent(user_id)}/questions`),
  turn: (user_id, message, profile) =>
    call("/profile/turn", { method: "POST", body: { user_id, message, profile } }),

  memory: (user_id) => call(`/memory/${encodeURIComponent(user_id)}`),
  recall: (user_id, query) =>
    call(`/memory/${encodeURIComponent(user_id)}/recall`, { method: "POST", body: { query } }),

  cards: () => call("/cards"),
  recommendCards: (profile) => call("/cards/recommend", { method: "POST", body: { profile } }),

  voiceStatus: () => call("/voice/status"),
};

/* ------------------------------------------------------------------ money -- */

/**
 * Format rupees the way Indian readers actually read them: lakh and crore,
 * not millions. A dashboard that says "₹1,500,000" to someone who thinks in
 * lakh is making them do arithmetic to understand their own money.
 */
export function inr(value, { compact = false } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (compact) {
    const abs = Math.abs(n);
    if (abs >= 1e7) return `₹${(n / 1e7).toFixed(abs >= 1e8 ? 0 : 2)} cr`;
    if (abs >= 1e5) return `₹${(n / 1e5).toFixed(abs >= 1e6 ? 0 : 2)} L`;
  }
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

export const COUNCIL_COLOR = {
  risk: "var(--risk)",
  growth: "var(--growth)",
  cashflow: "var(--cashflow)",
  benefits: "var(--benefits)",
  behavioral: "var(--behavioral)",
};

export const COUNCIL_LABEL = {
  risk: "Risk",
  growth: "Growth",
  cashflow: "Cashflow",
  benefits: "Benefits",
  behavioral: "Behavioural",
};

/** Which council an agent belongs to — mirrors COUNCIL_AGENTS on the server. */
export const AGENT_COUNCIL = {
  emergency_fund: "risk", insurance: "risk", debt_trap: "risk", fraud: "risk",
  asset_allocation: "growth", credit_card: "growth", loan_advisor: "growth",
  retirement: "growth",
  scheme_matching: "benefits", eligibility: "benefits",
  bias_detection: "behavioral", habit_formation: "behavioral",
  nudge_strategy: "behavioral", literacy: "behavioral",
  stability: "cashflow", income_projection: "cashflow",
  expense_optimizer: "cashflow", goal_allocation: "cashflow",
};

export function titleCase(key = "") {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
