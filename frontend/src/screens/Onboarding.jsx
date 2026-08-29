import { useEffect, useRef, useState } from "react";
import { api, inr } from "../lib/api.js";

/*
  Teaching Saathis — the onboarding conversation from the deck.

  Two things the server already decides, which this screen must not second-guess:

  1. WHICH question comes next. The Question Generator orders by what each
     answer unlocks, not by schema order, so the screen renders whatever it is
     handed rather than walking a fixed list.
  2. WHETHER an answer was accepted. A suspicious correction (a 100x jump in
     income) is held for confirmation, not written. That is surfaced here as a
     question rather than hidden.
*/

export default function Onboarding({ userId, onProfile }) {
  const [profile, setProfile] = useState({ user_id: userId, name: "" });
  const [plan, setPlan] = useState(null);
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    api.questions(userId).then(({ ok, data }) => ok && setPlan(data));
  }, [userId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [turns]);

  async function send() {
    const message = draft.trim();
    if (!message || busy) return;
    setDraft("");
    setTurns((t) => [...t, { who: "you", text: message }]);
    setBusy(true);
    setError(null);

    const { ok, data, error } = await api.turn(userId, message, profile);
    setBusy(false);
    if (!ok) {
      setError(error);
      return;
    }

    setProfile(data.profile);
    onProfile?.(data.profile);
    setPlan({
      next_question: data.next_question,
      completeness: data.completeness,
      queue: data.stages?.question_generator?.queue ?? [],
    });

    const learned = data.stages?.updater?.applied ?? [];
    const held = data.needs_confirmation ?? [];
    setTurns((t) => [
      ...t,
      { who: "artha", text: data.response, learned, held },
    ]);
  }

  const c = plan?.completeness;
  const answered = c?.answered ?? 0;
  const total = c?.total ?? 12;
  const pct = c?.percent ?? 0;

  return (
    <section className="section">
      <h2>Building your profile</h2>
      <p className="lede">
        Answer in whichever language you prefer. Each answer unlocks more of
        what the councils can tell you.
      </p>

      <div className="grid g2">
        <div className="panel">
          <h3>Conversation</h3>

          {turns.length === 0 && (
            <p className="empty">
              Start with anything — “I’m a farmer in Nashik and I earn about 35
              thousand a month” works.
            </p>
          )}

          {turns.map((t, i) => (
            <div className={`turn ${t.who}`} key={i}>
              <div className="who">{t.who === "you" ? "You" : "Artha"}</div>
              <div className="said">{t.text}</div>
              {t.learned?.length > 0 && (
                <div className="why" style={{ marginTop: 6, fontSize: 12.5, color: "var(--growth)" }}>
                  Recorded:{" "}
                  {t.learned
                    .map((l) =>
                      typeof l.to === "number" && String(l.field).includes("income")
                        ? `${l.field.replace(/_/g, " ")} ${inr(l.to)}`
                        : `${l.field.replace(/_/g, " ")} ${l.to}`
                    )
                    .join(", ")}
                </div>
              )}
              {t.held?.length > 0 && (
                <div className="why" style={{ marginTop: 6, fontSize: 12.5, color: "var(--bad)" }}>
                  Held for confirmation: {t.held.map((h) => h.field.replace(/_/g, " ")).join(", ")}
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />

          {error && <div className="err" style={{ marginTop: 12 }}>{error}</div>}

          <div className="ask-row" style={{ marginTop: 16 }}>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder={plan?.next_question?.question || "Tell me about your money…"}
              aria-label="Your answer"
            />
            <button className="go" onClick={send} disabled={busy}>
              {busy ? "…" : "Send"}
            </button>
          </div>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: 18 }}>
            <h3>Progress</h3>
            <div style={{ fontSize: 14, fontWeight: 500 }}>
              {answered} of {total} answered
            </div>
            <div className="progress">
              <i style={{ width: `${pct}%` }} />
            </div>
            <div className="muted" style={{ fontSize: 12.5 }}>
              {c?.can_advise
                ? "Enough to advise — the councils can run now."
                : "Income and essential expenses are needed before any council can advise."}
            </div>
          </div>

          {plan?.next_question && (
            <div className="panel" style={{ marginBottom: 18 }}>
              <h3>Next question</h3>
              <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 6 }}>
                {plan.next_question.question}
              </div>
              <div className="muted" style={{ fontSize: 13 }}>
                {plan.next_question.why_we_ask}
              </div>
              {plan.next_question.unlocks?.length > 0 && (
                <div style={{ marginTop: 10, display: "flex", gap: 5, flexWrap: "wrap" }}>
                  {plan.next_question.unlocks.map((u) => (
                    <span key={u} className="chip" style={{ fontSize: 11.5 }}>
                      unlocks {u}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {plan?.queue?.length > 0 && (
            <div className="panel">
              <h3>Still to ask</h3>
              <ul style={{ margin: 0, paddingLeft: 17, fontSize: 13.5 }}>
                {plan.queue.slice(1, 6).map((q) => (
                  <li key={q.field} style={{ marginBottom: 4 }}>
                    {q.question}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
