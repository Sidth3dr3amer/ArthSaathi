import { useState } from "react";
import Chamber from "../components/Chamber.jsx";
import { api, inr, AGENT_COUNCIL, COUNCIL_COLOR, titleCase } from "../lib/api.js";
import { PRESETS } from "../lib/persona.js";

/*
  The report screen: ask a question, watch the councils convene, read the plan.

  Ordering is deliberate. The Chamber comes FIRST, before the allocation and
  the findings, because the argument is what justifies the numbers. Leading
  with a bar chart would reduce a five-council deliberation to decoration.
*/

function Allocation({ plan }) {
  if (!plan?.length) return <p className="empty">No surplus to allocate this month.</p>;
  const max = Math.max(...plan.map((p) => p.monthly_allocation), 1);
  return (
    <div className="alloc">
      {plan.map((p) => (
        <div className="alloc-row" key={p.claim}>
          <div className="top">
            <span className="name">{p.label}</span>
            <span className="amt num">{inr(p.monthly_allocation)}/mo</span>
          </div>
          <div className="alloc-bar">
            <i
              style={{
                width: `${(p.monthly_allocation / max) * 100}%`,
                background:
                  p.kind === "guaranteed_return" ? "var(--risk)"
                  : p.kind === "protective" ? "var(--cashflow)"
                  : p.kind === "long_term" ? "var(--growth)"
                  : "var(--benefits)",
              }}
            />
          </div>
          <div className="why">
            {Math.round(p.share_of_surplus * 100)}% of surplus · {p.rationale}
          </div>
        </div>
      ))}
    </div>
  );
}

function Findings({ recommendations }) {
  const agents = Object.keys(recommendations || {});
  if (!agents.length) return <p className="empty">No findings returned.</p>;
  return (
    <div>
      {agents.map((agent) => (
        <div className="finding" key={agent}>
          <div
            className="agent"
            style={{ color: COUNCIL_COLOR[AGENT_COUNCIL[agent]] || "var(--ink-3)" }}
          >
            {titleCase(agent)}
          </div>
          <ul>
            {recommendations[agent].slice(0, 4).map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default function Report({ profile }) {
  const [query, setQuery] = useState("give me a full financial review");
  const [asked, setAsked] = useState("");
  const [result, setResult] = useState(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState(null);

  async function ask(q) {
    const question = (q ?? query).trim();
    if (!question || working) return;
    setQuery(question);
    setAsked(question);
    setWorking(true);
    setError(null);
    setResult(null);

    const { ok, data, error } = await api.chat(profile.user_id, question, profile);
    setWorking(false);
    if (!ok) return setError(error);
    setResult(data);
    if (data.errors?.length) setError(data.errors.join(" · "));
  }

  return (
    <>
      <div className="ask">
        <div className="ask-row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask about your money — in English, हिंदी or मराठी"
            aria-label="Ask a question about your finances"
          />
          <button className="go" onClick={() => ask()} disabled={working}>
            {working ? "Convening…" : "Ask"}
          </button>
        </div>
        <div className="chips">
          {PRESETS.map((p) => (
            <button key={p.label} className="chip" onClick={() => ask(p.q)} disabled={working}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="err">{error}</div>}

      <Chamber result={result} working={working} query={asked} />

      {result && (
        <>
          <section className="section">
            <div className="grid g2">
              <div className="panel">
                <h3>Where your surplus goes</h3>
                <Allocation plan={result.allocation_plan} />
              </div>
              <div className="panel">
                <h3>What each agent found</h3>
                <Findings recommendations={result.recommendations} />
              </div>
            </div>
          </section>

          <section className="section">
            <dl className="kv">
              <dt>Intent</dt>
              <dd>{result.intent}</dd>
              <dt>Workflow</dt>
              <dd>{result.workflow}</dd>
              <dt>Agents run</dt>
              <dd>{result.agents_run?.length ?? 0} of 18</dd>
              <dt>Councils</dt>
              <dd>{result.council_verdicts?.length ?? 0} of 5</dd>
              <dt>Remembered</dt>
              <dd>{result.memory_written ? "yes" : "no"}</dd>
              <dt>Took</dt>
              <dd className="num">{result.elapsed_seconds}s</dd>
            </dl>
          </section>
        </>
      )}

      {!result && !working && (
        <p className="empty">
          Ask a question, or pick one above. A narrow question activates one
          agent; a full review convenes all five councils.
        </p>
      )}
    </>
  );
}
