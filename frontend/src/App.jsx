import { useEffect, useState } from "react";
import Report from "./screens/Report.jsx";
import Onboarding from "./screens/Onboarding.jsx";
import Schemes from "./screens/Schemes.jsx";
import { api, inr } from "./lib/api.js";
import { RAHUL } from "./lib/persona.js";

const TABS = [
  ["report", "Report"],
  ["schemes", "Schemes"],
  ["onboarding", "Onboarding"],
  ["voice", "Voice"],
];

function Identity({ profile }) {
  const surplus =
    (profile.monthly_income || 0) -
    (profile.essential_expenses || 0) -
    (profile.debts || []).reduce(
      (t, d) => t + Math.max(d.minimum_due || 0, d.emi || 0),
      0
    );

  const facts = [
    profile.occupation && profile.occupation[0].toUpperCase() + profile.occupation.slice(1),
    profile.state,
    profile.dependents ? `${profile.dependents} dependants` : null,
    profile.has_health_insurance ? null : "no health cover",
  ].filter(Boolean);

  return (
    <div className="identity">
      <div>
        <h1>{profile.name || "New user"}</h1>
        <div className="meta">{facts.join(" · ")}</div>
      </div>
      <div className="figure">
        <div className="k">Monthly income</div>
        <div className="v num">{inr(profile.monthly_income)}</div>
      </div>
      <div className="figure" style={{ marginLeft: 0 }}>
        <div className="k">Surplus</div>
        <div className="v num" style={{ color: surplus < 0 ? "var(--bad)" : "var(--growth)" }}>
          {inr(surplus)}
        </div>
      </div>
    </div>
  );
}

function Voice() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    api.voiceStatus().then(({ ok, data }) => ok && setStatus(data));
  }, []);

  return (
    <section className="section">
      <h2>Voice assistant</h2>
      <p className="lede">
        Speak in Hindi, Marathi, Kannada or English. Speech in, speech back —
        for users who would rather talk than type.
      </p>
      <div className="panel">
        <h3>Service</h3>
        {status ? (
          <dl className="kv">
            <dt>Available</dt>
            <dd>{status.available ? "yes" : "not loaded"}</dd>
            <dt>Model</dt>
            <dd>{status.model || "faster-whisper small"}</dd>
            <dt>Languages</dt>
            <dd>{(status.languages || ["hi", "mr", "kn", "en"]).join(", ")}</dd>
          </dl>
        ) : (
          <p className="empty">Checking…</p>
        )}
        <p style={{ fontSize: 13.5, color: "var(--ink-2)", marginTop: 14, marginBottom: 0 }}>
          The speech service loads a ~500 MB model, so it runs as its own
          process rather than blocking the API at startup. Start it with{" "}
          <code>uvicorn backend:app --port 8000</code> inside <code>TestVoice/</code>,
          then open <code>TestVoice/index.html</code>.
        </p>
      </div>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState("report");
  const [profile, setProfile] = useState(RAHUL);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.health().then(({ ok, data }) => setHealth(ok ? data : null));
  }, []);

  return (
    <>
      <header className="masthead">
        <div className="masthead-in">
          <div className="wordmark">
            Artha<span>Saathi</span>
          </div>
          <nav className="tabs">
            {TABS.map(([key, label]) => (
              <button
                key={key}
                className="tab"
                aria-current={tab === key ? "page" : undefined}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="shell">
        {tab !== "onboarding" && <Identity profile={profile} />}

        {tab === "report" && <Report profile={profile} />}
        {tab === "schemes" && <Schemes profile={profile} />}
        {tab === "onboarding" && (
          <Onboarding userId="new-user" onProfile={() => {}} />
        )}
        {tab === "voice" && <Voice />}

        <footer className="foot">
          {health ? (
            <>
              {health.agents} agents · {health.workflows} workflows ·{" "}
              memory {health.store?.includes?.("Postgres") ? "on Neon" : "in-process"} ·{" "}
              providers{" "}
              {Object.entries(health.providers || {})
                .filter(([, on]) => on)
                .map(([n]) => n)
                .join(", ")}
            </>
          ) : (
            <>
              API unreachable — start it with{" "}
              <code>uvicorn server.main:app --port 8000</code>
            </>
          )}
        </footer>
      </main>
    </>
  );
}
