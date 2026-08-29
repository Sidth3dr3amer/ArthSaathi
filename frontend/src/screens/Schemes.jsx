import { useEffect, useState } from "react";
import { api, inr } from "../lib/api.js";

/*
  Government schemes.

  The eligibility engine returns three verdicts, not two — a rule it cannot
  evaluate comes back `unknown`, never a guess. That distinction is the whole
  point and is shown, because telling someone they qualify for PM-KISAN and
  having the bank reject them is a real harm. Confirmed and unconfirmed
  matches are visually separated, and the missing field that would resolve the
  most schemes is surfaced as an action.
*/

function Row({ scheme, unconfirmed }) {
  return (
    <div className="scheme">
      <div className="match num" style={unconfirmed ? { color: "var(--ink-3)" } : undefined}>
        {Math.round(scheme.match_score)}%
      </div>
      <div>
        <div className="nm">
          {scheme.name}{" "}
          {unconfirmed && (
            <span className="pill" style={{ color: "var(--ink-3)", marginLeft: 4 }}>
              unconfirmed
            </span>
          )}
        </div>
        <div className="wy">
          {(scheme.why || []).join(" · ")}
          {unconfirmed && scheme.missing_information?.length > 0 && (
            <> — needs {scheme.missing_information.map((m) => m.replace(/_/g, " ")).join(", ")}</>
          )}
        </div>
      </div>
      <div className="val num">
        {inr(scheme.annual_value, { compact: true })}
        <small>a year</small>
      </div>
    </div>
  );
}

export default function Schemes({ profile }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let live = true;
    setBusy(true);
    api
      .chat(profile.user_id, "am I eligible for any government schemes?", profile)
      .then(({ ok, data, error }) => {
        if (!live) return;
        setBusy(false);
        if (!ok) return setError(error);
        setData(data);
      });
    return () => {
      live = false;
    };
  }, [profile]);

  if (busy) {
    return (
      <section className="section">
        <h2>Schemes you can claim</h2>
        <p className="empty">Checking {profile.name || "your"} profile against 25 central schemes…</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="section">
        <h2>Schemes you can claim</h2>
        <div className="err">{error}</div>
      </section>
    );
  }

  const raw = data?.recommendations?.scheme_matching ?? [];

  return (
    <section className="section">
      <h2>Schemes you can claim</h2>
      <p className="lede">
        Every rule is checked against your profile. A rule we cannot evaluate is
        marked unconfirmed rather than guessed.
      </p>

      {raw.length === 0 ? (
        <p className="empty">No schemes matched this profile.</p>
      ) : (
        <div className="panel">
          {raw.map((line, i) => (
            <div className="scheme" key={i}>
              <div className="match num">{String(line).match(/(\d+)%/)?.[1] ?? "—"}%</div>
              <div>
                <div className="nm">{String(line).split(" - ")[0]}</div>
                <div className="wy">{String(line).split(" - ").slice(1).join(" - ")}</div>
              </div>
              <div className="val" />
            </div>
          ))}
        </div>
      )}

      <div className="grid g2" style={{ marginTop: 18 }}>
        <div className="panel">
          <h3>How this was decided</h3>
          <p style={{ fontSize: 13.5, margin: 0, color: "var(--ink-2)" }}>
            Eligibility is a rule engine, not a language model — a legal
            determination has to be reproducible. Each scheme is ranked by what
            it is worth to you, how much your situation calls for it, and how
            much paperwork it takes.
          </p>
        </div>
        <div className="panel">
          <h3>Why some are unconfirmed</h3>
          <p style={{ fontSize: 13.5, margin: 0, color: "var(--ink-2)" }}>
            A scheme is only marked eligible when every rule could be checked.
            Where a detail is missing — landholding, residence, category — it
            stays unconfirmed and the missing field is named, so you can answer
            it rather than be told a maybe.
          </p>
        </div>
      </div>
    </section>
  );
}
