import { useState } from "react";
import Prose from "./Prose.jsx";
import { COUNCIL_COLOR, COUNCIL_LABEL } from "../lib/api.js";

/*
  The Chamber — the signature of this interface.

  Most financial dashboards show a score and hide the reasoning. The one thing
  this system has that others do not is five councils that examine the same
  finances, reach DIFFERENT conclusions, critique each other, and hand it to a
  judge. So the disagreement is the thing on display, not a footnote.

  The inversion is deliberate: everything else in the report is measurement on
  paper; this is argument, and it should feel like a different room.
*/

function Bench({ verdict, critique }) {
  const council = verdict.council;
  const [open, setOpen] = useState(false);
  const text = verdict.rationale || "";
  // Councils are asked for 60 words. A model that ignores that must not be
  // allowed to turn five benches into a scroll.
  const long = text.length > 340;
  const shown = open || !long ? text : text.slice(0, 320).trimEnd() + "…";

  return (
    <div className="bench">
      <div className="bench-name" style={{ color: COUNCIL_COLOR[council] }}>
        <i style={{ background: COUNCIL_COLOR[council] }} />
        {COUNCIL_LABEL[council] || council}
      </div>
      <p>{shown || <span className="muted">No argument returned.</span>}</p>
      {long && (
        <button className="more" onClick={() => setOpen((o) => !o)}>
          {open ? "Show less" : "Read the full argument"}
        </button>
      )}
      {critique && (
        <div className="critique">
          On the {critique.stance?.replace("critique_of_", "")} position:{" "}
          {(critique.rationale || "").length > 200 && !open
            ? (critique.rationale || "").slice(0, 190).trimEnd() + "…"
            : critique.rationale}
        </div>
      )}
    </div>
  );
}

export default function Chamber({ result, working, query }) {
  if (working) {
    return (
      <section className="chamber">
        <div className="chamber-in">
          <div className="chamber-head">
            <h2>The chamber</h2>
            <div className="sub">
              <span className="working">
                Five councils are examining your finances
                <i /><i /><i /><i /><i />
              </span>
            </div>
          </div>
        </div>
      </section>
    );
  }

  if (!result) return null;

  const verdicts = result.council_verdicts || [];
  const critiques = result.council_critiques || [];

  // A workflow that did not need the councils is not a failure — a card
  // question has a computable answer. Say so rather than showing an empty room.
  if (!verdicts.length) {
    return (
      <section className="chamber quiet">
        <div className="chamber-in">
          <div className="chamber-head">
            <h2>The chamber</h2>
            <div className="sub">
              The <b>{result.workflow}</b> workflow answered this directly — no
              council debate was needed.
            </div>
          </div>
        </div>
      </section>
    );
  }

  const critiqueOf = {};
  for (const c of critiques) critiqueOf[c.council] = c;

  return (
    <section className="chamber">
      <div className="chamber-in">
        <div className="chamber-head">
          <h2>The chamber</h2>
          <div className="sub">
            <b>{verdicts.length}</b> councils argued over{" "}
            <b>&ldquo;{query}&rdquo;</b>
            {critiques.length > 0 && <> · each critiqued another&rsquo;s position</>}
          </div>
        </div>

        <div className="benches">
          {verdicts.map((v) => (
            <Bench key={v.council} verdict={v} critique={critiqueOf[v.council]} />
          ))}
        </div>

        {result.final_decision && (
          <div className="judgment">
            <div className="label">The judgment</div>
            <div className="text"><Prose text={result.final_decision} /></div>
          </div>
        )}
      </div>
    </section>
  );
}
