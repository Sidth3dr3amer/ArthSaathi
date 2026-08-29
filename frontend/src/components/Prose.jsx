/*
  A deliberately small markdown renderer.

  The councils are asked for plain prose, but a language model will sometimes
  return `### **Heading**` regardless of instruction, and showing those literal
  hashes and asterisks to a user reads as broken software. This strips or
  renders the handful of constructs that actually appear — headings, bold,
  bullets — and nothing else.

  Not a full markdown parser on purpose: a dependency would be larger than this
  whole screen, and everything beyond these four cases is noise we do not want
  to encourage.
*/

function inline(text) {
  // **bold** and *italic*, applied left to right without nesting.
  const parts = [];
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1]) parts.push(<b key={m.index}>{m[1]}</b>);
    else if (m[2]) parts.push(<i key={m.index}>{m[2]}</i>);
    else parts.push(<code key={m.index}>{m[3]}</code>);
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length ? parts : text;
}

export default function Prose({ text }) {
  if (!text) return null;

  const blocks = [];
  let list = null;

  const flush = () => {
    if (list) {
      blocks.push(
        <ul key={`l${blocks.length}`} style={{ margin: "6px 0", paddingLeft: 18 }}>
          {list.map((li, i) => (
            <li key={i}>{inline(li)}</li>
          ))}
        </ul>
      );
      list = null;
    }
  };

  for (const raw of String(text).split("\n")) {
    const line = raw.trim();

    if (!line) {
      flush();
      continue;
    }

    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      (list ||= []).push(bullet[1]);
      continue;
    }

    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (numbered) {
      (list ||= []).push(numbered[1]);
      continue;
    }

    flush();

    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      blocks.push(
        <div
          key={blocks.length}
          style={{ fontWeight: 600, marginTop: blocks.length ? 12 : 0, marginBottom: 2 }}
        >
          {inline(heading[1].replace(/\*\*/g, ""))}
        </div>
      );
      continue;
    }

    if (/^[-–—_]{3,}$/.test(line)) continue;   // horizontal rules add nothing here

    blocks.push(
      <p key={blocks.length} style={{ margin: "0 0 8px" }}>
        {inline(line)}
      </p>
    );
  }
  flush();

  return <>{blocks}</>;
}
