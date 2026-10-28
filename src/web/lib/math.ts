/**
 * Normalises display math so the renderer can never choke on it.
 *  - `$$ ... $$` alone on one line becomes a real display block (inline math rejects \tag, \begin{aligned}, ...).
 *  - `$$` followed by content on the same line (or content followed by a closing `$$`) is split onto its own lines
 *    (remark-math reads text after an opening `$$` as fence metadata, so such a block never closes).
 * Fenced code blocks are left untouched.
 */
export function normalizeMath(src: string): string {
  const out: string[] = [];
  let inCode = false;
  let inBlock = false;
  for (const line of src.split("\n")) {
    const s = line.trim();
    if (s.startsWith("```")) { inCode = !inCode; out.push(line); continue; }
    if (inCode) { out.push(line); continue; }
    if (inBlock) {
      if (s === "$$") { inBlock = false; out.push(line); }
      else if (s.endsWith("$$")) { out.push(line.slice(0, line.lastIndexOf("$$")).trimEnd(), "$$"); inBlock = false; }
      else out.push(line);
      continue;
    }
    if (s === "$$") { inBlock = true; out.push(line); continue; }
    if (s.startsWith("$$")) {
      const rest = s.slice(2);
      if (rest.endsWith("$$") && rest.length > 2) out.push("$$", rest.slice(0, -2).trim(), "$$");
      else { out.push("$$", rest); inBlock = true; }
      continue;
    }
    out.push(line);
  }
  return out.join("\n");
}
