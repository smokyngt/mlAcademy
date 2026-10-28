/**
 * Validates every `figures.json` (see docs/FIGURES.md): structure, ids, section index, Mermaid syntax (Mermaid's own
 * parser) and function-plot specs (rendered headlessly in jsdom; invalid expressions throw).
 *   bun scripts/verify-figures.ts [course-folder ...]
 */
import fs from "node:fs";
import path from "node:path";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><body></body>", { pretendToBeVisual: true });
const g = globalThis as any;
g.window = dom.window; g.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
for (const k of ["SVGElement", "Element", "HTMLElement", "Node"]) g[k] = (dom.window as any)[k];

const mermaid = (await import("mermaid")).default;
const functionPlot = (await import("function-plot")).default as (o: any) => unknown;

const root = path.resolve(import.meta.dir, "..", "content", "authored");
const only = process.argv.slice(2);
let problems = 0, files = 0, figures = 0;
const bad = (f: string, m: string) => { problems++; console.error(`FAIL ${path.relative(root, f)}: ${m}`); };

const h2Count = (md: string) => { let n = 0, code = false; for (const l of md.split("\n")) { if (l.trim().startsWith("```")) code = !code; else if (!code && /^## /.test(l)) n++; } return n; };

for (const course of fs.readdirSync(root)) {
  if (only.length && !only.includes(course)) continue;
  const cdir = path.join(root, course);
  if (!fs.statSync(cdir).isDirectory()) continue;
  for (const lesson of fs.readdirSync(cdir)) {
    const f = path.join(cdir, lesson, "figures.json");
    if (!fs.existsSync(f)) continue;
    files++;
    let list: any[];
    try { list = JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { bad(f, "invalid JSON: " + String(e).slice(0, 80)); continue; }
    if (!Array.isArray(list)) { bad(f, "must be an array"); continue; }
    const sections = h2Count(fs.readFileSync(path.join(cdir, lesson, "lesson.md"), "utf8"));
    const ids = new Set<string>();
    for (const fig of list) {
      figures++;
      const tag = `[${fig?.id}]`;
      if (typeof fig?.id !== "string" || ids.has(fig.id)) bad(f, `${tag} id missing or duplicated`);
      ids.add(fig?.id);
      if (!Number.isInteger(fig?.after) || fig.after < 0 || fig.after > sections) bad(f, `${tag} "after" must be an integer in 0..${sections}`);
      if (!fig?.caption?.en) bad(f, `${tag} caption.en is required`);
      if (fig?.kind === "mermaid") {
        for (const lang of ["en", "fr"]) {
          const code = fig?.code?.[lang];
          if (lang === "en" && !code) { bad(f, `${tag} code.en is required`); continue; }
          if (!code) continue;
          try { await mermaid.parse(code); } catch (e) { bad(f, `${tag} mermaid (${lang}) does not parse: ${String(e).split("\n").slice(0, 3).join(" | ").slice(0, 200)}`); }
        }
      } else if (fig?.kind === "plot") {
        const s = fig.spec;
        if (!s || !Array.isArray(s.data) || s.data.length === 0) { bad(f, `${tag} plot needs spec.data`); continue; }
        for (const k of ["xDomain", "yDomain"]) if (s[k] && (!Array.isArray(s[k]) || s[k].length !== 2 || s[k][0] >= s[k][1])) bad(f, `${tag} ${k} must be [min, max]`);
        try {
          const el = document.createElement("div");
          document.body.appendChild(el);
          functionPlot({ target: el, width: 480, height: s.height ?? 300, grid: s.grid ?? true, xAxis: { domain: s.xDomain ?? [-5, 5] }, yAxis: { domain: s.yDomain ?? [-5, 5] }, data: s.data, annotations: s.annotations, disableZoom: true });
          if (!el.querySelector("svg")) bad(f, `${tag} plot produced no SVG`);
          el.remove();
        } catch (e) { bad(f, `${tag} plot does not render: ${String(e).slice(0, 160)}`); }
      } else if (fig?.kind === "video") {
        if (typeof fig.src !== "string" || !fs.existsSync(path.resolve(import.meta.dir, "..", "src", "web", "public", "." + fig.src))) bad(f, `${tag} video src must be an existing file under src/web/public`);
      } else bad(f, `${tag} kind must be "mermaid", "plot" or "video"`);
    }
  }
}
console.log(`${files} figures file(s), ${figures} figure(s), ${problems} problem(s)`);
process.exit(problems ? 1 : 0);
