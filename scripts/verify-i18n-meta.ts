/**
 * Checks the translated catalog metadata and project notebooks against their English sources.
 *   bun scripts/verify-i18n-meta.ts [lang]      (default: fr)
 * - content/i18n/<lang>/*.json must cover every course, module, concept, capstone, roadmap topic and bibliography entry.
 * - content/projects/*.project.<lang>.py must have the same cells and kinds; code/check/solution/task bodies identical.
 */
import fs from "node:fs";
import path from "node:path";
import { parseCells } from "../src/shared/notebook";

const lang = process.argv[2] ?? "fr";
const root = path.resolve(import.meta.dir, "..", "content");
const dir = path.join(root, "i18n", lang);
let problems = 0;
const bad = (m: string) => { problems++; console.error("FAIL " + m); };
const json = (f: string) => JSON.parse(fs.readFileSync(f, "utf8"));
const missing = (what: string, want: string[], have: Iterable<string>) => {
  const h = new Set(have);
  const m = want.filter((k) => !h.has(k));
  if (m.length) bad(`${what}: ${m.length} missing (${m.slice(0, 5).join(", ")}${m.length > 5 ? ", ..." : ""})`);
};
const empty = (what: string, o: Record<string, unknown>) => {
  for (const [k, v] of Object.entries(o)) {
    const vals = typeof v === "object" && v ? Object.values(v as object) : [v];
    if (vals.some((x) => typeof x === "string" && !x.trim())) bad(`${what}: empty text for ${k}`);
  }
};
const read = (f: string) => (fs.existsSync(path.join(dir, f)) ? json(path.join(dir, f)) : (bad(`${f} is missing`), {}));

// courses + modules
const courses = read("courses.json");
missing("courses", json(path.join(root, "courses.json")).map((c: { id: string }) => c.id), Object.keys(courses));
empty("courses", courses);
const modules = read("modules.json");
const path_ = json(path.join(root, "path.json")) as { parts: { modules: { id: string }[] }[] };
missing("modules", [...path_.parts.flatMap((p) => p.modules.map((m) => m.id)), "more"], Object.keys(modules));
empty("modules", modules);

// concepts
const concepts: Record<string, unknown> = {};
for (const f of fs.existsSync(dir) ? fs.readdirSync(dir).filter((x) => /^concepts.*\.json$/.test(x)) : []) Object.assign(concepts, json(path.join(dir, f)));
const cdir = path.join(root, "concepts.d");
const all: { id: string; description: string }[] = [...json(path.join(root, "concepts.json")), ...fs.readdirSync(cdir).filter((f) => f.endsWith(".json")).flatMap((f) => json(path.join(cdir, f)))];
missing("concepts", all.map((c) => c.id), Object.keys(concepts));
empty("concepts", concepts);
for (const c of all) if ((concepts[c.id] as { description?: string })?.description === c.description) bad(`concepts: ${c.id} description is still English`);

// capstones
const caps = read("capstones.json");
for (const cap of json(path.join(root, "capstones.json")) as { id: string; criteria: { id: string }[] }[]) {
  if (!caps[cap.id]) { bad(`capstones: ${cap.id} missing`); continue; }
  missing(`capstones.${cap.id}.criteria`, cap.criteria.map((c) => c.id), Object.keys(caps[cap.id].criteria ?? {}));
}

// roadmap + bibliography
const road = read("roadmap.json"), roadEn = json(path.join(root, "roadmap.json"));
missing("roadmap.stages", roadEn.stages.map((s: { id: string }) => s.id), Object.keys(road.stages ?? {}));
missing("roadmap.topics", roadEn.stages.flatMap((s: { topics: { name: string }[] }) => s.topics.map((t) => t.name)), Object.keys(road.topics ?? {}));
if ((road.platform ?? []).length !== roadEn.platform.length) bad("roadmap.platform length differs");
const bib = read("bibliography.json"), bibEn = json(path.join(root, "bibliography.json"));
if (!bib.provenance) bad("bibliography.provenance missing");
missing("bibliography.library", bibEn.library.map((l: { path: string }) => l.path), Object.keys(bib.library ?? {}));
if ((bib.external ?? []).length !== bibEn.external.length) bad("bibliography.external length differs");

// projects
const pdir = path.join(root, "projects");
for (const f of fs.readdirSync(pdir).filter((x) => x.endsWith(".project.py"))) {
  const tf = path.join(pdir, f.replace(/\.project\.py$/, `.project.${lang}.py`));
  if (!fs.existsSync(tf)) { bad(`${path.basename(tf)} is missing`); continue; }
  const en = parseCells(fs.readFileSync(path.join(pdir, f), "utf8")), tr = parseCells(fs.readFileSync(tf, "utf8"));
  if (en.length !== tr.length) { bad(`${path.basename(tf)}: ${tr.length} cells vs ${en.length}`); continue; }
  const fences = (s: string) => [...s.matchAll(/```[\s\S]*?```/g)].map((m) => m[0]).join("\n");
  const maths = (s: string) => [...s.matchAll(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g)].map((m) => m[0].replace(/\s+/g, "")).sort().join("|");
  en.forEach((c, i) => {
    const x = tr[i], where = `${path.basename(tf)} cell ${i} (${c.kind})`;
    if (x.kind !== c.kind) return bad(`${where}: kind is ${x.kind}`);
    if (c.kind === "meta") { try { const a = JSON.parse(c.body), b = JSON.parse(x.body); for (const k of ["id", "level", "minutes", "requires"]) if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) bad(`${where}: meta.${k} differs`); if (a.outcomes?.length !== b.outcomes?.length || a.stages?.length !== b.stages?.length) bad(`${where}: outcomes/stages length differs`); } catch { bad(`${where}: meta is not valid JSON`); } }
    else if (["code", "check", "solution", "task"].includes(c.kind)) { if (x.body !== c.body) bad(`${where}: code body must be identical`); }
    else { if (fences(x.body) !== fences(c.body)) bad(`${where}: code fences differ`); if (maths(x.body) !== maths(c.body)) bad(`${where}: math differs`); }
  });
}
console.log(`i18n metadata (${lang}): ${problems} problem(s)`);
process.exit(problems ? 1 : 0);
