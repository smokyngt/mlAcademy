/**
 * Translated course content. A translation is an optional sibling file next to the English source, with the SAME ids:
 *   lesson.fr.md            frontmatter (id, title, summary) + translated body
 *   quiz.fr.json            [{ id, prompt, options, explanation }]
 *   cards*.fr.json          [{ id, front, back }]
 *   exercises/*.fr.json     { id, title, prompt, hints }
 * Anything without a translation falls back to the English text, so the app is never empty.
 */
import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";
import { ACADEMY_ROOT } from "./db";

export interface Overlays {
  lessons: Map<string, { title?: string; summary?: string; body: string }>;
  questions: Map<string, { prompt: string; options?: string[]; explanation: string }>;
  cards: Map<string, { front: string; back: string }>;
  exercises: Map<string, { title?: string; prompt?: string; hints?: string[] }>;
}

export function loadOverlays(lang: string, root = path.join(ACADEMY_ROOT, "content", "authored")): Overlays {
  const o: Overlays = { lessons: new Map(), questions: new Map(), cards: new Map(), exercises: new Map() };
  if (!fs.existsSync(root)) return o;
  const readJson = (f: string) => JSON.parse(fs.readFileSync(f, "utf8"));
  for (const course of fs.readdirSync(root)) {
    const cdir = path.join(root, course);
    if (!fs.statSync(cdir).isDirectory()) continue;
    for (const lesson of fs.readdirSync(cdir)) {
      const dir = path.join(cdir, lesson);
      if (!fs.statSync(dir).isDirectory()) continue;
      const lf = path.join(dir, `lesson.${lang}.md`);
      if (fs.existsSync(lf)) {
        const m = fs.readFileSync(lf, "utf8").match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
        if (m) {
          const fm = YAML.parse(m[1].replace(/^(\s+- )(?!["'\[{])(.*)$/gm, (_, a: string, b: string) => a + JSON.stringify(b.trim())).replace(/^(title|summary):[ \t]*(?![ 	"'|>\[{])(.+)$/gm, (_: string, k: string, v: string) => `${k}: ${JSON.stringify(v.trim())}`));
          o.lessons.set(String(fm.id), { title: fm.title, summary: fm.summary, body: m[2] });
        }
      }
      const qf = path.join(dir, `quiz.${lang}.json`);
      if (fs.existsSync(qf)) for (const q of readJson(qf)) o.questions.set(q.id, q);
      for (const f of fs.readdirSync(dir).filter((x) => new RegExp(`^cards.*\\.${lang}\\.json$`).test(x))) for (const c of readJson(path.join(dir, f))) o.cards.set(c.id, c);
      const ed = path.join(dir, "exercises");
      if (fs.existsSync(ed)) for (const f of fs.readdirSync(ed).filter((x) => x.endsWith(`.${lang}.json`))) { const e = readJson(path.join(ed, f)); o.exercises.set(e.id, e); }
    }
  }
  return o;
}

/**
 * Translated catalog metadata, from content/i18n/<lang>/*.json (all optional, missing keys fall back to English):
 *   courses.json      { <courseId>: { title, summary } }
 *   modules.json      { <moduleId>: title }                        (path.json modules)
 *   concepts*.json    { <conceptId>: { title, description } }      (several files are merged)
 *   capstones.json    { <capstoneId>: { title, brief, criteria: { <criterionId>: { text, evidence } } } }
 *   roadmap.json      { stages: { <stageId>: name }, topics: { <English topic name>: { name, note? } }, platform: [{ area, item }] }  (platform is positional)
 *   bibliography.json { provenance, library: { <path>: { used } }, external: [{ for }] }                                              (external is positional)
 */
export interface MetaOverlay {
  courses: Record<string, { title?: string; summary?: string }>;
  modules: Record<string, string>;
  concepts: Record<string, { title?: string; description?: string }>;
  capstones: Record<string, { title?: string; brief?: string; criteria?: Record<string, { text?: string; evidence?: string }> }>;
  roadmap: { stages?: Record<string, string>; topics?: Record<string, { name?: string; note?: string }>; platform?: { area?: string; item?: string }[] };
  bibliography: { provenance?: string; library?: Record<string, { used?: string }>; external?: { for?: string }[] };
}

export function loadMetaOverlay(lang: string, root = path.join(ACADEMY_ROOT, "content", "i18n")): MetaOverlay {
  const dir = path.join(root, lang);
  const o: MetaOverlay = { courses: {}, modules: {}, concepts: {}, capstones: {}, roadmap: {}, bibliography: {} };
  if (!fs.existsSync(dir)) return o;
  const read = (f: string) => JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
  for (const f of fs.readdirSync(dir).sort()) {
    if (!f.endsWith(".json")) continue;
    if (/^concepts.*\.json$/.test(f)) Object.assign(o.concepts, read(f));
    else if (f === "courses.json") o.courses = read(f);
    else if (f === "modules.json") o.modules = read(f);
    else if (f === "capstones.json") o.capstones = read(f);
    else if (f === "roadmap.json") o.roadmap = read(f);
    else if (f === "bibliography.json") o.bibliography = read(f);
  }
  return o;
}
