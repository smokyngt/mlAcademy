/**
 * Checks that every translated file faithfully mirrors its English source.
 *   bun scripts/verify-translations.ts [course-folder ...]      (default: all)
 * A translation may change prose only. Formulas, code, ids, list lengths and structure must be identical.
 */
import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";

const root = path.resolve(import.meta.dir, "..", "content", "authored");
const only = process.argv.slice(2);
let problems = 0;
let checked = 0;
const bad = (f: string, m: string) => { problems++; console.error(`FAIL ${path.relative(root, f)}: ${m}`); };

const mathTokens = (s: string) => [...s.matchAll(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g)].map((m) => m[0].replace(/\s+/g, ""));
const codeBlocks = (s: string) => [...s.matchAll(/```[\s\S]*?```/g)].map((m) => m[0]);
const headings = (s: string) => (s.match(/^#{1,3} /gm) ?? []).length;
const sameList = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const sortedEq = (a: string[], b: string[]) => sameList([...a].sort(), [...b].sort());

function body(md: string) {
  md = md.split("\r\n").join("\n"); // line endings are not content
  const m = md.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  return m ? { fm: m[1], text: m[2] } : null;
}
function fmField(fm: string, key: string) {
  const y = fm.replace(/^(\s+- )(?!["'\[{])(.*)$/gm, (_, a: string, b: string) => a + JSON.stringify(b.trim())).replace(/^(title|summary):[ \t]*(?![ 	"'|>\[{])(.+)$/gm, (_: string, k: string, v: string) => `${k}: ${JSON.stringify(v.trim())}`);
  return YAML.parse(y)?.[key];
}
const readJson = (f: string) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { bad(f, "invalid JSON: " + String(e).slice(0, 80)); return null; } };

for (const course of fs.readdirSync(root)) {
  if (only.length && !only.includes(course)) continue;
  const cdir = path.join(root, course);
  if (!fs.statSync(cdir).isDirectory()) continue;
  for (const lesson of fs.readdirSync(cdir)) {
    const dir = path.join(cdir, lesson);
    if (!fs.existsSync(path.join(dir, "lesson.md"))) continue;
    const frLesson = path.join(dir, "lesson.fr.md");
    if (!fs.existsSync(frLesson)) continue;
    checked++;

    // lesson body
    const en = body(fs.readFileSync(path.join(dir, "lesson.md"), "utf8"))!;
    const fr = body(fs.readFileSync(frLesson, "utf8"));
    if (!fr) { bad(frLesson, "missing frontmatter"); continue; }
    if (fmField(fr.fm, "id") !== fmField(en.fm, "id")) bad(frLesson, `id differs (${fmField(fr.fm, "id")} vs ${fmField(en.fm, "id")})`);
    if (!fmField(fr.fm, "title")) bad(frLesson, "no title");
    if (!sameList(codeBlocks(fr.text), codeBlocks(en.text))) bad(frLesson, "fenced code blocks (including viz/mermaid) must be identical to the English ones");
    if (!sortedEq(mathTokens(fr.text), mathTokens(en.text))) bad(frLesson, `math differs: ${mathTokens(en.text).length} English formulas vs ${mathTokens(fr.text).length}; formulas must be copied unchanged`);
    if (headings(fr.text) !== headings(en.text)) bad(frLesson, `heading count differs (${headings(fr.text)} vs ${headings(en.text)})`);
    const links = (s: string) => [...s.matchAll(/\[\[[\w.-]+\]\]/g)].map((m) => m[0]);
    if (!sortedEq(links(fr.text), links(en.text))) bad(frLesson, "[[lesson-id]] links differ");

    // quiz
    const qf = path.join(dir, "quiz.fr.json");
    if (fs.existsSync(qf)) {
      const frq = readJson(qf), enq = readJson(path.join(dir, "quiz.json"));
      if (frq && enq) {
        if (!sameList(frq.map((q: any) => q.id), enq.map((q: any) => q.id))) bad(qf, "question ids/order differ");
        for (const q of frq) {
          const e = enq.find((x: any) => x.id === q.id);
          if (!e) continue;
          if ((q.options?.length ?? 0) !== (e.options?.length ?? 0)) bad(qf, `${q.id}: option count differs`);
          if (!sortedEq(mathTokens(JSON.stringify([q.prompt, q.options, q.explanation])), mathTokens(JSON.stringify([e.prompt, e.options, e.explanation])))) bad(qf, `${q.id}: math differs`);
        }
      }
    } else bad(frLesson, "quiz.fr.json missing");

    // cards
    const enCards = fs.readdirSync(dir).filter((f) => /^cards(-[a-z0-9]+)?\.json$/.test(f));
    for (const f of enCards) {
      const frf = path.join(dir, f.replace(".json", ".fr.json"));
      if (!fs.existsSync(frf)) { bad(frLesson, `${path.basename(frf)} missing`); continue; }
      const a = readJson(path.join(dir, f)), b = readJson(frf);
      if (a && b) {
        if (!sameList(a.map((c: any) => c.id), b.map((c: any) => c.id))) bad(frf, "card ids/order differ");
        for (const c of b) {
          const e = a.find((x: any) => x.id === c.id);
          if (e && !sortedEq(mathTokens(c.front + c.back), mathTokens(e.front + e.back))) bad(frf, `${c.id}: math differs`);
        }
      }
    }

    // exercises: translated statement + hints
    const ed = path.join(dir, "exercises");
    if (fs.existsSync(ed)) {
      for (const f of fs.readdirSync(ed).filter((x) => x.endsWith(".exercise.txt"))) {
        const src = fs.readFileSync(path.join(ed, f), "utf8");
        const meta = JSON.parse(src.split("#@@ meta")[1].split("#@@")[0].trim());
        const frf = path.join(ed, f.replace(".exercise.txt", ".fr.json"));
        if (!fs.existsSync(frf)) { bad(frLesson, `exercise translation ${path.basename(frf)} missing`); continue; }
        const t = readJson(frf);
        if (t) {
          if (t.id !== meta.id) bad(frf, `id ${t.id} should be ${meta.id}`);
          if (!t.title || !t.prompt) bad(frf, "title and prompt are required");
          if ((t.hints?.length ?? 0) !== meta.hints.length) bad(frf, `hints: ${t.hints?.length} vs ${meta.hints.length}`);
        }
      }
    }
  }
}
console.log(`${checked} translated lesson(s) checked, ${problems} problem(s)`);
process.exit(problems ? 1 : 0);
