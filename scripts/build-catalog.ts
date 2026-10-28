/**
 * Content pipeline: content/ (authored) + external source material -> SQLite curriculum tables.
 *
 *   bun run sync
 *
 * Idempotent. Only static curriculum tables are rewritten; learner-state tables are never touched.
 * Source files are only ever read.
 */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import YAML from "yaml";
import { z } from "zod";
import { openDb, ACADEMY_ROOT } from "../src/server/db";
import * as t from "../src/server/db/schema";

const REPO = path.resolve(ACADEMY_ROOT, "..");
const CONTENT = path.join(ACADEMY_ROOT, "content");

/** ai-engineering-from-scratch phases to import (see docs/SOURCE_INVENTORY.md for why the others are left out). */
const IMPORT_PHASES: Record<string, { stage: string; course: string }> = {
  "01": { stage: "I", course: "Source library: math foundations" },
  "02": { stage: "III", course: "Source library: ML fundamentals" },
  "03": { stage: "IV", course: "Source library: deep learning core" },
  "04": { stage: "V", course: "Source library: computer vision" },
  "05": { stage: "V", course: "Source library: NLP foundations to advanced" },
  "06": { stage: "V", course: "Source library: speech and audio" },
  "07": { stage: "V", course: "Source library: transformers deep dive" },
  "08": { stage: "V", course: "Source library: generative AI" },
  "09": { stage: "VI", course: "Source library: reinforcement learning" },
  "10": { stage: "VI", course: "Source library: LLMs from scratch" },
  "11": { stage: "VI", course: "Source library: LLM engineering" },
  "12": { stage: "VI", course: "Source library: multimodal AI" },
  "18": { stage: "VI", course: "Source library: ethics, safety, alignment" },
};

/* ---------------- schemas for authored content ---------------- */

const Concept = z.object({
  id: z.string().regex(/^[a-z0-9.-]+$/),
  title: z.string(),
  description: z.string(),
  stage: z.string(),
  prereqs: z.array(z.string()).default([]),
});
const Course = z.object({ id: z.string(), part: z.enum(["math", "ml"]), title: z.string(), summary: z.string(), stage: z.string() });
const Front = z.object({
  id: z.string(),
  title: z.string(),
  summary: z.string(),
  minutes: z.number().int().positive(),
  teaches: z.array(z.string()).min(1),
  requires: z.array(z.string()).default([]),
  sources: z.array(z.string()).default([]),
});
const Question = z.object({
  id: z.string(),
  concept: z.string(),
  kind: z.enum(["mcq", "numeric"]).default("mcq"),
  stage: z.enum(["pre", "check", "assessment"]).default("check"),
  prompt: z.string(),
  options: z.array(z.string()).default([]),
  answer: z.union([z.number(), z.object({ value: z.number(), tol: z.number() })]),
  explanation: z.string(),
});
const Card = z.object({ id: z.string(), concept: z.string(), front: z.string(), back: z.string() });
const ExerciseMeta = z.object({
  id: z.string(),
  title: z.string(),
  difficulty: z.number().int().min(1).max(5),
  runtime: z.enum(["pyodide", "docker"]),
  concepts: z.array(z.string()).min(1),
  hints: z.array(z.string()),
});

const readJson = (p: string) => JSON.parse(fs.readFileSync(p, "utf8"));
const read = (p: string) => fs.readFileSync(p, "utf8");
const sha = (s: string) => crypto.createHash("sha1").update(s).digest("hex").slice(0, 12);
const rel = (p: string) => path.relative(REPO, p).replaceAll("\\", "/");

function parseFrontmatter(src: string): { data: unknown; body: string } {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!m) throw new Error("missing frontmatter");
  // List items are free text (they may contain ": "), so quote them before YAML sees them.
  // Free-text fields (they often contain ": ") are quoted before YAML sees them, so prose can never break parsing.
  const yaml = m[1].replace(/^(\s+- )(?!["'\[{])(.*)$/gm, (_, lead: string, text: string) => lead + JSON.stringify(text.trim())).replace(/^(title|summary):[ \t]*(?![ 	"'|>\[{])(.+)$/gm, (_: string, k: string, v: string) => `${k}: ${JSON.stringify(v.trim())}`);
  return { data: YAML.parse(yaml), body: m[2] };
}


/** Exercise file format: sections introduced by a line `#@@ meta|prompt|starter|tests|solution`. */
function parseExercise(src: string) {
  const parts: Record<string, string> = {};
  let cur = "";
  for (const line of src.replace(/\r\n/g, "\n").split("\n")) {
    const m = line.match(/^#@@ (\w+)\s*$/);
    if (m) {
      cur = m[1];
      parts[cur] = "";
    } else if (cur) parts[cur] += line + "\n";
  }
  for (const k of ["meta", "prompt", "starter", "tests", "solution"]) if (!(k in parts)) throw new Error(`exercise missing section ${k}`);
  const meta = ExerciseMeta.parse(JSON.parse(parts.meta));
  return {
    id: meta.id, title: meta.title, difficulty: meta.difficulty, runtime: meta.runtime,
    prompt: parts.prompt.trim(),
    starter: parts.starter.trimEnd() + "\n",
    tests: parts.tests.trimEnd() + "\n",
    hints: meta.hints,
    solution: parts.solution.trimEnd() + "\n",
    conceptIds: meta.concepts,
  };
}

/* ---------------- collection ---------------- */

const courses: (typeof t.courses.$inferInsert)[] = [];
const lessons: (typeof t.lessons.$inferInsert)[] = [];
const concepts: (typeof t.concepts.$inferInsert)[] = [];
const edges: (typeof t.conceptEdges.$inferInsert)[] = [];
const lessonConcepts: (typeof t.lessonConcepts.$inferInsert)[] = [];
const questions: (typeof t.questions.$inferInsert)[] = [];
const exercises: (typeof t.exercises.$inferInsert)[] = [];
const flashcards: (typeof t.flashcards.$inferInsert)[] = [];
const conceptIds = new Set<string>();
const problems: string[] = [];

function addLesson(l: Omit<typeof t.lessons.$inferInsert, "hash">) {
  lessons.push({ ...l, hash: sha(l.body) });
}

/* ---- authored ---- */
function loadAuthored() {
  const cs = z.array(Concept).parse(readJson(path.join(CONTENT, "concepts.json")));
  const cd = path.join(CONTENT, "concepts.d");
  if (fs.existsSync(cd)) for (const f of fs.readdirSync(cd).filter((x) => x.endsWith(".json")).sort()) cs.push(...z.array(Concept).parse(readJson(path.join(cd, f))));
  for (const c of cs) {
    concepts.push({ id: c.id, title: c.title, description: c.description, stage: c.stage });
    conceptIds.add(c.id);
  }
  for (const c of cs)
    for (const p of c.prereqs) {
      if (!conceptIds.has(p)) problems.push(`concept ${c.id}: unknown prerequisite ${p}`);
      else edges.push({ conceptId: c.id, prerequisiteId: p });
    }

  const cr = z.array(Course).parse(readJson(path.join(CONTENT, "courses.json")));
  cr.forEach((c, i) => courses.push({ ...c, kind: "authored", position: i }));

  const authoredRoot = path.join(CONTENT, "authored");
  for (const course of cr) {
    const cdir = path.join(authoredRoot, course.id);
    if (!fs.existsSync(cdir)) continue;
    const lessonDirs = fs.readdirSync(cdir).filter((d) => fs.existsSync(path.join(cdir, d, "lesson.md"))).sort();
    lessonDirs.forEach((d, idx) => {
      const dir = path.join(cdir, d);
      const { data, body } = parseFrontmatter(read(path.join(dir, "lesson.md")));
      const fm = Front.parse(data);
      for (const c of [...fm.teaches, ...fm.requires]) if (!conceptIds.has(c)) problems.push(`${fm.id}: unknown concept ${c}`);
      addLesson({
        id: fm.id,
        courseId: course.id,
        position: idx,
        title: fm.title,
        summary: fm.summary,
        kind: "authored",
        minutes: fm.minutes,
        body,
        sources: fm.sources,
      });
      fm.teaches.forEach((c) => lessonConcepts.push({ lessonId: fm.id, conceptId: c, role: "teaches" }));
      fm.requires.forEach((c) => lessonConcepts.push({ lessonId: fm.id, conceptId: c, role: "requires" }));

      const qf = path.join(dir, "quiz.json");
      if (fs.existsSync(qf))
        for (const q of z.array(Question).parse(readJson(qf))) {
          if (!conceptIds.has(q.concept)) problems.push(`${q.id}: unknown concept ${q.concept}`);
          if (q.kind === "mcq" && typeof q.answer === "number" && (q.answer < 0 || q.answer >= q.options.length))
            problems.push(`${q.id}: answer index out of range`);
          questions.push({ id: q.id, lessonId: fm.id, conceptId: q.concept, kind: q.kind, stage: q.stage, prompt: q.prompt, options: q.options, answer: q.answer, explanation: q.explanation });
        }

      for (const cf of fs.readdirSync(dir).filter((f) => /^cards(-[a-z0-9]+)?\.json$/.test(f)).sort().map((f) => path.join(dir, f)))
        for (const c of z.array(Card).parse(readJson(cf))) {
          if (!conceptIds.has(c.concept)) problems.push(`${c.id}: unknown concept ${c.concept}`);
          flashcards.push({ id: c.id, lessonId: fm.id, conceptId: c.concept, front: c.front, back: c.back });
        }

      const ed = path.join(dir, "exercises");
      if (fs.existsSync(ed))
        for (const f of fs.readdirSync(ed).filter((x) => x.endsWith(".exercise.txt")).sort()) {
          const ex = parseExercise(read(path.join(ed, f)));
          exercises.push({ ...ex, lessonId: fm.id });
        }
    });
  }
}

/* ---- ai-engineering-from-scratch ---- */
const SRC = path.join(REPO, "ai-engineering-from-scratch", "phases");

function loadAiEngineering() {
  if (!fs.existsSync(SRC)) return;
  let pos = courses.length;
  for (const phaseDir of fs.readdirSync(SRC).sort()) {
    const num = phaseDir.slice(0, 2);
    const cfg = IMPORT_PHASES[num];
    if (!cfg) continue;
    const cid = `src-${phaseDir}`;
    courses.push({ id: cid, part: num === "01" ? "math" : "ml", title: cfg.course, summary: `Imported from ai-engineering-from-scratch/phases/${phaseDir}. Original text, quizzes and code, unmodified.`, stage: cfg.stage, kind: "source", position: pos++ });
    const lessonDirs = fs.readdirSync(path.join(SRC, phaseDir)).filter((d) => fs.statSync(path.join(SRC, phaseDir, d)).isDirectory()).sort();
    lessonDirs.forEach((ld, idx) => {
      const dir = path.join(SRC, phaseDir, ld);
      const md = path.join(dir, "docs", "en.md");
      if (!fs.existsSync(md)) return;
      let body = read(md);
      const title = (body.match(/^#\s+(.+)$/m)?.[1] ?? ld).trim();
      const summary = (body.match(/^>\s+(.+)$/m)?.[1] ?? "").trim();
      const minutes = Number(body.match(/\*\*Time:\*\*\s*~?(\d+)/)?.[1] ?? 45);
      const sources = [rel(md)];
      const codeDir = path.join(dir, "code");
      if (fs.existsSync(codeDir)) {
        const py = fs.readdirSync(codeDir).filter((f) => f.endsWith(".py")).sort();
        for (const f of py) sources.push(rel(path.join(codeDir, f)));
        const first = py[0] && path.join(codeDir, py[0]);
        if (first && fs.statSync(first).size < 14_000)
          body += `\n\n## Source code: \`${py[0]}\`\n\n\`\`\`python\n${read(first).trimEnd()}\n\`\`\`\n`;
      }
      const id = `src.${phaseDir}.${ld}`;
      const cId = `src.${num}.${ld.slice(0, 2)}`;
      addLesson({ id, courseId: cid, position: idx, title, summary, kind: "source", minutes, body, sources });
      concepts.push({ id: cId, title, description: summary || title, stage: cfg.stage });
      conceptIds.add(cId);
      lessonConcepts.push({ lessonId: id, conceptId: cId, role: "teaches" });
      const qf = path.join(dir, "quiz.json");
      if (fs.existsSync(qf)) {
        sources.push(rel(qf));
        const raw = readJson(qf);
        (raw.questions ?? []).forEach((q: any, i: number) => {
          if (!q || typeof q.question !== "string" || !Array.isArray(q.options) || typeof q.correct !== "number") return;
          questions.push({ id: `${id}#${i}`, lessonId: id, conceptId: cId, kind: "mcq", stage: q.stage ?? "check", prompt: q.question, options: q.options, answer: q.correct, explanation: q.explanation ?? "" });
        });
      }
    });
  }
}

/* ---- notebook-based books ---- */
function notebookToMarkdown(file: string): string {
  const nb = readJson(file);
  const out: string[] = [];
  for (const cell of nb.cells ?? []) {
    const src = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source ?? "");
    if (!src.trim()) continue;
    if (cell.cell_type === "markdown") out.push(src.replace(/<img[^>]*>/g, "").replace(/!\[[^\]]*\]\([^)]*\)/g, "").trim());
    else if (cell.cell_type === "code") out.push("```python\n" + src.trimEnd() + "\n```");
  }
  return out.filter(Boolean).join("\n\n");
}

function loadNotebookBook(opts: { dir: string; courseId: string; title: string; summary: string; stage: string; pick: (chDir: string) => string | null; lessonTitle: (chDir: string, md: string) => string }) {
  if (!fs.existsSync(opts.dir)) return;
  courses.push({ id: opts.courseId, part: "ml", title: opts.title, summary: opts.summary, stage: opts.stage, kind: "source", position: courses.length });
  const chDirs = fs.readdirSync(opts.dir).filter((d) => /^(ch\d+|appendix-[A-Z]|ch[A-Z])/.test(d) && fs.statSync(path.join(opts.dir, d)).isDirectory()).sort();
  let idx = 0;
  for (const ch of chDirs) {
    const nb = opts.pick(path.join(opts.dir, ch));
    if (!nb) continue;
    const md = notebookToMarkdown(nb);
    if (md.length < 200) continue;
    const id = `${opts.courseId}.${ch.toLowerCase()}`;
    const title = opts.lessonTitle(ch, md);
    addLesson({ id, courseId: opts.courseId, position: idx++, title, summary: "", kind: "source", minutes: Math.max(20, Math.round(md.length / 1500)), body: md, sources: [rel(nb)] });
    const cId = `src.${opts.courseId}.${ch.toLowerCase()}`;
    concepts.push({ id: cId, title, description: title, stage: opts.stage });
    conceptIds.add(cId);
    lessonConcepts.push({ lessonId: id, conceptId: cId, role: "teaches" });
  }
}

function firstHeading(md: string, fallback: string) {
  return (md.match(/^#{1,2}\s+(.+)$/m)?.[1] ?? fallback).replace(/[*_`]/g, "").trim();
}

function pickMainNotebook(chDir: string): string | null {
  const dirs = fs.readdirSync(chDir).filter((d) => d.startsWith("01_") || d === "01_main-chapter-code");
  const main = dirs.length ? path.join(chDir, dirs[0]) : chDir;
  if (!fs.existsSync(main)) return null;
  const nbs = fs.readdirSync(main).filter((f) => f.endsWith(".ipynb") && !/exercise|solution|bonus/i.test(f)).sort();
  if (!nbs.length) return null;
  const base = path.basename(chDir).toLowerCase();
  return path.join(main, nbs.find((f) => f.toLowerCase().startsWith(base)) ?? nbs[0]);
}

function pickFlat(chDir: string): string | null {
  const nbs = fs.readdirSync(chDir).filter((f) => f.endsWith(".ipynb")).sort();
  return nbs.length ? path.join(chDir, nbs[0]) : null;
}

/* ---------------- main ---------------- */

loadAuthored();
// Library files are SOURCE MATERIAL for writing lessons, not lessons themselves. Importing them verbatim is opt-in (IMPORT_SOURCES=1).
const IMPORT = process.env.IMPORT_SOURCES === "1";
if (IMPORT) loadAiEngineering();
if (IMPORT) loadNotebookBook({
  dir: path.join(REPO, "LLMs-from-scratch"),
  courseId: "src-llms-from-scratch",
  title: "Source library: Build a Large Language Model (From Scratch)",
  summary: "Chapter notebooks from LLMs-from-scratch (Raschka): text data, attention, GPT, pretraining, fine-tuning. Markdown and code cells only; outputs dropped.",
  stage: "VI",
  pick: pickMainNotebook,
  lessonTitle: (ch, md) => firstHeading(md, ch),
});
if (IMPORT) loadNotebookBook({
  dir: path.join(REPO, "reasoning-from-scratch"),
  courseId: "src-reasoning-from-scratch",
  title: "Source library: Reasoning models from scratch",
  summary: "Chapter notebooks from reasoning-from-scratch: inference-time scaling, evaluation, reinforcement learning for reasoning.",
  stage: "VII",
  pick: (d) => pickMainNotebook(d) ?? pickFlat(d),
  lessonTitle: (ch, md) => firstHeading(md, ch),
});

// Referential integrity of authored content
const seen = new Set<string>();
for (const id of [...questions.map((q) => q.id), ...exercises.map((e) => e.id), ...flashcards.map((c) => c.id), ...lessons.map((l) => l.id)]) {
  if (seen.has(id)) problems.push(`duplicate id ${id}`);
  seen.add(id);
}
for (const e of exercises) for (const c of e.conceptIds) if (!conceptIds.has(c)) problems.push(`${e.id}: unknown concept ${c}`);
if (problems.length) {
  console.error("Content problems:\n - " + problems.join("\n - "));
  process.exit(1);
}

if (process.env.DRY === "1") {
  console.log(`dry run OK: ${courses.length} courses, ${lessons.length} lessons, ${concepts.length} concepts, ${questions.length} questions, ${exercises.length} exercises, ${flashcards.length} flashcards`);
  process.exit(0);
}
const { db, sqlite } = openDb();
function chunked<T>(rows: T[], insert: (c: T[]) => void, n = 200) {
  for (let i = 0; i < rows.length; i += n) insert(rows.slice(i, i + n));
}
db.transaction((tx) => {
  for (const tbl of [t.lessonConcepts, t.conceptEdges, t.flashcards, t.exercises, t.questions, t.concepts, t.lessons, t.courses]) tx.delete(tbl).run();
  sqlite.exec("DELETE FROM lesson_fts");
  chunked(courses, (c) => tx.insert(t.courses).values(c).run());
  chunked(lessons, (c) => tx.insert(t.lessons).values(c).run(), 50);
  chunked(concepts, (c) => tx.insert(t.concepts).values(c).run());
  chunked(edges, (c) => tx.insert(t.conceptEdges).values(c).run());
  chunked(lessonConcepts, (c) => tx.insert(t.lessonConcepts).values(c).run());
  chunked(questions, (c) => tx.insert(t.questions).values(c).run(), 100);
  chunked(exercises, (c) => tx.insert(t.exercises).values(c).run(), 20);
  chunked(flashcards, (c) => tx.insert(t.flashcards).values(c).run());
  const ins = sqlite.prepare("INSERT INTO lesson_fts (lesson_id, title, body) VALUES (?, ?, ?)");
  for (const l of lessons) ins.run(l.id, l.title, l.body);
});

const authored = lessons.filter((l) => l.kind === "authored").length;
console.log(
  `synced: ${courses.length} courses, ${lessons.length} lessons (${authored} authored, ${lessons.length - authored} source), ${concepts.length} concepts, ${edges.length} prerequisite edges, ${questions.length} questions, ${exercises.length} exercises, ${flashcards.length} flashcards`,
);
