import { Hono } from "hono";
import { and, asc, desc, eq, sql } from "drizzle-orm";
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import type { Db, Sqlite } from "./db";
import { ACADEMY_ROOT } from "./db";
import * as t from "./db/schema";
import { EMPTY_MASTERY, cardGradeTarget, isMastered, reviewPlan, updateMastery, type Evidence, type MasteryState } from "./mastery";
import { NEW_CARD, schedule, type Grade } from "./srs";
import { runnerStatus, runInDocker, runNotebookInDocker } from "./runner/docker";
import { loadMetaOverlay, loadOverlays, type MetaOverlay, type Overlays } from "./content-i18n";
import { parseProject, taskGroups } from "../shared/notebook";
import type * as S from "../shared/types";

const DAY = 86_400_000;

export function createApp(db: Db, sqlite: Sqlite) {
  const app = new Hono();
  const now = () => Date.now();

  // lesson id -> folder (to find the optional figures.json next to lesson.md)
  let dirsCache: Map<string, string> | null = null;
  const lessonDir = (id: string) => {
    if (!dirsCache) {
      dirsCache = new Map();
      const root = path.join(ACADEMY_ROOT, "content", "authored");
      for (const c of fs.readdirSync(root)) {
        const cd = path.join(root, c);
        if (!fs.statSync(cd).isDirectory()) continue;
        for (const l of fs.readdirSync(cd)) {
          const f = path.join(cd, l, "lesson.md");
          if (fs.existsSync(f)) { const m = fs.readFileSync(f, "utf8").match(/^id:\s*(\S+)/m); if (m) dirsCache.set(m[1], path.join(cd, l)); }
        }
      }
    }
    return dirsCache.get(id);
  };
  const figuresFor = (id: string, lang: string): S.FigureData[] => {
    const dir = lessonDir(id);
    const f = dir && path.join(dir, "figures.json");
    if (!f || !fs.existsSync(f)) return [];
    try {
      const list = JSON.parse(fs.readFileSync(f, "utf8")) as { id: string; after: number; kind: "mermaid" | "plot" | "video"; caption: Record<string, string>; code?: Record<string, string>; spec?: unknown; src?: string }[];
      return list.map((x) => ({ id: x.id, after: x.after, kind: x.kind, caption: x.caption[lang] ?? x.caption.en, code: x.code ? (x.code[lang] ?? x.code.en) : undefined, spec: x.spec, src: x.src }));
    } catch { return []; }
  };

  // Translated content, loaded once per language; missing translations fall back to the English text.
  const overlayCache = new Map<string, Overlays>();
  const ov = (lang?: string): Overlays | null => {
    if (!lang || lang === "en") return null;
    if (!/^[a-z]{2}$/.test(lang)) return null;
    if (!overlayCache.has(lang)) overlayCache.set(lang, loadOverlays(lang));
    return overlayCache.get(lang)!;
  };

  const metaCache = new Map<string, MetaOverlay>();
  /** Translated catalog metadata (course/concept/capstone/roadmap/bibliography text); empty for English. */
  const meta = (lang?: string): MetaOverlay => {
    if (!lang || !/^[a-z]{2}$/.test(lang) || lang === "en") return loadMetaOverlay("en", "/nonexistent");
    if (!metaCache.has(lang)) metaCache.set(lang, loadMetaOverlay(lang));
    return metaCache.get(lang)!;
  };

  // Language of the request being served (handlers are synchronous, so a module-level variable is safe here).
  let reqLang: string | undefined;
  app.use("/api/*", async (c, next) => { reqLang = c.req.query("lang"); await next(); });
  const ctitle = (id: string) => meta(reqLang).concepts[id]?.title ?? conceptTitle.get(id) ?? id;
  const ltitle = (id: string, fallback: string) => ov(reqLang)?.lessons.get(id)?.title ?? fallback;
  const cotitle = (id: string, fallback: string) => meta(reqLang).courses[id]?.title ?? fallback;

  /* ---------------- helpers ---------------- */

  const conceptTitle = new Map<string, string>();
  const refreshConcepts = () => {
    conceptTitle.clear();
    for (const c of db.select().from(t.concepts).all()) conceptTitle.set(c.id, c.title);
  };
  refreshConcepts();

  function masteryMap(): Map<string, MasteryState> {
    const m = new Map<string, MasteryState>();
    for (const r of db.select().from(t.conceptMastery).all()) m.set(r.conceptId, { score: r.score, attempts: r.attempts, recentFailures: r.recentFailures });
    return m;
  }

  function prereqMap(): Map<string, string[]> {
    const m = new Map<string, string[]>();
    for (const e of db.select().from(t.conceptEdges).all()) m.set(e.conceptId, [...(m.get(e.conceptId) ?? []), e.prerequisiteId]);
    return m;
  }

  function recordEvidence(conceptIds: string[], kind: Evidence, target: number) {
    const all = masteryMap();
    for (const id of new Set(conceptIds)) {
      const next = updateMastery(all.get(id) ?? EMPTY_MASTERY, kind, target);
      db.insert(t.conceptMastery)
        .values({ conceptId: id, score: next.score, attempts: next.attempts, recentFailures: next.recentFailures, updatedAt: now() })
        .onConflictDoUpdate({ target: t.conceptMastery.conceptId, set: { score: next.score, attempts: next.attempts, recentFailures: next.recentFailures, updatedAt: now() } })
        .run();
    }
  }

  function teachingLesson(conceptId: string): string | null {
    const r = db.select({ id: t.lessonConcepts.lessonId }).from(t.lessonConcepts).where(and(eq(t.lessonConcepts.conceptId, conceptId), eq(t.lessonConcepts.role, "teaches"))).limit(1).get();
    return r?.id ?? null;
  }

  function recommendation(conceptId: string): S.Recommendation | null {
    const mastery = masteryMap();
    const plan = reviewPlan({ prereqs: prereqMap(), mastery }, conceptId);
    if (!plan) return null;
    return {
      conceptId,
      conceptTitle: ctitle(conceptId),
      reason: plan.reason,
      review: plan.review.slice(0, 5).map((id) => ({ conceptId: id, title: ctitle(id), score: mastery.get(id)?.score ?? null, lessonId: teachingLesson(id) })),
    };
  }

  const lessonStatus = (id: string): "new" | "started" | "completed" => {
    const r = db.select().from(t.lessonProgress).where(eq(t.lessonProgress.lessonId, id)).get();
    return !r ? "new" : r.status === "completed" ? "completed" : "started";
  };

  /* ---------------- catalog ---------------- */

  /** The single beginner-to-expert path (content/path.json), resolved against the lessons that really exist. */
  app.get("/api/path", (c) => {
    const spec = JSON.parse(fs.readFileSync(path.join(ACADEMY_ROOT, "content", "path.json"), "utf8")) as { parts: { id: "math" | "ml"; modules: { id: string; level: S.Level; title: string; items: string[] }[] }[] };
    const O = ov(c.req.query("lang"));
    const M = meta(c.req.query("lang"));
    const progress = new Map(db.select().from(t.lessonProgress).all().map((p) => [p.lessonId, p.status]));
    const courses = new Map(db.select().from(t.courses).all().map((k) => [k.id, k]));
    const all = new Map(db.select({ id: t.lessons.id, title: t.lessons.title, minutes: t.lessons.minutes, position: t.lessons.position, courseId: t.lessons.courseId }).from(t.lessons).all().map((l) => [l.id, l]));
    const stub = (id: string): S.LessonStub => {
      const l = all.get(id)!;
      return { id, title: O?.lessons.get(id)?.title ?? l.title, minutes: l.minutes, position: l.position, status: (progress.get(id) as S.LessonStub["status"]) ?? "new" };
    };
    const used = new Set<string>();
    const parts: S.PathPart[] = spec.parts.map((p) => ({
      id: p.id,
      modules: p.modules
        .map((m) => ({ id: m.id, level: m.level, title: M.modules[m.id] ?? m.title, lessons: m.items.filter((id) => all.has(id) && !used.has(id)).map((id) => (used.add(id), stub(id))) }))
        .filter((m) => m.lessons.length > 0),
    }));
    // ML modules show the maths they rest on. Derived, not hand-written: the concepts a module's lessons require,
    // mapped to the lessons of the Mathematics part that teach those concepts (transitively through math prerequisites).
    const teaches = new Map<string, string[]>();
    const requires = new Map<string, string[]>();
    for (const r of db.select().from(t.lessonConcepts).all()) {
      const m = r.role === "teaches" ? teaches : requires;
      m.set(r.lessonId, [...(m.get(r.lessonId) ?? []), r.conceptId]);
    }
    const mathTeacher = new Map<string, string>(); // concept id -> math lesson id
    const mathOrder: string[] = parts.filter((p) => p.id === "math").flatMap((p) => p.modules.flatMap((m) => m.lessons.map((l) => l.id)));
    for (const id of mathOrder) for (const c of teaches.get(id) ?? []) if (!mathTeacher.has(c)) mathTeacher.set(c, id);
    const prereqs = prereqMap();
    for (const p of parts.filter((x) => x.id === "ml")) {
      for (const m of p.modules) {
        const inModule = new Set(m.lessons.map((l) => l.id));
        const needed = new Set<string>();
        const seen = new Set<string>();
        const visit = (concept: string) => {
          if (seen.has(concept)) return;
          seen.add(concept);
          const lid = mathTeacher.get(concept);
          if (lid && !inModule.has(lid)) needed.add(lid);
          for (const q of prereqs.get(concept) ?? []) visit(q);
        };
        for (const l of m.lessons) for (const c of requires.get(l.id) ?? []) visit(c);
        const order = new Map(mathOrder.map((id, i) => [id, i]));
        m.mathNeeded = [...needed].sort((a, b) => order.get(a)! - order.get(b)!).map((id) => { const s = stub(id); return { id, title: s.title, status: s.status }; });
      }
    }

    // Safety net: a lesson that exists but is not placed in the path still shows up.
    for (const p of parts) {
      const extra = [...all.keys()].filter((id) => !used.has(id) && courses.get(all.get(id)!.courseId)?.part === p.id).sort();
      if (extra.length) p.modules.push({ id: `${p.id}-more`, level: "expert", title: M.modules["more"] ?? "More lessons (not yet placed in the path)", lessons: extra.map(stub) });
    }
    return c.json(parts);
  });

  app.get("/api/catalog", (c) => {
    const M = meta(c.req.query("lang"));
    const O = ov(c.req.query("lang"));
    const progress = new Map(db.select().from(t.lessonProgress).all().map((p) => [p.lessonId, p.status]));
    const lessons = db
      .select({ id: t.lessons.id, courseId: t.lessons.courseId, title: t.lessons.title, minutes: t.lessons.minutes, position: t.lessons.position })
      .from(t.lessons)
      .orderBy(asc(t.lessons.courseId), asc(t.lessons.position))
      .all();
    const byCourse = new Map<string, S.LessonStub[]>();
    for (const l of lessons)
      byCourse.set(l.courseId, [...(byCourse.get(l.courseId) ?? []), { id: l.id, title: O?.lessons.get(l.id)?.title ?? l.title, minutes: l.minutes, position: l.position, status: (progress.get(l.id) as S.LessonStub["status"]) ?? "new" }]);
    const out: S.CourseSummary[] = db
      .select()
      .from(t.courses)
      .orderBy(asc(t.courses.position))
      .all()
      .map((co) => ({ id: co.id, title: M.courses[co.id]?.title ?? co.title, summary: M.courses[co.id]?.summary ?? co.summary, stage: co.stage, kind: co.kind as "authored" | "source", part: co.part as "math" | "ml", lessons: byCourse.get(co.id) ?? [] }));
    return c.json(out);
  });

  app.get("/api/lessons/:id", (c) => {
    const id = c.req.param("id");
    const l = db.select().from(t.lessons).where(eq(t.lessons.id, id)).get();
    if (!l) return c.json({ error: "not found" }, 404);
    const course = db.select().from(t.courses).where(eq(t.courses.id, l.courseId)).get()!;
    const mastery = masteryMap();
    const refs = (role: string): S.ConceptRef[] =>
      db
        .select({ id: t.lessonConcepts.conceptId })
        .from(t.lessonConcepts)
        .where(and(eq(t.lessonConcepts.lessonId, id), eq(t.lessonConcepts.role, role)))
        .all()
        .map((r) => ({ id: r.id, title: ctitle(r.id), mastery: mastery.get(r.id)?.score ?? null }));
    const O = ov(c.req.query("lang"));
    const tl = O?.lessons.get(id);
    const qs: S.QuestionPublic[] = db
      .select()
      .from(t.questions)
      .where(eq(t.questions.lessonId, id))
      .all()
      .map((q) => { const x = O?.questions.get(q.id); return { id: q.id, conceptId: q.conceptId, kind: q.kind as "mcq" | "numeric", stage: q.stage, prompt: x?.prompt ?? q.prompt, options: x?.options ?? q.options }; });
    const states = new Map(db.select().from(t.exerciseState).all().map((s) => [s.exerciseId, s]));
    const exs: S.ExercisePublic[] = db
      .select()
      .from(t.exercises)
      .where(eq(t.exercises.lessonId, id))
      .all()
      .map((e) => {
        const s = states.get(e.id);
        const x = O?.exercises.get(e.id);
        return { id: e.id, title: x?.title ?? e.title, difficulty: e.difficulty, runtime: e.runtime as S.Runtime, prompt: x?.prompt ?? e.prompt, starter: e.starter, tests: e.tests, hintCount: e.hints.length, conceptIds: e.conceptIds, savedCode: s?.code ?? null, solved: s?.solved ?? false, hintsUsed: s?.hintsUsed ?? 0, solutionRevealed: s?.solutionRevealed ?? false };
      });
    const siblings = db.select({ id: t.lessons.id, title: t.lessons.title, position: t.lessons.position }).from(t.lessons).where(eq(t.lessons.courseId, l.courseId)).orderBy(asc(t.lessons.position)).all();
    const i = siblings.findIndex((s) => s.id === id);
    const out: S.LessonDetail = {
      id: l.id, courseId: l.courseId, courseTitle: cotitle(course.id, course.title), title: tl?.title ?? l.title, summary: tl?.summary ?? l.summary, kind: l.kind as "authored" | "source", minutes: l.minutes, body: tl?.body ?? l.body, translated: !!tl, figures: figuresFor(id, c.req.query("lang") ?? "en"), sources: l.sources,
      teaches: refs("teaches"), requires: refs("requires"), questions: qs, exercises: exs, status: lessonStatus(id),
      bookmarked: !!db.select().from(t.bookmarks).where(eq(t.bookmarks.lessonId, id)).get(),
      prev: i > 0 ? { id: siblings[i - 1].id, title: ltitle(siblings[i - 1].id, siblings[i - 1].title) } : null,
      next: i >= 0 && i < siblings.length - 1 ? { id: siblings[i + 1].id, title: ltitle(siblings[i + 1].id, siblings[i + 1].title) } : null,
    };
    return c.json(out);
  });

  app.post("/api/lessons/:id/visit", (c) => {
    const id = c.req.param("id");
    if (!db.select({ id: t.lessons.id }).from(t.lessons).where(eq(t.lessons.id, id)).get()) return c.json({ error: "not found" }, 404);
    db.insert(t.lessonProgress).values({ lessonId: id, status: "started", lastVisitedAt: now() }).onConflictDoUpdate({ target: t.lessonProgress.lessonId, set: { lastVisitedAt: now() } }).run();
    return c.json({ ok: true });
  });

  app.post("/api/lessons/:id/complete", async (c) => {
    const id = c.req.param("id");
    const { completed } = z.object({ completed: z.boolean() }).parse(await c.req.json());
    db.insert(t.lessonProgress)
      .values({ lessonId: id, status: completed ? "completed" : "started", lastVisitedAt: now(), completedAt: completed ? now() : null })
      .onConflictDoUpdate({ target: t.lessonProgress.lessonId, set: { status: completed ? "completed" : "started", completedAt: completed ? now() : null, lastVisitedAt: now() } })
      .run();
    return c.json({ ok: true });
  });

  /* ---------------- quizzes ---------------- */

  app.post("/api/questions/:id/answer", async (c) => {
    const q = db.select().from(t.questions).where(eq(t.questions.id, c.req.param("id"))).get();
    if (!q) return c.json({ error: "not found" }, 404);
    const { response } = z.object({ response: z.number() }).parse(await c.req.json());
    let correct: boolean;
    let answer: number;
    if (q.kind === "numeric") {
      const a = q.answer as { value: number; tol: number };
      correct = Math.abs(response - a.value) <= a.tol;
      answer = a.value;
    } else {
      answer = q.answer as number;
      correct = response === answer;
    }
    db.insert(t.quizAttempts).values({ questionId: q.id, conceptId: q.conceptId, correct, response: String(response), at: now() }).run();
    recordEvidence([q.conceptId], "quiz", correct ? 1 : 0);
    const out: S.AnswerResult = { correct, explanation: ov(c.req.query("lang"))?.questions.get(q.id)?.explanation ?? q.explanation, answer, mastery: masteryMap().get(q.conceptId)?.score ?? 0, recommendation: correct ? null : recommendation(q.conceptId) };
    return c.json(out);
  });

  /** Cumulative / module assessment: questions sampled across a course, weighted toward weak concepts. */
  app.get("/api/courses/:id/assessment", (c) => {
    const n = Math.min(30, Math.max(3, Number(c.req.query("n") ?? 10)));
    const qs = db
      .select({ q: t.questions })
      .from(t.questions)
      .innerJoin(t.lessons, eq(t.questions.lessonId, t.lessons.id))
      .where(eq(t.lessons.courseId, c.req.param("id")))
      .all()
      .map((r) => r.q);
    const mastery = masteryMap();
    const keyed = qs.map((q) => ({ q, k: Math.pow(Math.random(), 1 / (1 + 2 * (1 - (mastery.get(q.conceptId)?.score ?? 0)))) }));
    keyed.sort((a, b) => b.k - a.k);
    return c.json(keyed.slice(0, n).map(({ q }): S.QuestionPublic => { const x = ov(c.req.query("lang"))?.questions.get(q.id); return { id: q.id, conceptId: q.conceptId, kind: q.kind as "mcq" | "numeric", stage: "assessment", prompt: x?.prompt ?? q.prompt, options: x?.options ?? q.options }; }));
  });

  /* ---------------- exercises ---------------- */

  const getExercise = (id: string) => db.select().from(t.exercises).where(eq(t.exercises.id, id)).get();
  const getState = (id: string) => db.select().from(t.exerciseState).where(eq(t.exerciseState.exerciseId, id)).get();

  app.put("/api/exercises/:id/code", async (c) => {
    const id = c.req.param("id");
    if (!getExercise(id)) return c.json({ error: "not found" }, 404);
    const { code } = z.object({ code: z.string().max(200_000) }).parse(await c.req.json());
    db.insert(t.exerciseState).values({ exerciseId: id, code, updatedAt: now() }).onConflictDoUpdate({ target: t.exerciseState.exerciseId, set: { code, updatedAt: now() } }).run();
    return c.json({ ok: true });
  });

  app.post("/api/exercises/:id/hint", (c) => {
    const e = getExercise(c.req.param("id"));
    if (!e) return c.json({ error: "not found" }, 404);
    const used = getState(e.id)?.hintsUsed ?? 0;
    if (used >= e.hints.length) return c.json({ hints: e.hints });
    db.insert(t.exerciseState).values({ exerciseId: e.id, code: e.starter, hintsUsed: used + 1, updatedAt: now() }).onConflictDoUpdate({ target: t.exerciseState.exerciseId, set: { hintsUsed: used + 1 } }).run();
    return c.json({ hints: (ov(c.req.query("lang"))?.exercises.get(e.id)?.hints ?? e.hints).slice(0, used + 1) });
  });

  app.get("/api/exercises/:id/hints", (c) => {
    const e = getExercise(c.req.param("id"));
    if (!e) return c.json({ error: "not found" }, 404);
    return c.json({ hints: (ov(c.req.query("lang"))?.exercises.get(e.id)?.hints ?? e.hints).slice(0, getState(e.id)?.hintsUsed ?? 0) });
  });

  app.post("/api/exercises/:id/solution", (c) => {
    const e = getExercise(c.req.param("id"));
    if (!e) return c.json({ error: "not found" }, 404);
    db.insert(t.exerciseState).values({ exerciseId: e.id, code: e.starter, solutionRevealed: true, updatedAt: now() }).onConflictDoUpdate({ target: t.exerciseState.exerciseId, set: { solutionRevealed: true } }).run();
    return c.json({ solution: e.solution });
  });

  /** Record the outcome of a run. Failed runs are normal debugging and do NOT lower mastery; only the first solve counts. */
  function recordAttempt(e: typeof t.exercises.$inferSelect, code: string, passed: boolean, output: string) {
    const prior = getState(e.id);
    db.insert(t.exerciseAttempts).values({ exerciseId: e.id, passed, runtime: e.runtime, output: output.slice(0, 20_000), at: now() }).run();
    const firstSolve = passed && !prior?.solved;
    db.insert(t.exerciseState)
      .values({ exerciseId: e.id, code, solved: passed, updatedAt: now() })
      .onConflictDoUpdate({ target: t.exerciseState.exerciseId, set: { code, solved: passed || (prior?.solved ?? false), updatedAt: now() } })
      .run();
    // Solving after looking at the solution is weaker evidence than solving unaided.
    if (firstSolve) recordEvidence(e.conceptIds, "exercise", prior?.solutionRevealed ? 0.5 : 1);
    return { firstSolve };
  }

  const AttemptBody = z.object({ code: z.string().max(200_000), passed: z.boolean(), output: z.string().max(100_000) });

  /** For the Pyodide backend, execution happens in the browser's WASM sandbox and the client reports the outcome. */
  app.post("/api/exercises/:id/attempt", async (c) => {
    const e = getExercise(c.req.param("id"));
    if (!e) return c.json({ error: "not found" }, 404);
    if (e.runtime !== "pyodide") return c.json({ error: "this exercise runs on the server runner; use /run" }, 400);
    const b = AttemptBody.parse(await c.req.json());
    return c.json(recordAttempt(e, b.code, b.passed, b.output));
  });

  /** Server-side execution (Docker). Never runs user code in this process. */
  app.post("/api/exercises/:id/run", async (c) => {
    const e = getExercise(c.req.param("id"));
    if (!e) return c.json({ error: "not found" }, 404);
    const { code } = z.object({ code: z.string().max(200_000) }).parse(await c.req.json());
    const res = await runInDocker({ kind: "exercise", code, tests: e.tests });
    if ("unavailable" in res) return c.json({ unavailable: res.unavailable }, 503);
    const passed = res.phase === "ok" && res.results.length > 0 && res.results.every((r) => r.passed);
    const { firstSolve } = recordAttempt(e, code, passed, JSON.stringify(res));
    return c.json({ result: res, passed, firstSolve });
  });

  app.get("/api/runner/status", async (c) => c.json(await runnerStatus()));

  /* ---------------- flashcards / review ---------------- */

  app.get("/api/review/session", (c) => {
    const newLimit = Math.min(50, Number(c.req.query("new") ?? 10));
    const t0 = now();
    const startOfDay = new Date(); startOfDay.setHours(0, 0, 0, 0);
    const states = new Map(db.select().from(t.cardState).all().map((s) => [s.cardId, s]));
    const cards = db.select().from(t.flashcards).all();
    const newToday = db.select({ n: sql<number>`count(*)` }).from(t.reviewLog).where(sql`${t.reviewLog.at} >= ${startOfDay.getTime()} and ${t.reviewLog.id} in (select min(id) from review_log group by card_id)`).get()?.n ?? 0;
    const due = cards.filter((k) => states.has(k.id) && states.get(k.id)!.dueAt <= t0).sort((a, b) => states.get(a.id)!.dueAt - states.get(b.id)!.dueAt);
    const fresh = cards.filter((k) => !states.has(k.id)).slice(0, Math.max(0, newLimit - newToday));
    const O = ov(c.req.query("lang"));
    const mk = (k: typeof cards[number], isNew: boolean): S.Card => { const x = O?.cards.get(k.id); return { id: k.id, conceptId: k.conceptId, conceptTitle: ctitle(k.conceptId), lessonId: k.lessonId, front: x?.front ?? k.front, back: x?.back ?? k.back, isNew }; };
    return c.json({ cards: [...due.map((k) => mk(k, false)), ...fresh.map((k) => mk(k, true))], dueCount: due.length, newCount: fresh.length });
  });

  app.post("/api/review/:cardId", async (c) => {
    const card = db.select().from(t.flashcards).where(eq(t.flashcards.id, c.req.param("cardId"))).get();
    if (!card) return c.json({ error: "not found" }, 404);
    const { grade } = z.object({ grade: z.number().int().min(0).max(3) }).parse(await c.req.json());
    const prev = db.select().from(t.cardState).where(eq(t.cardState.cardId, card.id)).get();
    const nextS = schedule(prev ? { ease: prev.ease, intervalDays: prev.intervalDays, reps: prev.reps, lapses: prev.lapses } : NEW_CARD, grade as Grade, now());
    db.insert(t.cardState)
      .values({ cardId: card.id, ease: nextS.ease, intervalDays: nextS.intervalDays, reps: nextS.reps, lapses: nextS.lapses, dueAt: nextS.dueAt, lastReviewedAt: now() })
      .onConflictDoUpdate({ target: t.cardState.cardId, set: { ease: nextS.ease, intervalDays: nextS.intervalDays, reps: nextS.reps, lapses: nextS.lapses, dueAt: nextS.dueAt, lastReviewedAt: now() } })
      .run();
    db.insert(t.reviewLog).values({ cardId: card.id, conceptId: card.conceptId, grade, at: now() }).run();
    recordEvidence([card.conceptId], "card", cardGradeTarget(grade as Grade));
    return c.json({ dueAt: nextS.dueAt, intervalDays: nextS.intervalDays });
  });

  /* ---------------- concepts / mastery ---------------- */

  app.get("/api/concepts", (c) => {
    const M = meta(c.req.query("lang"));
    const mastery = masteryMap();
    const prereqs = prereqMap();
    const lessonsBy = new Map<string, string[]>();
    for (const r of db.select().from(t.lessonConcepts).where(eq(t.lessonConcepts.role, "teaches")).all()) lessonsBy.set(r.conceptId, [...(lessonsBy.get(r.conceptId) ?? []), r.lessonId]);
    const out: S.ConceptNode[] = db
      .select()
      .from(t.concepts)
      .all()
      .map((k) => ({ id: k.id, title: M.concepts[k.id]?.title ?? k.title, description: M.concepts[k.id]?.description ?? k.description, stage: k.stage, prerequisites: prereqs.get(k.id) ?? [], mastery: mastery.get(k.id)?.score ?? null, attempts: mastery.get(k.id)?.attempts ?? 0, lessonIds: lessonsBy.get(k.id) ?? [] }));
    return c.json(out);
  });

  /* ---------------- dashboard ---------------- */

  app.get("/api/dashboard", (c) => {
    const t0 = now();
    const lessons = db.select({ id: t.lessons.id, kind: t.lessons.kind, title: t.lessons.title, courseId: t.lessons.courseId }).from(t.lessons).all();
    const progress = db.select().from(t.lessonProgress).all();
    const done = new Set(progress.filter((p) => p.status === "completed").map((p) => p.lessonId));
    const mastery = masteryMap();
    const concepts = db.select().from(t.concepts).all();
    const authoredConcepts = concepts.filter((k) => !k.id.startsWith("src."));
    const exercises = db.select({ id: t.exercises.id }).from(t.exercises).all();
    const solved = db.select().from(t.exerciseState).where(eq(t.exerciseState.solved, true)).all().length;
    const states = db.select().from(t.cardState).all();
    const dueCards = states.filter((s) => s.dueAt <= t0).length;
    const totalCards = db.select({ n: sql<number>`count(*)` }).from(t.flashcards).get()?.n ?? 0;

    // activity days for streak
    const stamps: number[] = [
      ...db.select({ at: t.quizAttempts.at }).from(t.quizAttempts).all().map((r) => r.at),
      ...db.select({ at: t.exerciseAttempts.at }).from(t.exerciseAttempts).all().map((r) => r.at),
      ...db.select({ at: t.reviewLog.at }).from(t.reviewLog).all().map((r) => r.at),
      ...progress.map((p) => p.lastVisitedAt),
    ];
    const dayKey = (ms: number) => new Date(ms).toLocaleDateString("en-CA");
    const days = new Set(stamps.map(dayKey));
    let streak = 0;
    for (let d = days.has(dayKey(t0)) ? 0 : 1; days.has(dayKey(t0 - d * DAY)); d++) streak++;

    const stages = ["I", "II", "III", "IV", "V", "VI", "VII"];
    const stageMastery = stages.map((st) => {
      const cs = authoredConcepts.filter((k) => k.stage === st);
      const scores = cs.map((k) => mastery.get(k.id)).filter((m): m is MasteryState => !!m);
      return { stage: st, mastered: cs.filter((k) => { const m = mastery.get(k.id); return !!m && isMastered(m); }).length, total: cs.length, avgScore: scores.length ? scores.reduce((a, m) => a + m.score, 0) / scores.length : null };
    });

    const last = [...progress].filter((p) => p.status !== "completed").sort((a, b) => b.lastVisitedAt - a.lastVisitedAt)[0];
    let cont: S.Dashboard["continueLesson"] = null;
    const courseTitles = new Map(db.select().from(t.courses).all().map((k) => [k.id, cotitle(k.id, k.title)]));
    const lessonById = new Map(lessons.map((l) => [l.id, l]));
    const pick = last ? lessonById.get(last.lessonId) : lessons.filter((l) => l.kind === "authored").sort((a, b) => a.id.localeCompare(b.id)).find((l) => !done.has(l.id));
    if (pick) cont = { id: pick.id, title: ltitle(pick.id, pick.title), courseTitle: courseTitles.get(pick.courseId) ?? "" };

    const recs: S.Recommendation[] = [];
    for (const [cid, m] of mastery) {
      if (m.recentFailures >= 2 || (m.attempts >= 2 && m.score < 0.5)) {
        const r = recommendation(cid);
        if (r) recs.push(r);
      }
    }

    const recent: S.Dashboard["recent"] = [
      ...db.select().from(t.quizAttempts).orderBy(desc(t.quizAttempts.at)).limit(8).all().map((r) => ({ kind: r.correct ? "quiz correct" : "quiz missed", label: ctitle(r.conceptId), at: r.at })),
      ...db.select().from(t.exerciseAttempts).orderBy(desc(t.exerciseAttempts.at)).limit(8).all().map((r) => ({ kind: r.passed ? "exercise passed" : "exercise run", label: r.exerciseId, at: r.at })),
      ...db.select().from(t.reviewLog).orderBy(desc(t.reviewLog.at)).limit(8).all().map((r) => ({ kind: "card review", label: ctitle(r.conceptId), at: r.at })),
    ].sort((a, b) => b.at - a.at).slice(0, 10);

    const out: S.Dashboard = {
      lessonsCompleted: done.size, lessonsTotal: lessons.length,
      authoredCompleted: lessons.filter((l) => l.kind === "authored" && done.has(l.id)).length, authoredTotal: lessons.filter((l) => l.kind === "authored").length,
      dueCards, newCards: Math.max(0, totalCards - states.length), exercisesSolved: solved, exercisesTotal: exercises.length, streakDays: streak,
      masteredConcepts: authoredConcepts.filter((k) => { const m = mastery.get(k.id); return !!m && isMastered(m); }).length, conceptsTotal: authoredConcepts.length,
      stageMastery, continueLesson: cont, recommendations: recs.slice(0, 4), recent,
    };
    return c.json(out);
  });

  /* ---------------- profile: where I am + per-course reports ---------------- */

  app.get("/api/profile", (c) => {
    const t0 = now();
    const courses = db.select().from(t.courses).orderBy(asc(t.courses.position)).all();
    const lessons = db.select({ id: t.lessons.id, courseId: t.lessons.courseId, title: t.lessons.title, position: t.lessons.position, minutes: t.lessons.minutes }).from(t.lessons).orderBy(asc(t.lessons.courseId), asc(t.lessons.position)).all();
    const progress = new Map(db.select().from(t.lessonProgress).all().map((p) => [p.lessonId, p]));
    const mastery = masteryMap();
    const qs = new Map(db.select({ id: t.questions.id, lessonId: t.questions.lessonId, conceptId: t.questions.conceptId }).from(t.questions).all().map((q) => [q.id, q]));
    const attempts = db.select().from(t.quizAttempts).all();
    const exs = db.select({ id: t.exercises.id, lessonId: t.exercises.lessonId }).from(t.exercises).all();
    const solved = new Set(db.select().from(t.exerciseState).where(eq(t.exerciseState.solved, true)).all().map((e) => e.exerciseId));
    const teaches = new Map<string, string[]>();
    for (const r of db.select().from(t.lessonConcepts).where(eq(t.lessonConcepts.role, "teaches")).all()) teaches.set(r.lessonId, [...(teaches.get(r.lessonId) ?? []), r.conceptId]);

    const courseOf = new Map(lessons.map((l) => [l.id, l.courseId]));
    const reports: S.CourseReport[] = courses.map((co) => {
      const ls = lessons.filter((l) => l.courseId === co.id);
      const done = ls.filter((l) => progress.get(l.id)?.status === "completed").length;
      const started = ls.filter((l) => progress.has(l.id)).length;
      const att = attempts.filter((a) => qs.get(a.questionId) && courseOf.get(qs.get(a.questionId)!.lessonId) === co.id);
      const ex = exs.filter((e) => courseOf.get(e.lessonId) === co.id);
      const scores = new Set(ls.flatMap((l) => teaches.get(l.id) ?? []));
      const ms = [...scores].map((k) => mastery.get(k)).filter((m): m is MasteryState => !!m);
      const avg = ms.length ? ms.reduce((a, m) => a + m.score, 0) / ms.length : null;
      const status: S.CourseReport["status"] = done === ls.length && ls.length > 0 ? "done" : started === 0 ? "todo" : "inProgress";
      const next = ls.find((l) => progress.get(l.id)?.status !== "completed") ?? null;
      return {
        id: co.id, title: cotitle(co.id, co.title), part: co.part as "math" | "ml", stage: co.stage, status,
        lessonsDone: done, lessonsTotal: ls.length, minutesLeft: ls.filter((l) => progress.get(l.id)?.status !== "completed").reduce((a, l) => a + l.minutes, 0),
        quizAccuracy: att.length ? att.filter((a) => a.correct).length / att.length : null, quizAttempts: att.length,
        exercisesSolved: ex.filter((e) => solved.has(e.id)).length, exercisesTotal: ex.length,
        mastery: avg, nextLesson: next ? { id: next.id, title: ltitle(next.id, next.title) } : null,
      };
    });

    const activity: S.ProfileDay[] = [];
    const stamps = [
      ...attempts.map((a) => a.at),
      ...db.select({ at: t.exerciseAttempts.at }).from(t.exerciseAttempts).all().map((r) => r.at),
      ...db.select({ at: t.reviewLog.at }).from(t.reviewLog).all().map((r) => r.at),
      ...[...progress.values()].map((p) => p.lastVisitedAt),
    ];
    const dayKey = (ms: number) => new Date(ms).toLocaleDateString("en-CA");
    const counts = new Map<string, number>();
    for (const ms of stamps) counts.set(dayKey(ms), (counts.get(dayKey(ms)) ?? 0) + 1);
    for (let d = 29; d >= 0; d--) { const k = dayKey(t0 - d * DAY); activity.push({ day: k, n: counts.get(k) ?? 0 }); }

    const ranked = [...mastery].filter(([, m]) => m.attempts > 0).map(([id, m]) => ({ conceptId: id, title: ctitle(id), score: m.score }));
    const strengths = [...ranked].sort((a, b) => b.score - a.score).filter((r) => r.score >= 0.6).slice(0, 5);
    const weaknesses = [...ranked].sort((a, b) => a.score - b.score).filter((r) => r.score < 0.6).slice(0, 5);
    const totalLessons = lessons.length;
    const doneLessons = [...progress.values()].filter((p) => p.status === "completed").length;
    const first = [...progress.values()].map((p) => p.lastVisitedAt).sort((a, b) => a - b)[0] ?? null;
    const out: S.Profile = {
      since: first, lessonsDone: doneLessons, lessonsTotal: totalLessons,
      minutesDone: lessons.filter((l) => progress.get(l.id)?.status === "completed").reduce((a, l) => a + l.minutes, 0),
      minutesLeft: lessons.filter((l) => progress.get(l.id)?.status !== "completed").reduce((a, l) => a + l.minutes, 0),
      activeDays: counts.size, courses: reports, activity, strengths, weaknesses,
      current: reports.find((r) => r.status === "inProgress") ?? reports.find((r) => r.status === "todo") ?? null,
    };
    return c.json(out);
  });

  /* ---------------- search / bookmarks ---------------- */

  app.get("/api/search", (c) => {
    const raw = (c.req.query("q") ?? "").trim();
    if (raw.length < 2) return c.json([]);
    // Quote each token so user punctuation can never be parsed as FTS5 syntax.
    const match = raw.split(/\s+/).map((w) => `"${w.replace(/"/g, "")}"*`).join(" ");
    const rows = sqlite
      .query(
        `SELECT f.lesson_id AS lessonId, l.title AS title, c.title AS courseTitle,
                snippet(lesson_fts, 2, '[[', ']]', ' … ', 18) AS snippet
         FROM lesson_fts f JOIN lessons l ON l.id = f.lesson_id JOIN courses c ON c.id = l.course_id
         WHERE lesson_fts MATCH ? ORDER BY l.kind = 'authored' DESC, rank LIMIT 25`,
      )
      .all(match) as S.SearchHit[];
    return c.json(rows.map((r) => ({ ...r, title: ltitle(r.lessonId, r.title) })));
  });

  /** Fumadocs search-dialog contract: ?query= -> SortedResult[] ({id,url,type,content}). Backed by the same FTS5 index. */
  app.get("/api/docs-search", (c) => {
    const raw = (c.req.query("query") ?? "").trim();
    if (raw.length < 2) return c.json([]);
    const match = raw.split(/\s+/).map((w) => `"${w.replace(/"/g, "")}"*`).join(" ");
    const rows = sqlite
      .query(
        `SELECT f.lesson_id AS id, l.title AS title, snippet(lesson_fts, 2, '', '', ' … ', 18) AS snip
         FROM lesson_fts f JOIN lessons l ON l.id = f.lesson_id
         WHERE lesson_fts MATCH ? ORDER BY l.kind = 'authored' DESC, rank LIMIT 12`,
      )
      .all(match) as { id: string; title: string; snip: string }[];
    return c.json(rows.flatMap((r) => [
      { id: r.id, url: `/lesson/${r.id}`, type: "page", content: ltitle(r.id, r.title) },
      { id: `${r.id}#t`, url: `/lesson/${r.id}`, type: "text", content: r.snip.replace(/\s+/g, " ") },
    ]));
  });

  /* ---------------- course guides, roadmap, sources ---------------- */

  const readJson = (f: string) => JSON.parse(fs.readFileSync(path.join(ACADEMY_ROOT, "content", f), "utf8"));

  app.get("/api/courses/:id/guide", (c) => {
    const id = c.req.param("id");
    if (!/^[\w-]+$/.test(id)) return c.json({ error: "not found" }, 404);
    const dir = path.join(ACADEMY_ROOT, "content", "authored", id);
    const lang = c.req.query("lang");
    const tr = lang && /^[a-z]{2}$/.test(lang) && lang !== "en" ? path.join(dir, `guide.${lang}.md`) : null;
    const f = tr && fs.existsSync(tr) ? tr : path.join(dir, "guide.md");
    return fs.existsSync(f) ? c.json({ markdown: fs.readFileSync(f, "utf8") }) : c.json({ markdown: null });
  });

  /** Coverage of the requested curriculum, computed from what is really in the database. */
  app.get("/api/roadmap", (c) => {
    const road = readJson("roadmap.json") as { stages: { id: string; name: string; topics: { name: string; authored?: string[]; keywords?: string[]; partial?: boolean; note?: string }[] }[]; platform: unknown[] };
    const source = db.select({ id: t.lessons.id, title: t.lessons.title }).from(t.lessons).where(eq(t.lessons.kind, "source")).all();
    const authoredIds = new Set(db.select({ id: t.lessons.id }).from(t.lessons).where(eq(t.lessons.kind, "authored")).all().map((l) => l.id));
    const R = meta(c.req.query("lang")).roadmap;
    const stages = road.stages.map((st) => ({
      id: st.id,
      name: R.stages?.[st.id] ?? st.name,
      topics: st.topics.map((tp) => {
        const authored = (tp.authored ?? []).filter((id) => authoredIds.has(id));
        const kws = (tp.keywords ?? []).map((k) => k.toLowerCase());
        const matches = kws.length ? source.filter((l) => kws.some((k) => l.title.toLowerCase().includes(k))).slice(0, 4) : [];
        const status = authored.length ? (tp.partial ? "partial" : "authored") : matches.length ? "source" : "missing";
        return { name: R.topics?.[tp.name]?.name ?? tp.name, status, authored, source: matches, note: R.topics?.[tp.name]?.note ?? tp.note ?? null };
      }),
    }));
    const counts = { authored: 0, partial: 0, source: 0, missing: 0 };
    for (const st of stages) for (const tp of st.topics) counts[tp.status as keyof typeof counts]++;
    return c.json({ stages, counts, platform: (road.platform as { area: string; item: string }[]).map((p, i) => ({ ...p, area: R.platform?.[i]?.area ?? p.area, item: R.platform?.[i]?.item ?? p.item })) });
  });

  /** Sources: what each authored lesson cites (with a file-exists check), plus the bibliography and the provenance statement. */
  app.get("/api/sources", (c) => {
    const repo = path.resolve(ACADEMY_ROOT, "..");
    const bib = readJson("bibliography.json");
    const B = meta(c.req.query("lang")).bibliography;
    const O = ov(c.req.query("lang"));
    const lessons = db.select().from(t.lessons).where(eq(t.lessons.kind, "authored")).orderBy(asc(t.lessons.id)).all();
    const check = (s: string) => {
      const m = s.match(/^((?:math|ai-engineering-from-scratch|LLMs-from-scratch|reasoning-from-scratch)\/[^()]*?)\s*(\(|$)/);
      return m ? { exists: fs.existsSync(path.join(repo, m[1].trim())) } : { exists: null };
    };
    return c.json({
      provenance: B.provenance ?? bib.provenance,
      library: bib.library.map((l: { path: string; used: string }) => ({ ...l, used: B.library?.[l.path]?.used ?? l.used, exists: fs.existsSync(path.join(repo, l.path)) })),
      external: (bib.external as { ref: string; for: string }[]).map((x, i) => ({ ...x, for: B.external?.[i]?.for ?? x.for })),
      lessons: lessons.map((l) => ({ id: l.id, title: O?.lessons.get(l.id)?.title ?? l.title, sources: l.sources.map((s) => ({ text: s, read: /\bREAD\b/.test(s), ...check(s) })) })),
    });
  });

  app.get("/api/bookmarks", (c) => {
    const rows = db.select({ lessonId: t.bookmarks.lessonId, note: t.bookmarks.note, createdAt: t.bookmarks.createdAt, title: t.lessons.title }).from(t.bookmarks).innerJoin(t.lessons, eq(t.lessons.id, t.bookmarks.lessonId)).orderBy(desc(t.bookmarks.createdAt)).all();
    return c.json(rows.map((r) => ({ ...r, title: ltitle(r.lessonId, r.title) })));
  });
  app.put("/api/bookmarks/:lessonId", async (c) => {
    const { note } = z.object({ note: z.string().max(5000).default("") }).parse(await c.req.json().catch(() => ({})));
    db.insert(t.bookmarks).values({ lessonId: c.req.param("lessonId"), note, createdAt: now() }).onConflictDoUpdate({ target: t.bookmarks.lessonId, set: { note } }).run();
    return c.json({ ok: true });
  });
  app.delete("/api/bookmarks/:lessonId", (c) => {
    db.delete(t.bookmarks).where(eq(t.bookmarks.lessonId, c.req.param("lessonId"))).run();
    return c.json({ ok: true });
  });

  /* ---------------- notebooks (lab) ---------------- */

  const Cells = z.array(z.object({ id: z.string(), source: z.string().max(100_000) })).max(200);
app.get("/api/notebooks", (c) => c.json(db.select({ id: t.notebooks.id, title: t.notebooks.title, updatedAt: t.notebooks.updatedAt }).from(t.notebooks).orderBy(desc(t.notebooks.updatedAt)).all().filter((n) => !n.id.startsWith("project.") && !n.id.startsWith("inline."))));

  /* ---------------- projects: long guided notebooks (content/projects) ---------------- */

  /**
   * Projects, optionally translated: `<name>.project.<lang>.py` has the same cells in the same order. Only prose cells
   * (md, hint, report), task titles and the meta text are taken from it; code, checks and solutions always stay the English file's.
   */
  const projectFiles = (lang?: string) => {
    const dir = path.join(ACADEMY_ROOT, "content", "projects");
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir).filter((f) => f.endsWith(".project.py")).sort().flatMap((f) => {
      try {
        const base = parseProject(fs.readFileSync(path.join(dir, f), "utf8"));
        const tf = lang && /^[a-z]{2}$/.test(lang) && lang !== "en" ? path.join(dir, f.replace(/\.project\.py$/, `.project.${lang}.py`)) : null;
        if (!tf || !fs.existsSync(tf)) return [base];
        const tr = parseProject(fs.readFileSync(tf, "utf8"));
        const PROSE = new Set(["md", "hint", "report"]);
        const same = tr.cells.length === base.cells.length;
        const cells = base.cells.map((cell, i) => {
          const x = tr.cells[i];
          if (!same || x.kind !== cell.kind) return cell;
          if (PROSE.has(cell.kind)) return { ...cell, title: x.title, body: x.body };
          return cell.kind === "task" ? { ...cell, title: x.title } : cell; // a task's body is code the learner edits: only its title is translated
        });
        const pick = <T,>(a: T[] | undefined, b: T[] | undefined) => (a && b && a.length === b.length ? a : b);
        return [{ meta: { ...base.meta, title: tr.meta.title, summary: tr.meta.summary, outcomes: pick(tr.meta.outcomes, base.meta.outcomes), stages: pick(tr.meta.stages, base.meta.stages) }, cells }];
      } catch { return []; }
    });
  };
  const saved = (id: string) => db.select().from(t.notebooks).where(eq(t.notebooks.id, "project." + id)).get()?.cells ?? [];

  app.get("/api/projects", (c) =>
    c.json(projectFiles(c.req.query("lang")).map(({ meta, cells }) => {
      const tasks = taskGroups(cells);
      const done = new Set(saved(meta.id).filter((x) => x.id.startsWith("done-")).map((x) => x.id));
      return { ...meta, tasks: tasks.length, done: tasks.filter((g) => done.has("done-" + g.task)).length };
    })),
  );

  app.get("/api/projects/:id", (c) => {
    const p = projectFiles(c.req.query("lang")).find((x) => x.meta.id === c.req.param("id"));
    if (!p) return c.json({ error: "not found" }, 404);
    const groups = taskGroups(p.cells);
    const hidden = new Set(groups.map((g) => g.solution).filter((x): x is number => x !== null));
    // Solutions never leave the server until the learner asks (POST .../solution/:index).
    const cells = p.cells.map((cell, index) => ({ index, kind: cell.kind, title: cell.title, body: hidden.has(index) ? "" : cell.body }));
    return c.json({ meta: p.meta, cells, groups, saved: saved(p.meta.id) });
  });

  app.put("/api/projects/:id/state", async (c) => {
    const body = z.object({ cells: z.array(z.object({ id: z.string().max(40), source: z.string().max(100_000) })).max(400) }).parse(await c.req.json());
    const id = "project." + c.req.param("id");
    db.insert(t.notebooks).values({ id, title: id, cells: body.cells, updatedAt: now() }).onConflictDoUpdate({ target: t.notebooks.id, set: { cells: body.cells, updatedAt: now() } }).run();
    return c.json({ ok: true });
  });

  app.post("/api/projects/:id/solution/:index", (c) => {
    const p = projectFiles().find((x) => x.meta.id === c.req.param("id"));
    const cell = p?.cells[Number(c.req.param("index"))];
    return cell && cell.kind === "solution" ? c.json({ solution: cell.body }) : c.json({ error: "not found" }, 404);
  });
  app.get("/api/notebooks/:id", (c) => {
    const n = db.select().from(t.notebooks).where(eq(t.notebooks.id, c.req.param("id"))).get();
    return n ? c.json(n) : c.json({ error: "not found" }, 404);
  });
  app.put("/api/notebooks/:id", async (c) => {
    const b = z.object({ title: z.string().min(1).max(200), cells: Cells }).parse(await c.req.json());
    db.insert(t.notebooks).values({ id: c.req.param("id"), title: b.title, cells: b.cells, updatedAt: now() }).onConflictDoUpdate({ target: t.notebooks.id, set: { title: b.title, cells: b.cells, updatedAt: now() } }).run();
    return c.json({ ok: true });
  });
  app.delete("/api/notebooks/:id", (c) => {
    db.delete(t.notebooks).where(eq(t.notebooks.id, c.req.param("id"))).run();
    return c.json({ ok: true });
  });
  /** Server-side notebook execution: cells are replayed in order in one sandboxed container. */
  app.post("/api/notebooks/run", async (c) => {
    const { cells } = z.object({ cells: z.array(z.string().max(100_000)).max(100) }).parse(await c.req.json());
    const res = await runNotebookInDocker(cells);
    return "unavailable" in res ? c.json({ unavailable: res.unavailable }, 503) : c.json(res);
  });

  /* ---------------- capstones (written evidence) ---------------- */

  const capstonesFile = path.join(ACADEMY_ROOT, "content", "capstones.json");
  const capstones: { id: string; title: string; brief: string; criteria: { id: string; text: string; evidence: string }[] }[] = fs.existsSync(capstonesFile) ? JSON.parse(fs.readFileSync(capstonesFile, "utf8")) : [];

  app.get("/api/capstones", (c) => {
    const rows = db.select().from(t.evidence).orderBy(desc(t.evidence.at)).all();
    const K = meta(c.req.query("lang")).capstones;
    return c.json(
      capstones.map((cap) => ({
        ...cap,
        title: K[cap.id]?.title ?? cap.title,
        brief: K[cap.id]?.brief ?? cap.brief,
        criteria: cap.criteria.map((cr) => {
          const latest = rows.find((r) => r.capstoneId === cap.id && r.criterion === cr.id);
          const x = K[cap.id]?.criteria?.[cr.id];
          return { ...cr, text: x?.text ?? cr.text, evidence: x?.evidence ?? cr.evidence, latest: latest ? { text: latest.text, selfRating: latest.selfRating, at: latest.at } : null };
        }),
      })),
    );
  });
  app.post("/api/capstones/:id/evidence", async (c) => {
    const cap = capstones.find((k) => k.id === c.req.param("id"));
    if (!cap) return c.json({ error: "not found" }, 404);
    const b = z.object({ criterion: z.string(), text: z.string().min(80, "Write at least 80 characters of evidence").max(20_000), selfRating: z.number().int().min(0).max(3) }).parse(await c.req.json());
    if (!cap.criteria.some((k) => k.id === b.criterion)) return c.json({ error: "unknown criterion" }, 400);
    db.insert(t.evidence).values({ capstoneId: cap.id, criterion: b.criterion, text: b.text, selfRating: b.selfRating, at: now() }).run();
    return c.json({ ok: true });
  });

  app.onError((err, c) => {
    if (err instanceof z.ZodError) return c.json({ error: "invalid request", issues: err.issues.map((i) => i.message) }, 400);
    console.error(err);
    return c.json({ error: "internal error" }, 500);
  });

  return app;
}

