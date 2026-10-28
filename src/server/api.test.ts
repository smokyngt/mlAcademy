import { beforeAll, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Build a throw-away database from the real content, then exercise the real routes against it.
const dbFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "academy-test-")), "test.db");
process.env.ACADEMY_DB = dbFile;
const { ACADEMY_ROOT, openDb } = await import("./db");
const { createApp } = await import("./app");

let app: ReturnType<typeof createApp>;
const call = async (method: string, url: string, body?: unknown) => {
  const res = await app.request(url, { method, headers: { "content-type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, json: (await res.json()) as any };
};

beforeAll(() => {
  const r = spawnSync("bun", [path.join(ACADEMY_ROOT, "scripts", "build-catalog.ts")], { env: { ...process.env, ACADEMY_DB: dbFile }, encoding: "utf8" });
  expect(r.status).toBe(0);
  const { db, sqlite } = openDb(dbFile);
  app = createApp(db, sqlite);
}, 60_000); // building the catalog takes longer when the other test files run alongside

describe("catalog and lessons", () => {
  test("math and ML are separate parts; the shared foundation lessons exist with stable ids", async () => {
    const { json } = await call("GET", "/api/catalog");
    expect(json.every((c: any) => c.kind === "authored")).toBe(true); // library files are used to write lessons, not imported
    const byId = new Map(json.map((c: any) => [c.id, c]));
    expect((byId.get("math") as any).part).toBe("math");
    expect((byId.get("ml") as any).part).toBe("ml");
    const ids = (id: string) => (byId.get(id) as any).lessons.map((l: any) => l.id);
    expect(ids("math")).toEqual(expect.arrayContaining(["f2n.01", "f2n.02", "f2n.03", "f2n.04", "f2n.05", "f2n.06"]));
    expect(ids("ml")).toEqual(expect.arrayContaining(["f2n.07", "f2n.08", "f2n.09", "f2n.10", "f2n.11", "f2n.12"]));
  });

  test("the learning path lists every lesson exactly once, beginner modules first", async () => {
    const { json: parts } = await call("GET", "/api/path");
    expect(parts.map((p: any) => p.id)).toEqual(["math", "ml"]);
    const all: string[] = parts.flatMap((p: any) => p.modules.flatMap((m: any) => m.lessons.map((l: any) => l.id)));
    expect(new Set(all).size).toBe(all.length); // no duplicates
    const catalog = (await call("GET", "/api/catalog")).json;
    const lessons: string[] = catalog.flatMap((c: any) => c.lessons.map((l: any) => l.id));
    expect(all.sort()).toEqual(lessons.sort()); // none hidden, none invented
    expect(parts[0].modules[0].level).toBe("beginner");
    const order = ["beginner", "intermediate", "advanced", "expert"];
    for (const p of parts) {
      const lv = p.modules.filter((m: any) => !m.id.endsWith("-more")).map((m: any) => order.indexOf(m.level));
      expect([...lv].sort((x, y) => x - y)).toEqual(lv); // levels never go backwards inside a part
    }
  });

  test("lesson detail hides answers and solutions", async () => {
    const { json } = await call("GET", "/api/lessons/f2n.10");
    expect(json.questions.length).toBe(6);
    expect(json.questions[0].answer).toBeUndefined();
    expect(json.exercises[0].solution).toBeUndefined();
    expect(json.exercises[0].tests).toContain("check(");
    expect(json.requires.length).toBeGreaterThan(0);
  });
});

describe("quiz grading and mastery", () => {
  test("numeric answers use tolerance", async () => {
    expect((await call("POST", "/api/questions/f2n.01.q2/answer", { response: 15 })).json.correct).toBe(true);
    expect((await call("POST", "/api/questions/f2n.01.q2/answer", { response: 14 })).json.correct).toBe(false);
  });

  test("repeated misses recommend reviewing prerequisites", async () => {
    await call("POST", "/api/questions/f2n.10.q1/answer", { response: 0 });
    const second = await call("POST", "/api/questions/f2n.10.q1/answer", { response: 0 });
    expect(second.json.correct).toBe(false);
    expect(second.json.answer).toBe(1);
    expect(second.json.recommendation.conceptId).toBe("nn.backprop");
    const ids = second.json.recommendation.review.map((r: any) => r.conceptId);
    expect(ids).toContain("calc.jacobian");
  });

  test("a correct answer does not create a recommendation", async () => {
    const r = await call("POST", "/api/questions/f2n.03.q1/answer", { response: 0 });
    expect(r.json.correct).toBe(true);
    expect(r.json.recommendation).toBeNull();
  });
});

describe("exercises", () => {
  test("pyodide attempts are recorded; only the first solve moves mastery", async () => {
    const before = (await call("GET", "/api/concepts")).json.find((c: any) => c.id === "la.dot").attempts;
    const a = await call("POST", "/api/exercises/f2n.02.ex1/attempt", { code: "x=1", passed: false, output: "fail" });
    expect(a.json.firstSolve).toBe(false);
    const b = await call("POST", "/api/exercises/f2n.02.ex1/attempt", { code: "x=2", passed: true, output: "ok" });
    expect(b.json.firstSolve).toBe(true);
    const c2 = await call("POST", "/api/exercises/f2n.02.ex1/attempt", { code: "x=2", passed: true, output: "ok" });
    expect(c2.json.firstSolve).toBe(false);
    const after = (await call("GET", "/api/concepts")).json.find((c: any) => c.id === "la.dot").attempts;
    expect(after - before).toBe(1); // three attempts, but only the first solve is evidence
  });

  test("hints are revealed one at a time and the solution is gated", async () => {
    expect((await call("POST", "/api/exercises/f2n.01.ex1/hint")).json.hints.length).toBe(1);
    expect((await call("POST", "/api/exercises/f2n.01.ex1/hint")).json.hints.length).toBe(2);
    const s = await call("POST", "/api/exercises/f2n.01.ex1/solution");
    expect(s.json.solution).toContain("log_sum_exp");
  });

  test("docker exercises refuse the browser attempt endpoint and the runner reports honestly when unavailable", async () => {
    expect((await call("POST", "/api/exercises/f2n.12.ex2/attempt", { code: "", passed: true, output: "" })).status).toBe(400);
    const st = (await call("GET", "/api/runner/status")).json;
    expect(typeof st.available).toBe("boolean");
    if (!st.available) {
      const run = await call("POST", "/api/exercises/f2n.12.ex2/run", { code: "print(1)" });
      expect(run.status).toBe(503);
      expect(run.json.unavailable).toBeTruthy();
    }
  });
});

describe("flashcards", () => {
  test("session -> grade -> rescheduled", async () => {
    const s = (await call("GET", "/api/review/session?new=5")).json;
    expect(s.cards.length).toBe(5);
    const id = s.cards[0].id;
    const r = await call("POST", `/api/review/${id}`, { grade: 2 });
    expect(r.json.intervalDays).toBe(1);
    const s2 = (await call("GET", "/api/review/session?new=5")).json;
    expect(s2.cards.some((c: any) => c.id === id)).toBe(false); // not due for a day
    const again = await call("POST", `/api/review/${s2.cards[0].id}`, { grade: 0 });
    expect(again.json.intervalDays).toBeLessThan(0.01);
  });
});

describe("search, bookmarks, notebooks, dashboard", () => {
  test("full-text search finds authored lessons first", async () => {
    const { json } = await call("GET", "/api/search?q=backpropagation");
    expect(json.length).toBeGreaterThan(0);
    expect(json[0].lessonId.startsWith("f2n.")).toBe(true);
    expect((await call("GET", "/api/search?q=%22%20OR%20)")).status).toBe(200); // hostile FTS syntax is neutralised
  });

  test("bookmarks round trip", async () => {
    await call("PUT", "/api/bookmarks/f2n.06", { note: "revisit shapes" });
    expect((await call("GET", "/api/bookmarks")).json[0].note).toBe("revisit shapes");
    expect((await call("GET", "/api/lessons/f2n.06")).json.bookmarked).toBe(true);
    await call("DELETE", "/api/bookmarks/f2n.06");
    expect((await call("GET", "/api/bookmarks")).json.length).toBe(0);
  });

  test("notebooks persist", async () => {
    await call("PUT", "/api/notebooks/nb1", { title: "scratch", cells: [{ id: "a", source: "1+1" }] });
    expect((await call("GET", "/api/notebooks/nb1")).json.cells[0].source).toBe("1+1");
  });

  test("dashboard reflects real activity only", async () => {
    await call("POST", "/api/lessons/f2n.01/complete", { completed: true });
    const d = (await call("GET", "/api/dashboard")).json;
    expect(d.authoredCompleted).toBe(1);
    expect(d.exercisesSolved).toBe(1);
    expect(d.streakDays).toBe(1);
    expect(d.recommendations.some((r: any) => r.conceptId === "nn.backprop")).toBe(true);
  });

  test("assessment sampling returns questions without answers", async () => {
    const { json } = await call("GET", "/api/courses/ml/assessment?n=6");
    expect(json.length).toBe(6);
    expect(json.every((q: any) => q.answer === undefined && q.stage === "assessment")).toBe(true);
  });
});

describe("fumadocs search contract", () => {
  test("returns SortedResult entries (page + text) for a query", async () => {
    const { json } = await call("GET", "/api/docs-search?query=backpropagation");
    expect(json.length).toBeGreaterThan(1);
    expect(json[0]).toMatchObject({ type: "page" });
    expect(json[0].url.startsWith("/lesson/")).toBe(true);
    expect(typeof json[1].content).toBe("string");
  });
});
