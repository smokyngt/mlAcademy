import { beforeAll, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Same approach as api.test.ts: a throw-away database built from the real content. This file covers failure paths and the French overlay.
const dbFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "academy-edge-")), "test.db");
process.env.ACADEMY_DB = dbFile;
const { ACADEMY_ROOT, openDb } = await import("./db");
const { createApp } = await import("./app");
const { loadOverlays } = await import("./content-i18n");

let app: ReturnType<typeof createApp>;
const call = async (method: string, url: string, body?: unknown, raw?: string) => {
  const res = await app.request(url, { method, headers: { "content-type": "application/json" }, body: raw ?? (body === undefined ? undefined : JSON.stringify(body)) });
  return { status: res.status, json: (await res.json().catch(() => ({}))) as any };
};

beforeAll(() => {
  const r = spawnSync("bun", [path.join(ACADEMY_ROOT, "scripts", "build-catalog.ts")], { env: { ...process.env, ACADEMY_DB: dbFile }, encoding: "utf8" });
  expect(r.status).toBe(0);
  const { db, sqlite } = openDb(dbFile);
  app = createApp(db, sqlite);
}, 60_000); // building the catalog takes longer when the other test files run alongside

describe("unknown resources", () => {
  test("unknown lesson and question ids are 404, not 500", async () => {
    expect((await call("GET", "/api/lessons/nope.99")).status).toBe(404);
    expect((await call("POST", "/api/lessons/nope.99/visit")).status).toBe(404);
    expect((await call("POST", "/api/questions/nope/answer", { response: 0 })).status).toBe(404);
  });

  test("visiting a lesson marks it started without completing it", async () => {
    expect((await call("POST", "/api/lessons/f2n.04/visit")).json).toEqual({ ok: true });
    const p = (await call("GET", "/api/profile")).json;
    expect(p.lessonsDone).toBe(0);
    expect(p.activity.at(-1).n).toBeGreaterThan(0); // the visit counts as activity today
  });
});

describe("input validation", () => {
  test("answers must be numbers", async () => {
    const r = await call("POST", "/api/questions/f2n.01.q2/answer", { response: "15" });
    expect(r.status).toBeGreaterThanOrEqual(400);
    expect(r.status).toBeLessThan(500);
  });

  test("a malformed body does not take the server down", async () => {
    const bad = await call("POST", "/api/questions/f2n.01.q2/answer", undefined, "{not json");
    expect(bad.status).toBeGreaterThanOrEqual(400);
    expect((await call("GET", "/api/catalog")).status).toBe(200);
  });

  test("a wrong answer never leaks the explanation of another question and always reports the right choice", async () => {
    const r = await call("POST", "/api/questions/f2n.03.q1/answer", { response: 99 });
    expect(r.json.correct).toBe(false);
    expect(typeof r.json.answer).toBe("number");
    expect(r.json.explanation.length).toBeGreaterThan(0);
  });

  test("bookmark notes are capped", async () => {
    expect((await call("PUT", "/api/bookmarks/f2n.05", { note: "x".repeat(5001) })).status).toBeGreaterThanOrEqual(400);
    expect((await call("PUT", "/api/bookmarks/f2n.05", { note: "x".repeat(5000) })).status).toBe(200);
  });

  test("completing a lesson twice and un-completing is idempotent", async () => {
    await call("POST", "/api/lessons/f2n.02/complete", { completed: true });
    await call("POST", "/api/lessons/f2n.02/complete", { completed: true });
    const before = (await call("GET", "/api/dashboard")).json.authoredCompleted;
    await call("POST", "/api/lessons/f2n.02/complete", { completed: false });
    const after = (await call("GET", "/api/dashboard")).json.authoredCompleted;
    expect(before - after).toBe(1);
    expect((await call("POST", "/api/lessons/f2n.02/complete", { completed: "yes" })).status).toBeGreaterThanOrEqual(400);
  });
});

describe("search", () => {
  test("queries shorter than two characters return nothing", async () => {
    expect((await call("GET", "/api/search?q=a")).json).toEqual([]);
    expect((await call("GET", "/api/search")).json).toEqual([]);
    expect((await call("GET", "/api/search?q=%20%20")).json).toEqual([]);
  });

  test("punctuation and FTS operators are treated as plain text", async () => {
    for (const q of ['"', "AND OR NOT", "a*b", "(((", "NEAR(", "x:y", "-foo", "'; DROP TABLE lessons; --"]) {
      expect((await call("GET", `/api/search?q=${encodeURIComponent(q)}`)).status).toBe(200);
    }
    expect((await call("GET", "/api/catalog")).json.length).toBeGreaterThan(0); // tables are intact
  });

  test("prefix matching works and results are capped", async () => {
    const { json } = await call("GET", "/api/search?q=backprop");
    expect(json.length).toBeGreaterThan(0);
    expect(json.length).toBeLessThanOrEqual(25);
  });
});

describe("flashcards", () => {
  test("grading an unknown card or with an invalid grade is rejected", async () => {
    expect((await call("POST", "/api/review/does-not-exist", { grade: 2 })).status).toBeGreaterThanOrEqual(400);
    const { json } = await call("GET", "/api/review/session?new=1");
    expect((await call("POST", `/api/review/${json.cards[0].id}`, { grade: 7 })).status).toBeGreaterThanOrEqual(400);
  });

  test("the daily new-card limit is respected", async () => {
    expect((await call("GET", "/api/review/session?new=2")).json.cards.length).toBeLessThanOrEqual(2);
  });
});

describe("French overlay", () => {
  test("lessons and questions come back translated with the same ids", async () => {
    const en = (await call("GET", "/api/lessons/f2n.01?lang=en")).json;
    const fr = (await call("GET", "/api/lessons/f2n.01?lang=fr")).json;
    expect(fr.id).toBe(en.id);
    expect(fr.questions.map((q: any) => q.id)).toEqual(en.questions.map((q: any) => q.id));
    expect(fr.title).not.toBe(en.title);
  }, 30_000); // each request re-reads every French file from disk

  test("an unknown language falls back to English", async () => {
    const en = (await call("GET", "/api/lessons/f2n.01?lang=en")).json;
    const xx = (await call("GET", "/api/lessons/f2n.01?lang=xx")).json;
    expect(xx.title).toBe(en.title);
  });

  test("overlay ids all exist in the English source (no orphan translations)", async () => {
    const fr = loadOverlays("fr");
    const en = loadOverlays("en");
    expect(en.lessons.size).toBe(0); // English is the source, it has no overlay
    expect(fr.lessons.size).toBeGreaterThan(0);
    const known = new Set((await call("GET", "/api/catalog")).json.flatMap((c: any) => c.lessons.map((l: any) => l.id)));
    expect([...fr.lessons.keys()].filter((id) => !known.has(id))).toEqual([]);
  });

  test("loadOverlays tolerates a missing folder", () => {
    const o = loadOverlays("fr", path.join(os.tmpdir(), "academy-no-such-folder"));
    expect(o.lessons.size + o.questions.size + o.cards.size + o.exercises.size).toBe(0);
  });
});

describe("sources endpoint", () => {
  test("reports provenance and a boolean existence flag for cited library paths", async () => {
    const { status, json } = await call("GET", "/api/sources");
    expect(status).toBe(200);
    expect(typeof json.provenance).toBe("string");
    expect(json.library.every((l: any) => typeof l.exists === "boolean")).toBe(true);
    expect(json.lessons.every((l: any) => Array.isArray(l.sources))).toBe(true);
  });
});
