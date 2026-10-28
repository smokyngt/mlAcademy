import { describe, expect, test } from "bun:test";
import { schedule, NEW_CARD } from "./srs";
import { updateMastery, EMPTY_MASTERY, isMastered, reviewPlan, type MasteryState } from "./mastery";

describe("srs", () => {
  const t0 = 1_000_000;
  test("good progression 1d -> 3d -> ease growth", () => {
    const a = schedule(NEW_CARD, 2, t0);
    expect(a.intervalDays).toBe(1);
    const b = schedule(a, 2, t0);
    expect(b.intervalDays).toBe(3);
    const c = schedule(b, 2, t0);
    expect(c.intervalDays).toBeCloseTo(7.5, 5);
  });
  test("lapse resets reps, lowers ease, floors at 1.3", () => {
    let s = { ...NEW_CARD, reps: 5, intervalDays: 30 };
    for (let i = 0; i < 10; i++) s = schedule(s, 0, t0);
    expect(s.reps).toBe(0);
    expect(s.ease).toBeCloseTo(1.3, 5);
    expect(s.lapses).toBe(10);
    expect(s.intervalDays).toBeLessThan(0.01);
  });
  test("easy grows faster than good", () => {
    const s = { ...NEW_CARD, reps: 3, intervalDays: 10 };
    expect(schedule(s, 3, t0).intervalDays).toBeGreaterThan(schedule(s, 2, t0).intervalDays);
  });
});

describe("mastery", () => {
  test("one correct answer is not mastery", () => {
    const m = updateMastery(EMPTY_MASTERY, "quiz", 1);
    expect(isMastered(m)).toBe(false);
    expect(m.score).toBeCloseTo(0.6, 5);
  });
  test("repeated success reaches mastery", () => {
    let m: MasteryState = EMPTY_MASTERY;
    for (let i = 0; i < 6; i++) m = updateMastery(m, "quiz", 1);
    expect(isMastered(m)).toBe(true);
  });
  test("failures accumulate and recover", () => {
    let m: MasteryState = EMPTY_MASTERY;
    m = updateMastery(m, "quiz", 0);
    m = updateMastery(m, "quiz", 0);
    expect(m.recentFailures).toBe(2);
    m = updateMastery(m, "quiz", 1);
    expect(m.recentFailures).toBe(1);
  });
  test("reviewPlan points at weak prerequisites, nearest first, skipping solid ones", () => {
    const prereqs = new Map([
      ["backprop", ["chain", "matmul"]],
      ["chain", ["deriv"]],
      ["matmul", []],
      ["deriv", []],
    ]);
    const solid: MasteryState = { score: 0.95, attempts: 5, recentFailures: 0 };
    const mastery = new Map<string, MasteryState>([
      ["backprop", { score: 0.2, attempts: 3, recentFailures: 3 }],
      ["matmul", solid],
      ["chain", { score: 0.3, attempts: 2, recentFailures: 1 }],
    ]);
    const plan = reviewPlan({ prereqs, mastery }, "backprop");
    expect(plan?.review).toEqual(["chain", "deriv"]); // deriv never attempted; matmul is solid
  });
  test("no plan when not struggling", () => {
    const mastery = new Map<string, MasteryState>([["x", { score: 0.7, attempts: 3, recentFailures: 0 }]]);
    expect(reviewPlan({ prereqs: new Map([["x", ["y"]]]), mastery }, "x")).toBeNull();
  });
});
