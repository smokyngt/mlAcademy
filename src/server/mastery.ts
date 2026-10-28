/**
 * Concept mastery model.
 *
 * score in [0,1] is an exponential moving average of evidence. Evidence sources have different weights:
 * an exercise pass is stronger evidence than one quiz answer, which is stronger than one flashcard review.
 * A concept counts as mastered only with score >= 0.8 AND at least 3 evidence items, so one lucky answer never counts.
 */

export type Evidence = "quiz" | "exercise" | "card";

const ALPHA: Record<Evidence, number> = { quiz: 0.3, exercise: 0.5, card: 0.15 };

export const MASTERED_SCORE = 0.8;
export const MASTERED_MIN_ATTEMPTS = 3;
export const WEAK_SCORE = 0.5;

export interface MasteryState {
  score: number;
  attempts: number;
  recentFailures: number;
}

export const EMPTY_MASTERY: MasteryState = { score: 0, attempts: 0, recentFailures: 0 };

/** `target` is 1 for a correct result, 0 for a wrong one, or a fraction (card grades). */
export function updateMastery(prev: MasteryState, kind: Evidence, target: number): MasteryState {
  const a = ALPHA[kind];
  // The first observation seeds the score so that one correct answer is not dragged toward 0.
  const score = prev.attempts === 0 ? target * 0.6 : prev.score + a * (target - prev.score);
  const failed = target < 0.5;
  return {
    score: Math.min(1, Math.max(0, score)),
    attempts: prev.attempts + 1,
    recentFailures: failed ? prev.recentFailures + 1 : Math.max(0, prev.recentFailures - 1),
  };
}

export const isMastered = (m: MasteryState) => m.score >= MASTERED_SCORE && m.attempts >= MASTERED_MIN_ATTEMPTS;

export const cardGradeTarget = (grade: 0 | 1 | 2 | 3) => [0, 0.4, 0.8, 1][grade];

export interface Graph {
  prereqs: Map<string, string[]>;
  mastery: Map<string, MasteryState>;
}

/**
 * If the learner keeps failing `conceptId`, find the prerequisites (transitively, nearest first) that are not solid
 * and should be reviewed first. Never-attempted prerequisites count as unsolid.
 * Returns null when the concept is not in trouble.
 */
export function reviewPlan(g: Graph, conceptId: string): { reason: string; review: string[] } | null {
  const m = g.mastery.get(conceptId) ?? EMPTY_MASTERY;
  const struggling = m.recentFailures >= 2 || (m.attempts >= 2 && m.score < WEAK_SCORE);
  if (!struggling) return null;
  const seen = new Set<string>([conceptId]);
  const order: string[] = [];
  let frontier = g.prereqs.get(conceptId) ?? [];
  while (frontier.length) {
    const next: string[] = [];
    for (const p of frontier) {
      if (seen.has(p)) continue;
      seen.add(p);
      const pm = g.mastery.get(p) ?? EMPTY_MASTERY;
      if (pm.score < 0.6 || pm.attempts === 0) order.push(p);
      next.push(...(g.prereqs.get(p) ?? []));
    }
    frontier = next;
  }
  const reason =
    m.recentFailures >= 2
      ? `You missed ${m.recentFailures} recent questions on this concept.`
      : "Your accumulated score on this concept is still low.";
  return { reason, review: order };
}
