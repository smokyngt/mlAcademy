/** Spaced repetition: an SM-2 variant with relearning steps, as a pure function. */

export type Grade = 0 | 1 | 2 | 3; // again, hard, good, easy

export interface CardSchedule {
  ease: number;
  intervalDays: number;
  reps: number;
  lapses: number;
}

const DAY_MS = 86_400_000;
export const NEW_CARD: CardSchedule = { ease: 2.5, intervalDays: 0, reps: 0, lapses: 0 };

export function schedule(prev: CardSchedule, grade: Grade, now: number): CardSchedule & { dueAt: number } {
  let { ease, intervalDays, reps, lapses } = prev;
  if (grade === 0) {
    lapses += 1;
    reps = 0;
    ease = Math.max(1.3, ease - 0.2);
    intervalDays = 10 / (24 * 60); // see it again in ten minutes
  } else {
    if (grade === 1) {
      ease = Math.max(1.3, ease - 0.15);
      intervalDays = reps === 0 ? 1 : Math.max(1, intervalDays * 1.2);
    } else if (grade === 2) {
      intervalDays = reps === 0 ? 1 : reps === 1 ? 3 : intervalDays * ease;
    } else {
      ease += 0.15;
      intervalDays = reps === 0 ? 3 : Math.max(4, intervalDays * ease * 1.3);
    }
    reps += 1;
  }
  return { ease, intervalDays, reps, lapses, dueAt: now + intervalDays * DAY_MS };
}
