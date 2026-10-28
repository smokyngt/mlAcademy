import { sqliteTable, text, integer, real, primaryKey, index } from "drizzle-orm/sqlite-core";

/* ---------- Static curriculum (rebuilt from content/ by `bun run sync`) ---------- */

export const courses = sqliteTable("courses", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  summary: text("summary").notNull(),
  stage: text("stage").notNull(), // "I" .. "VII"
  kind: text("kind").notNull(), // "authored" | "source"
  part: text("part").notNull().default("ml"), // "math" | "ml": the two top-level parts of the academy
  position: integer("position").notNull(),
});

export const lessons = sqliteTable(
  "lessons",
  {
    id: text("id").primaryKey(),
    courseId: text("course_id").notNull().references(() => courses.id),
    position: integer("position").notNull(),
    title: text("title").notNull(),
    summary: text("summary").notNull().default(""),
    kind: text("kind").notNull(), // "authored" | "source"
    minutes: integer("minutes").notNull().default(30),
    body: text("body").notNull(),
    sources: text("sources", { mode: "json" }).$type<string[]>().notNull(), // paths relative to the repo root
    hash: text("hash").notNull(),
  },
  (t) => [index("lessons_course_idx").on(t.courseId, t.position)],
);

export const concepts = sqliteTable("concepts", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  stage: text("stage").notNull(),
});

export const conceptEdges = sqliteTable(
  "concept_edges",
  {
    conceptId: text("concept_id").notNull().references(() => concepts.id),
    prerequisiteId: text("prerequisite_id").notNull().references(() => concepts.id),
  },
  (t) => [primaryKey({ columns: [t.conceptId, t.prerequisiteId] })],
);

export const lessonConcepts = sqliteTable(
  "lesson_concepts",
  {
    lessonId: text("lesson_id").notNull().references(() => lessons.id),
    conceptId: text("concept_id").notNull().references(() => concepts.id),
    role: text("role").notNull(), // "teaches" | "requires"
  },
  (t) => [primaryKey({ columns: [t.lessonId, t.conceptId, t.role] })],
);

export const questions = sqliteTable("questions", {
  id: text("id").primaryKey(),
  lessonId: text("lesson_id").notNull().references(() => lessons.id),
  conceptId: text("concept_id").notNull().references(() => concepts.id),
  kind: text("kind").notNull(), // "mcq" | "numeric"
  stage: text("stage").notNull().default("check"), // "pre" | "check" | "assessment"
  prompt: text("prompt").notNull(),
  options: text("options", { mode: "json" }).$type<string[]>().notNull(),
  answer: text("answer", { mode: "json" }).$type<number | { value: number; tol: number }>().notNull(),
  explanation: text("explanation").notNull(),
});

export const exercises = sqliteTable("exercises", {
  id: text("id").primaryKey(),
  lessonId: text("lesson_id").notNull().references(() => lessons.id),
  title: text("title").notNull(),
  difficulty: integer("difficulty").notNull(), // 1..5
  runtime: text("runtime").notNull(), // "pyodide" (numpy, browser WASM sandbox) | "docker" (torch, server container)
  prompt: text("prompt").notNull(),
  starter: text("starter").notNull(),
  tests: text("tests").notNull(),
  hints: text("hints", { mode: "json" }).$type<string[]>().notNull(),
  solution: text("solution").notNull(),
  conceptIds: text("concept_ids", { mode: "json" }).$type<string[]>().notNull(),
});

export const flashcards = sqliteTable("flashcards", {
  id: text("id").primaryKey(),
  lessonId: text("lesson_id").references(() => lessons.id),
  conceptId: text("concept_id").notNull().references(() => concepts.id),
  front: text("front").notNull(),
  back: text("back").notNull(),
});

/* ---------- Learner state (never wiped by sync) ---------- */

export const lessonProgress = sqliteTable("lesson_progress", {
  lessonId: text("lesson_id").primaryKey(),
  status: text("status").notNull(), // "started" | "completed"
  lastVisitedAt: integer("last_visited_at").notNull(),
  completedAt: integer("completed_at"),
});

export const quizAttempts = sqliteTable(
  "quiz_attempts",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    questionId: text("question_id").notNull(),
    conceptId: text("concept_id").notNull(),
    correct: integer("correct", { mode: "boolean" }).notNull(),
    response: text("response").notNull(),
    at: integer("at").notNull(),
  },
  (t) => [index("quiz_attempts_q").on(t.questionId)],
);

export const exerciseState = sqliteTable("exercise_state", {
  exerciseId: text("exercise_id").primaryKey(),
  code: text("code").notNull(),
  solved: integer("solved", { mode: "boolean" }).notNull().default(false),
  hintsUsed: integer("hints_used").notNull().default(0),
  solutionRevealed: integer("solution_revealed", { mode: "boolean" }).notNull().default(false),
  updatedAt: integer("updated_at").notNull(),
});

export const exerciseAttempts = sqliteTable("exercise_attempts", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  exerciseId: text("exercise_id").notNull(),
  passed: integer("passed", { mode: "boolean" }).notNull(),
  runtime: text("runtime").notNull(),
  output: text("output").notNull(),
  at: integer("at").notNull(),
});

export const cardState = sqliteTable("card_state", {
  cardId: text("card_id").primaryKey(),
  ease: real("ease").notNull().default(2.5),
  intervalDays: real("interval_days").notNull().default(0),
  reps: integer("reps").notNull().default(0),
  lapses: integer("lapses").notNull().default(0),
  dueAt: integer("due_at").notNull(),
  lastReviewedAt: integer("last_reviewed_at"),
});

export const reviewLog = sqliteTable("review_log", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  cardId: text("card_id").notNull(),
  conceptId: text("concept_id").notNull(),
  grade: integer("grade").notNull(), // 0 again, 1 hard, 2 good, 3 easy
  at: integer("at").notNull(),
});

export const conceptMastery = sqliteTable("concept_mastery", {
  conceptId: text("concept_id").primaryKey(),
  score: real("score").notNull(), // 0..1
  attempts: integer("attempts").notNull().default(0),
  recentFailures: integer("recent_failures").notNull().default(0),
  updatedAt: integer("updated_at").notNull(),
});

export const bookmarks = sqliteTable("bookmarks", {
  lessonId: text("lesson_id").primaryKey(),
  note: text("note").notNull().default(""),
  createdAt: integer("created_at").notNull(),
});

export const notebooks = sqliteTable("notebooks", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  cells: text("cells", { mode: "json" }).$type<{ id: string; source: string }[]>().notNull(),
  updatedAt: integer("updated_at").notNull(),
});

/** Written evidence (explanations, experiment reports) for capstone criteria. */
export const evidence = sqliteTable("evidence", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  capstoneId: text("capstone_id").notNull(),
  criterion: text("criterion").notNull(),
  text: text("text").notNull(),
  selfRating: integer("self_rating").notNull(), // 0..3
  at: integer("at").notNull(),
});
