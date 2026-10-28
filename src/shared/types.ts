/** Types shared by server and web client. Server routes return exactly these shapes. */

export type Runtime = "pyodide" | "docker";

export interface CourseSummary {
  id: string;
  title: string;
  summary: string;
  stage: string;
  kind: "authored" | "source";
  part: "math" | "ml";
  lessons: LessonStub[];
}

export type Level = "beginner" | "intermediate" | "advanced" | "expert";

export interface PathModule {
  id: string;
  level: Level;
  title: string;
  lessons: LessonStub[];
  /** Machine-learning modules only: the mathematics lessons this module's lessons require (computed from concept prerequisites). */
  mathNeeded?: { id: string; title: string; status: LessonStub["status"] }[];
}
export interface PathPart { id: "math" | "ml"; modules: PathModule[] }

export interface LessonStub {
  id: string;
  title: string;
  minutes: number;
  position: number;
  status: "new" | "started" | "completed";
}

export interface QuestionPublic {
  id: string;
  conceptId: string;
  kind: "mcq" | "numeric";
  stage: string;
  prompt: string;
  options: string[];
}

export interface ExercisePublic {
  id: string;
  title: string;
  difficulty: number;
  runtime: Runtime;
  prompt: string;
  starter: string;
  tests: string;
  hintCount: number;
  conceptIds: string[];
  /** The learner's saved code, if any. */
  savedCode: string | null;
  solved: boolean;
  hintsUsed: number;
  solutionRevealed: boolean;
}

export interface LessonDetail {
  id: string;
  courseId: string;
  courseTitle: string;
  title: string;
  summary: string;
  kind: "authored" | "source";
  minutes: number;
  body: string;
  /** true when the body is a real translation into the requested language (otherwise the English text is shown). */
  translated?: boolean;
  figures?: FigureData[];
  sources: string[];
  teaches: ConceptRef[];
  requires: ConceptRef[];
  questions: QuestionPublic[];
  exercises: ExercisePublic[];
  status: "new" | "started" | "completed";
  bookmarked: boolean;
  prev: { id: string; title: string } | null;
  next: { id: string; title: string } | null;
}

export interface FigureData {
  id: string;
  /** Appears after the n-th `##` section of the lesson (0 = after the intro). */
  after: number;
  kind: "mermaid" | "plot" | "video";
  caption: string;
  code?: string;
  /** kind "video": a Manim render under /manim/. */
  src?: string;
  spec?: unknown;
}

export interface ConceptRef {
  id: string;
  title: string;
  mastery: number | null;
}

export interface AnswerResult {
  correct: boolean;
  explanation: string;
  /** Index of the correct option (mcq) or the correct number (numeric). */
  answer: number;
  mastery: number;
  recommendation: Recommendation | null;
}

export interface Recommendation {
  conceptId: string;
  conceptTitle: string;
  reason: string;
  review: { conceptId: string; title: string; score: number | null; lessonId: string | null }[];
}

export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  stage: string;
  prerequisites: string[];
  mastery: number | null;
  attempts: number;
  lessonIds: string[];
}

export interface Card {
  id: string;
  conceptId: string;
  conceptTitle: string;
  lessonId: string | null;
  front: string;
  back: string;
  isNew: boolean;
}

export interface ExerciseResult {
  phase: "ok" | "user_error" | "test_error";
  stdout: string;
  error: string;
  results: { name: string; passed: boolean; message: string }[];
}

export interface Dashboard {
  lessonsCompleted: number;
  lessonsTotal: number;
  authoredCompleted: number;
  authoredTotal: number;
  dueCards: number;
  newCards: number;
  exercisesSolved: number;
  exercisesTotal: number;
  streakDays: number;
  masteredConcepts: number;
  conceptsTotal: number;
  stageMastery: { stage: string; mastered: number; total: number; avgScore: number | null }[];
  continueLesson: { id: string; title: string; courseTitle: string } | null;
  recommendations: Recommendation[];
  recent: { kind: string; label: string; at: number }[];
}

export interface SearchHit {
  lessonId: string;
  title: string;
  courseTitle: string;
  snippet: string;
}

export interface CourseReport {
  id: string; title: string; part: "math" | "ml"; stage: string;
  status: "todo" | "inProgress" | "done";
  lessonsDone: number; lessonsTotal: number; minutesLeft: number;
  quizAccuracy: number | null; quizAttempts: number;
  exercisesSolved: number; exercisesTotal: number;
  mastery: number | null;
  nextLesson: { id: string; title: string } | null;
}
export interface ProfileDay { day: string; n: number }
export interface Profile {
  since: number | null;
  lessonsDone: number; lessonsTotal: number;
  minutesDone: number; minutesLeft: number; activeDays: number;
  courses: CourseReport[];
  activity: ProfileDay[];
  strengths: { conceptId: string; title: string; score: number }[];
  weaknesses: { conceptId: string; title: string; score: number }[];
  current: CourseReport | null;
}
