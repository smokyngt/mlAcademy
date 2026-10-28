import { Link } from "react-router-dom";
import type { CourseReport, Profile } from "@shared/types";
import { useApi } from "../api";
import { useT, type Key } from "../i18n";

const pct = (x: number | null) => (x === null ? "–" : `${Math.round(x * 100)}%`);
const hours = (min: number) => (min >= 60 ? `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")}` : `${min} min`);
const STATUS_CLS = { todo: "text-muted", inProgress: "border-warn text-warn bg-warn/10", done: "border-good text-good bg-good/10" } as const;

function verdict(r: CourseReport): Key {
  if (r.quizAttempts < 3 && (r.mastery === null || r.mastery === 0 || r.quizAttempts === 0)) return "prof.verdict.none";
  const m = r.mastery ?? r.quizAccuracy ?? 0;
  return m >= 0.8 ? "prof.verdict.solid" : m >= 0.55 ? "prof.verdict.ok" : "prof.verdict.weak";
}

function Bar({ value }: { value: number }) {
  return <div className="h-2 flex-1 rounded bg-surface2" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}><div className="h-2 rounded bg-accent" style={{ width: `${Math.min(1, value) * 100}%` }} /></div>;
}

function CourseCard({ r }: { r: CourseReport }) {
  const { t } = useT();
  return (
    <div className="card space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Link to={`/courses/${r.id}`} className="font-semibold hover:underline">{r.title}</Link>
        <span className={`chip ${STATUS_CLS[r.status]}`}>{t(`prof.status.${r.status}` as Key)}</span>
        <span className="ml-auto text-sm text-muted">{r.lessonsDone}/{r.lessonsTotal}</span>
      </div>
      <Bar value={r.lessonsTotal ? r.lessonsDone / r.lessonsTotal : 0} />
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="stat"><div className="stat-label">{t("prof.quiz")}</div><div className="num font-semibold">{pct(r.quizAccuracy)}</div></div>
        <div className="stat"><div className="stat-label">{t("prof.ex")}</div><div className="num font-semibold">{r.exercisesSolved}/{r.exercisesTotal}</div></div>
        <div className="stat"><div className="stat-label">{t("prof.mastery")}</div><div className="num font-semibold">{pct(r.mastery)}</div></div>
      </div>
      <p className="text-sm">{t(verdict(r))}</p>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted">
        {r.nextLesson && <span>{t("prof.next")} <Link className="text-accent underline" to={`/lesson/${r.nextLesson.id}`}>{r.nextLesson.title}</Link></span>}
        {r.status !== "done" && <span>{t("prof.left", { n: r.minutesLeft })}</span>}
        {r.lessonsDone > 0 && <Link className="ml-auto text-accent underline" to={`/courses/${r.id}/assessment`}>{t("prof.assess")}</Link>}
      </div>
    </div>
  );
}

function ConceptList({ title, items }: { title: string; items: Profile["strengths"] }) {
  const { t } = useT();
  return (
    <div className="card p-4">
      <h2 className="mb-2 font-semibold">{title}</h2>
      {items.length === 0 ? <p className="text-sm text-muted">{t("prof.noneYet")}</p> : (
        <ul className="space-y-1 text-sm">{items.map((c) => <li key={c.conceptId} className="flex justify-between gap-3"><span>{c.title}</span><span className="num text-muted">{pct(c.score)}</span></li>)}</ul>
      )}
    </div>
  );
}

export function ProfilePage() {
  const { t, lang } = useT();
  const { data: p, error } = useApi<Profile>("/api/profile");
  if (error) return <p className="text-bad">{error}</p>;
  if (!p) return <p className="text-muted">{t("common.loading")}</p>;
  const max = Math.max(1, ...p.activity.map((d) => d.n));
  const parts = [["math", t("cat.part.math")], ["ml", t("cat.part.ml")]] as const;
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{t("prof.title")}</h1>
        <p className="text-muted">{t("prof.sub")}{p.since && <> {t("prof.since", { d: new Date(p.since).toLocaleDateString(lang) })}</>}</p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[
          [t("prof.lessons"), `${p.lessonsDone}/${p.lessonsTotal}`],
          [t("prof.timeDone"), hours(p.minutesDone)],
          [t("prof.timeLeft"), hours(p.minutesLeft)],
          [t("prof.days"), String(p.activeDays)],
        ].map(([k, v]) => <div key={k} className="stat"><div className="stat-label">{k}</div><div className="stat-value num">{v}</div></div>)}
      </div>

      <div className="card p-4">
        <p className="text-xs uppercase tracking-wide text-muted">{t("prof.now")}</p>
        {p.current ? (
          <>
            <p className="mt-1 text-xl font-semibold">{p.current.title}</p>
            <p className="text-sm text-muted">{p.current.lessonsDone}/{p.current.lessonsTotal} · {t(`prof.status.${p.current.status}` as Key)}</p>
            {p.current.nextLesson && <Link className="btn btn-primary mt-3" to={`/lesson/${p.current.nextLesson.id}`}>{t("prof.continue")} → {p.current.nextLesson.title}</Link>}
          </>
        ) : <p className="mt-1">{t("prof.nowNone")}</p>}
      </div>

      <section>
        <h2 className="mb-2 text-xl font-semibold">{t("prof.activity")}</h2>
        <div className="card flex h-24 items-end gap-1 p-3" role="img" aria-label={t("prof.activity")}>
          {p.activity.map((d) => <div key={d.day} title={`${d.day}: ${d.n}`} className="flex-1 rounded-sm bg-accent" style={{ height: `${d.n ? Math.max(8, (d.n / max) * 100) : 3}%`, opacity: d.n ? 1 : 0.25 }} />)}
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <ConceptList title={t("prof.strengths")} items={p.strengths} />
        <ConceptList title={t("prof.weaknesses")} items={p.weaknesses} />
      </div>

      <section className="space-y-6">
        <h2 className="text-xl font-semibold">{t("prof.reports")}</h2>
        {parts.map(([id, label]) => {
          const rs = p.courses.filter((c) => c.part === id);
          return rs.length === 0 ? null : (
            <div key={id}>
              <h3 className="mb-2 font-medium text-muted">{label}</h3>
              <div className="grid gap-4 lg:grid-cols-2">{rs.map((r) => <CourseCard key={r.id} r={r} />)}</div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
