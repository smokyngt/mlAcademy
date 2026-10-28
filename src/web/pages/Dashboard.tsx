import { Link } from "react-router-dom";
import type { Dashboard as D } from "@shared/types";
import { useApi } from "../api";
import { useT, type Key } from "../i18n";

type T = ReturnType<typeof useT>["t"];
const ago = (t: T, at: number) => {
  const m = Math.round((Date.now() - at) / 60000);
  return m < 1 ? t("dash.ago.now") : m < 60 ? t("dash.ago.min", { n: m }) : m < 1440 ? t("dash.ago.h", { n: Math.round(m / 60) }) : t("dash.ago.d", { n: Math.round(m / 1440) });
};
const KINDS: Record<string, Key> = { "quiz correct": "dash.kind.quizCorrect", "quiz missed": "dash.kind.quizMissed", "exercise passed": "dash.kind.exercisePassed", "exercise run": "dash.kind.exerciseRun", "card review": "dash.kind.cardReview" };

export function Dashboard() {
  const { t } = useT();
  const { data: d, error } = useApi<D>("/api/dashboard");
  if (error) return <p className="text-bad">{t("dash.apiError", { err: error })}</p>;
  if (!d) return <p className="text-muted">{t("common.loading")}</p>;
  const empty = d.lessonsCompleted === 0 && d.recent.length === 0;
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{empty ? t("dash.welcome") : t("dash.title")}</h1>
        <p className="text-muted">{t("dash.subtitle")}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="card p-4 md:col-span-2">
          <p className="text-xs uppercase tracking-wide text-muted">{t("dash.continue")}</p>
          {d.continueLesson ? (
            <>
              <p className="mt-1 text-xl font-semibold">{d.continueLesson.title}</p>
              <p className="text-sm text-muted">{d.continueLesson.courseTitle}</p>
              <Link className="btn btn-primary mt-3" to={`/lesson/${d.continueLesson.id}`}>{empty ? t("dash.begin") : t("dash.resume")} →</Link>
            </>
          ) : <p className="mt-1">{t("dash.allDone")}</p>}
        </div>
        <div className="card p-4">
          <p className="text-xs uppercase tracking-wide text-muted">{t("dash.reviews")}</p>
          <p className="mt-1 text-3xl font-bold">{d.dueCards}</p>
          <p className="text-sm text-muted">{t("dash.dueNow", { n: d.newCards })}</p>
          <Link className="btn mt-3" to="/review">{t("dash.startReview")}</Link>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          [t("dash.authored"), `${d.authoredCompleted}/${d.authoredTotal}`],
          [t("dash.source"), `${d.lessonsCompleted - d.authoredCompleted}/${d.lessonsTotal - d.authoredTotal}`],
          [t("dash.exercises"), `${d.exercisesSolved}/${d.exercisesTotal}`],
          [t("dash.mastered"), `${d.masteredConcepts}/${d.conceptsTotal}`],
        ].map(([k, v]) => <div key={k} className="card p-4"><p className="text-xs uppercase tracking-wide text-muted">{k}</p><p className="mt-1 text-2xl font-bold">{v}</p></div>)}
      </div>
      <p className="-mt-4 text-sm text-muted">{t("dash.streak", { n: d.streakDays })}</p>

      {d.recommendations.length > 0 && (
        <section>
          <h2 className="mb-2 text-xl font-semibold">{t("dash.recommended")}</h2>
          <div className="space-y-3">
            {d.recommendations.map((r) => (
              <div key={r.conceptId} className="rounded-lg border border-warn/50 bg-warn/10 p-3 text-sm">
                <p className="font-semibold">{r.conceptTitle}</p>
                <p>{r.reason} {t("dash.revisit")}</p>
                <ul className="list-disc pl-5">
                  {r.review.length === 0 && <li>{t("dash.noWeak")}</li>}
                  {r.review.map((x) => <li key={x.conceptId}>{x.lessonId ? <Link className="text-accent underline" to={`/lesson/${x.lessonId}`}>{x.title}</Link> : x.title} <span className="text-muted">({x.score === null ? t("common.notAttempted") : `${Math.round(x.score * 100)}%`})</span></li>)}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h2 className="mb-2 text-xl font-semibold">{t("dash.byStage")}</h2>
        <div className="card divide-y divide-line">
          {d.stageMastery.map((s) => (
            <Link key={s.stage} to={`/concepts?stage=${s.stage}`} className="flex items-center gap-4 p-3 text-sm transition hover:bg-surface2">
              <span className="w-56 shrink-0">{t(`stage.${s.stage}` as Key)}</span>
              <div className="h-2 flex-1 rounded bg-surface2" role="progressbar" aria-valuenow={Math.round((s.avgScore ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}><div className="h-2 rounded bg-accent" style={{ width: `${(s.avgScore ?? 0) * 100}%` }} /></div>
              <span className="w-40 shrink-0 text-right text-muted">{s.mastered}/{s.total} · {s.avgScore === null ? t("dash.noData") : `${Math.round(s.avgScore * 100)}%`}</span>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-xl font-semibold">{t("dash.recent")}</h2>
        {d.recent.length === 0 ? <p className="text-sm text-muted">{t("dash.nothing")}</p> : (
          <ul className="card divide-y divide-line text-sm">
            {d.recent.map((r, i) => <li key={i} className="flex justify-between p-2.5"><span><span className="chip mr-2">{KINDS[r.kind] ? t(KINDS[r.kind]) : r.kind}</span>{r.label}</span><span className="text-muted">{ago(t, r.at)}</span></li>)}
          </ul>
        )}
      </section>
    </div>
  );
}
