import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { LessonDetail } from "@shared/types";
import { api, useApi } from "../api";
import { headings } from "../components/Markdown";
import { LessonBody } from "../components/LessonBody";
import { Question } from "../components/Quiz";
import { ExerciseCard } from "../components/Exercise";
import { useT } from "../i18n";
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/layouts/docs/page";

export function LessonPage() {
  const { t } = useT();
  const { id = "" } = useParams();
  const { data: l, error, reload } = useApi<LessonDetail>(`/api/lessons/${id}`);
  const [answered, setAnswered] = useState({ right: 0, total: 0 });
  const [note, setNote] = useState("");

  useEffect(() => { window.scrollTo(0, 0); setAnswered({ right: 0, total: 0 }); api(`/api/lessons/${id}/visit`, { body: {} }).catch(() => {}); }, [id]);
  useEffect(() => { if (l) document.title = `${l.title} · ML Academy`; }, [l]);

  if (error) return <p className="text-bad">{t("lesson.notFound", { err: error })}</p>;
  if (!l) return <p className="text-muted">{t("common.loading")}</p>;

  const toc = headings(l.body);
  const weakPrereqs = l.requires.filter((r) => r.mastery !== null && r.mastery < 0.5);
  const untouched = l.requires.filter((r) => r.mastery === null);
  const toggleComplete = async () => { await api(`/api/lessons/${id}/complete`, { body: { completed: l.status !== "completed" } }); reload(); };
  const toggleBookmark = async () => {
    if (l.bookmarked) await api(`/api/bookmarks/${id}`, { method: "DELETE" });
    else await api(`/api/bookmarks/${id}`, { method: "PUT", body: { note } });
    reload();
  };

  const tocItems = [
    ...toc.map((h) => ({ title: h.title, url: `#${h.id}`, depth: 2 })),
    ...(l.questions.length ? [{ title: t("lesson.check"), url: "#knowledge-check", depth: 2 }] : []),
    ...(l.exercises.length ? [{ title: t("lesson.lab"), url: "#exercises", depth: 2 }] : []),
  ];

  return (
    <DocsPage toc={tocItems}>
      <p className="text-sm text-muted"><Link to={`/courses/${l.courseId}`} className="hover:underline">{l.courseTitle}</Link> · {l.minutes} {t("common.min")} · <span className="chip">{l.kind === "authored" ? t("lesson.authored") : t("lesson.sourceMaterial")}</span> · <span className="text-xs">{t("lesson.contentNote")}</span></p>
      <DocsTitle>{l.title}</DocsTitle>
      {l.summary && <DocsDescription>{l.summary}</DocsDescription>}

      <div className="not-prose mb-6 flex flex-wrap items-center gap-2">
        <button className={`btn ${l.status === "completed" ? "" : "btn-primary"}`} onClick={toggleComplete}>{l.status === "completed" ? t("lesson.completed") : t("lesson.complete")}</button>
        <button className="btn" onClick={toggleBookmark}>{l.bookmarked ? t("lesson.bookmarked") : t("lesson.bookmark")}</button>
        {!l.bookmarked && <input className="input text-xs" placeholder={t("lesson.note")} value={note} onChange={(e) => setNote(e.target.value)} aria-label={t("lesson.noteAria")} />}
      </div>

      {(weakPrereqs.length > 0 || untouched.length > 0) && l.kind === "authored" && (
        <div className="not-prose mb-6 rounded-lg border border-warn/50 bg-warn/10 p-3 text-sm">
          <p className="font-semibold">{t("lesson.prereq")}</p>
          <ul className="list-disc pl-5">
            {weakPrereqs.map((r) => <li key={r.id}>{t("lesson.prereqWeak", { title: r.title, pct: Math.round((r.mastery ?? 0) * 100) })}</li>)}
            {untouched.map((r) => <li key={r.id}>{t("lesson.prereqNone", { title: r.title })}</li>)}
          </ul>
        </div>
      )}

      {l.kind === "authored" && (l.teaches.length > 0 || l.requires.length > 0) && (
        <div className="not-prose card mb-6 grid gap-4 p-3 text-sm sm:grid-cols-2">
          {([[t("lesson.teaches"), l.teaches], [t("lesson.requires"), l.requires]] as const).map(([label, list]) => (
            <div key={label}>
              <p className="text-xs uppercase tracking-wide text-muted">{label}</p>
              {list.length === 0 && <p className="text-xs text-muted">{t("lesson.none")}</p>}
              {list.map((c) => (
                <div key={c.id} className="mt-1">
                  <div className="flex justify-between text-xs"><span>{c.title}</span><span className="font-mono text-muted">{c.mastery === null ? "–" : `${Math.round(c.mastery * 100)}%`}</span></div>
                  <div className="h-1.5 rounded bg-surface2"><div className="h-1.5 rounded bg-accent" style={{ width: `${(c.mastery ?? 0) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      <DocsBody><LessonBody body={l.body} figures={l.figures ?? []} math={l.kind === "authored"} /></DocsBody>

      {l.questions.length > 0 && (
        <section className="not-prose mt-12" id="knowledge-check">
          <h2 className="text-2xl font-bold">{t("lesson.check")}</h2>
          <p className="mb-4 text-sm text-muted">{t("lesson.checkHelp")}</p>
          <div className="space-y-4">
            {l.questions.map((q, i) => <Question key={q.id} q={q} index={i} onAnswered={(r) => setAnswered((a) => ({ right: a.right + (r.correct ? 1 : 0), total: a.total + 1 }))} />)}
          </div>
          {answered.total > 0 && <p className="mt-3 text-sm text-muted">{t("lesson.session", { right: answered.right, total: answered.total })}</p>}
        </section>
      )}

      {l.exercises.length > 0 && (
        <section className="not-prose mt-12" id="exercises">
          <h2 className="text-2xl font-bold">{t("lesson.lab")}</h2>
          <div className="mt-3 space-y-6">{l.exercises.map((e) => <ExerciseCard key={e.id} ex={e} onSolved={reload} />)}</div>
        </section>
      )}

      {l.sources.length > 0 && (
        <section className="not-prose mt-12 text-sm">
          <h2 className="text-lg font-bold">{t("lesson.sources")}</h2>
          <p className="mb-1 text-muted">{l.kind === "authored" ? t("lesson.sourcesAuthored") : t("lesson.sourcesImported")}</p>
          <ul className="list-disc pl-5 font-mono text-xs">{l.sources.slice(0, 12).map((s) => <li key={s}>{s}</li>)}</ul>
        </section>
      )}
    </DocsPage>
  );
}
