import { Link } from "react-router-dom";
import type { Level, PathModule, PathPart } from "@shared/types";
import { useApi } from "../api";
import { useT } from "../i18n";

const LEVEL_STYLE: Record<Level, string> = {
  beginner: "border-good text-good bg-good/10",
  intermediate: "border-accent text-accent bg-accentsoft",
  advanced: "border-warn text-warn bg-warn/10",
  expert: "border-bad text-bad bg-bad/10",
};

export function Catalog() {
  const { t } = useT();
  const { data, error } = useApi<PathPart[]>("/api/path");
  if (error) return <p className="text-bad">{error}</p>;
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  const total = data.reduce((n, p) => n + p.modules.reduce((m, x) => m + x.lessons.length, 0), 0);
  const done = data.reduce((n, p) => n + p.modules.reduce((m, x) => m + x.lessons.filter((l) => l.status === "completed").length, 0), 0);

  const Module = ({ m, index }: { m: PathModule; index: number }) => {
    const d = m.lessons.filter((l) => l.status === "completed").length;
    const mins = m.lessons.reduce((a, l) => a + l.minutes, 0);
    return (
      <li className="card p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="num flex h-7 w-7 items-center justify-center rounded-full bg-surface2 text-sm font-bold text-muted">{index}</span>
          <span className={`rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide ${LEVEL_STYLE[m.level]}`}>{t(`level.${m.level}` as "level.beginner")}</span>
          <h3 className="text-lg font-semibold leading-snug">{m.title}</h3>
          <span className="ml-auto text-sm text-muted">{t("path.lessons", { n: m.lessons.length, min: mins })}</span>
        </div>
        <div className="mt-3 h-1.5 rounded bg-surface2"><div className="h-1.5 rounded bg-good" style={{ width: `${(d / m.lessons.length) * 100}%` }} /></div>
        {m.mathNeeded && m.mathNeeded.length > 0 && (
          <div className="mt-3 rounded-lg bg-surface2 p-2.5 text-sm">
            <span className="mr-2 font-semibold text-muted">{t("path.mathNeeded")}</span>
            {m.mathNeeded.map((x) => (
              <Link key={x.id} to={`/lesson/${x.id}`} className={`mr-1.5 inline-block rounded-full border px-2 py-0.5 text-xs hover:border-accent ${x.status === "completed" ? "border-good text-good" : "border-line"}`}>
                {x.status === "completed" ? "✓ " : ""}{x.title}
              </Link>
            ))}
          </div>
        )}
        <ol className="mt-3 grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {m.lessons.map((l) => (
            <li key={l.id}>
              <Link to={`/lesson/${l.id}`} className="flex items-baseline gap-2 rounded px-1 py-0.5 text-sm hover:bg-surface2">
                <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${l.status === "completed" ? "bg-good" : l.status === "started" ? "bg-accent" : "bg-line"}`} />
                <span className="flex-1">{l.title}</span>
                <span className="num text-xs text-muted">{l.minutes}′</span>
              </Link>
            </li>
          ))}
        </ol>
      </li>
    );
  };

  return (
    <div className="space-y-12">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{t("path.title")}</h1>
        <p className="max-w-3xl text-muted">{t("path.sub")}</p>
        <p className="mt-2 text-sm"><b className="num">{done}</b> / <b className="num">{total}</b> {t("common.completed")}</p>
        <Link to="/roadmap" className="btn mt-3">{t("nav.roadmap")} →</Link>
      </div>
      {data.map((part) => {
        let n = 0;
        return (
          <section key={part.id}>
            <h2 className="mb-4 text-2xl font-bold tracking-tight">{t(part.id === "math" ? "cat.part.math" : "cat.part.ml")}</h2>
            <ol className="space-y-3">{part.modules.map((m) => <Module key={m.id} m={m} index={++n} />)}</ol>
          </section>
        );
      })}
    </div>
  );
}
