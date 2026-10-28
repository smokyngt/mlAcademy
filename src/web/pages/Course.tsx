import { Link, useParams } from "react-router-dom";
import type { CourseSummary, ConceptNode } from "@shared/types";
import { useApi } from "../api";
import { useT } from "../i18n";

export function CoursePage() {
  const { t } = useT();
  const { id = "" } = useParams();
  const { data } = useApi<CourseSummary[]>("/api/catalog");
  const { data: concepts } = useApi<ConceptNode[]>("/api/concepts");
  const c = data?.find((x) => x.id === id);
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  if (!c) return <p>{t("course.notFound")}</p>;
  const byId = new Map((concepts ?? []).map((k) => [k.id, k]));
  const lessonOf = new Map<string, number>();
  concepts?.forEach((k) => k.lessonIds.forEach((l) => lessonOf.set(l, 1)));
  return (
    <div className="space-y-6">
      <div>
        <Link to="/courses" className="text-sm text-muted hover:underline">{t("course.back")}</Link>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">{c.title}</h1>
        <p className="max-w-3xl text-muted">{c.summary}</p>
        {c.kind === "authored" && <Link className="btn btn-primary mt-3" to={`/courses/${c.id}/assessment`}>{t("course.assessment")}</Link>}
      </div>
      <ol className="card divide-y divide-line">
        {c.lessons.map((l, i) => {
          const taught = [...byId.values()].filter((k) => k.lessonIds.includes(l.id) && c.kind === "authored");
          return (
            <li key={l.id}>
              <Link to={`/lesson/${l.id}`} className="flex items-center gap-3 p-3 hover:bg-surface2">
                <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs font-semibold ${l.status === "completed" ? "border-good bg-good/15 text-good" : l.status === "started" ? "border-accent text-accent" : "border-line text-muted"}`}>{l.status === "completed" ? "✓" : i + 1}</span>
                <span className="flex-1">
                  <span className="block font-medium">{l.title}</span>
                  {taught.length > 0 && <span className="block text-xs text-muted">{taught.map((k) => `${k.title} (${k.mastery === null ? "–" : Math.round(k.mastery * 100) + "%"})`).join(" · ")}</span>}
                </span>
                <span className="text-xs text-muted">{l.minutes} {t("common.min")}</span>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
