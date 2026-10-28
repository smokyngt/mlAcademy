import { Link, useSearchParams } from "react-router-dom";
import type { SearchHit } from "@shared/types";
import { useApi } from "../api";
import { useT } from "../i18n";

function Snippet({ s }: { s: string }) {
  const parts = s.split(/(\[\[|\]\])/);
  let on = false;
  return <>{parts.map((p, i) => (p === "[[" ? ((on = true), null) : p === "]]" ? ((on = false), null) : on ? <mark key={i} className="rounded bg-warn/30 px-0.5 text-ink">{p}</mark> : <span key={i}>{p}</span>))}</>;
}

export function SearchPage() {
  const { t } = useT();
  const [sp] = useSearchParams();
  const q = sp.get("q") ?? "";
  const { data, error } = useApi<SearchHit[]>(q.length >= 2 ? `/api/search?q=${encodeURIComponent(q)}` : null);
  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-3xl font-bold tracking-tight">{t("search.title")}</h1>
      <p className="text-muted">{t("search.desc", { q })}</p>
      {error && <p className="text-bad">{error}</p>}
      {q.length < 2 && <p className="text-muted">{t("search.min")}</p>}
      {data && data.length === 0 && <p className="text-muted">{t("search.none")}</p>}
      <ul className="space-y-3">
        {data?.map((h) => (
          <li key={h.lessonId} className="card p-3">
            <Link to={`/lesson/${h.lessonId}`} className="font-semibold text-accent hover:underline">{h.title}</Link>
            <p className="text-xs text-muted">{h.courseTitle}</p>
            <p className="mt-1 text-sm"><Snippet s={h.snippet} /></p>
          </li>
        ))}
      </ul>
    </div>
  );
}
