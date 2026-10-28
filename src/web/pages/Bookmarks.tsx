import { Link } from "react-router-dom";
import { api, useApi } from "../api";
import { useT } from "../i18n";

interface B { lessonId: string; title: string; note: string; createdAt: number }

export function BookmarksPage() {
  const { t, lang } = useT();
  const { data, reload } = useApi<B[]>("/api/bookmarks");
  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-3xl font-bold tracking-tight">{t("bm.title")}</h1>
      {data && data.length === 0 && <p className="text-muted">{t("bm.empty")}</p>}
      <ul className="space-y-2">
        {data?.map((b) => (
          <li key={b.lessonId} className="card flex items-start justify-between gap-3 p-3">
            <div><Link className="font-semibold text-accent hover:underline" to={`/lesson/${b.lessonId}`}>{b.title}</Link>{b.note && <p className="text-sm text-muted">{b.note}</p>}<p className="text-xs text-muted">{new Date(b.createdAt).toLocaleDateString(lang)}</p></div>
            <button className="btn text-xs" onClick={async () => { await api(`/api/bookmarks/${b.lessonId}`, { method: "DELETE" }); reload(); }}>{t("bm.remove")}</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
