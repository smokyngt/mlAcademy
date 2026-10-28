import { Link } from "react-router-dom";
import { useApi } from "../api";
import { useT } from "../i18n";

interface Src {
  provenance: string;
  library: { path: string; title: string; authors: string; publisher: string; used: string; exists: boolean }[];
  external: { ref: string; for: string }[];
  lessons: { id: string; title: string; sources: { text: string; read: boolean; exists: boolean | null }[] }[];
}

export function SourcesPage() {
  const { t } = useT();
  const { data, error } = useApi<Src>("/api/sources");
  if (error) return <p className="text-bad">{error}</p>;
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  return (
    <div className="max-w-4xl space-y-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{t("nav.sources")}</h1>
        <div className="mt-3 rounded-xl border border-warn/50 bg-warn/10 p-4">
          <p className="mb-1 text-sm font-bold uppercase tracking-wide text-warn">{t("sources.how")}</p>
          <p>{data.provenance}</p>
        </div>
      </div>

      <section>
        <h2 className="mb-3 text-xl font-bold">{t("sources.library")}</h2>
        <ul className="card divide-y divide-line">
          {data.library.map((l) => (
            <li key={l.path} className="p-3">
              <p className="font-semibold">{l.title} <span className="font-normal text-muted">· {l.authors}</span></p>
              <p className="text-sm text-muted">{l.publisher}</p>
              <p className="mt-1 text-sm">{l.used}</p>
              <p className="mt-1 font-mono text-xs">{l.path} <span className={l.exists ? "text-good" : "text-bad"}>{l.exists ? t("sources.found") : t("sources.notFound")}</span></p>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="mb-3 text-xl font-bold">{t("sources.perLesson")}</h2>
        <div className="space-y-3">
          {data.lessons.map((l) => (
            <div key={l.id} className="card p-3">
              <Link to={`/lesson/${l.id}`} className="font-semibold text-accent hover:underline">{l.title}</Link>
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm">
                {l.sources.length === 0 && <li className="text-muted">{t("sources.none")}</li>}
                {l.sources.map((s) => <li key={s.text}><span className={`mr-2 rounded-full border px-2 text-xs font-semibold ${s.read ? "border-good text-good" : "border-warn text-warn"}`}>{s.read ? t("sources.read") : t("sources.pointer")}</span>{s.text} {s.exists === true && <span className="text-good">✓</span>}{s.exists === false && <span className="text-bad">{t("sources.missing")}</span>}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-xl font-bold">{t("sources.external")}</h2>
        <ul className="card divide-y divide-line">
          {data.external.map((e) => <li key={e.ref} className="p-3"><p>{e.ref}</p><p className="text-sm text-muted">{t("sources.usedFor", { f: e.for })}</p></li>)}
        </ul>
      </section>
    </div>
  );
}
