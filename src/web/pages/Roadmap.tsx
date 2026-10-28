import { Link } from "react-router-dom";
import { useApi } from "../api";
import { useT, type Key } from "../i18n";

interface Topic { name: string; status: "authored" | "partial" | "source" | "missing"; authored: string[]; source: { id: string; title: string }[]; note: string | null }
interface Road {
  stages: { id: string; name: string; topics: Topic[] }[];
  counts: Record<Topic["status"], number>;
  platform: { area: string; item: string; severity: "high" | "medium" | "low" }[];
}

const STYLE: Record<Topic["status"], { label: Key; cls: string }> = {
  authored: { label: "roadmap.authored", cls: "border-good text-good bg-good/10" },
  partial: { label: "roadmap.partial", cls: "border-warn text-warn bg-warn/10" },
  source: { label: "roadmap.source", cls: "border-accent text-accent bg-accentsoft" },
  missing: { label: "roadmap.missing", cls: "border-bad text-bad bg-bad/10" },
};
const SEV = { high: "text-bad", medium: "text-warn", low: "text-muted" } as const;

export function RoadmapPage() {
  const { t } = useT();
  const { data, error } = useApi<Road>("/api/roadmap");
  if (error) return <p className="text-bad">{error}</p>;
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  const total = Object.values(data.counts).reduce((a, b) => a + b, 0);
  return (
    <div className="max-w-4xl space-y-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{t("nav.roadmap")}</h1>
        <p className="mt-1 text-muted">{t("roadmap.intro")}</p>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {(Object.keys(STYLE) as Topic["status"][]).map((k) => (
          <div key={k} className="stat"><div className="stat-label">{t(STYLE[k].label)}</div><div className="stat-value num">{data.counts[k]}<span className="text-base font-medium text-muted"> / {total}</span></div></div>
        ))}
      </div>

      {data.stages.map((st) => (
        <section key={st.id}>
          <h2 className="mb-3 text-xl font-bold">{t("cat.stage", { s: st.id })} · {st.name}</h2>
          <ul className="card divide-y divide-line">
            {st.topics.map((tp) => (
              <li key={tp.name} className="flex flex-wrap items-start gap-x-3 gap-y-1 p-3">
                <span className={`mt-0.5 shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${STYLE[tp.status].cls}`}>{t(STYLE[tp.status].label)}</span>
                <div className="min-w-0 flex-1">
                  <p className="font-medium">{tp.name}</p>
                  {tp.authored.length > 0 && <p className="text-sm">{tp.authored.map((id) => <Link key={id} className="mr-2 text-accent underline" to={`/lesson/${id}`}>{id}</Link>)}</p>}
                  {tp.note && <p className="text-sm text-muted">{tp.note}</p>}
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}

      <section>
        <h2 className="mb-3 text-xl font-bold">{t("roadmap.platform")}</h2>
        <ul className="card divide-y divide-line">
          {data.platform.map((p) => (
            <li key={p.item} className="p-3"><span className={`text-xs font-bold uppercase tracking-wide ${SEV[p.severity]}`}>{t(`roadmap.sev.${p.severity}` as Key)} · {p.area}</span><p>{p.item}</p></li>
          ))}
        </ul>
      </section>
    </div>
  );
}
