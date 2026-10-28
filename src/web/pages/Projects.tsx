import { Link, useParams } from "react-router-dom";
import { api, useApi } from "../api";
import { Markdown } from "../components/Markdown";
import { NotebookRunner, type NbCell, type NbGroup, type SavedCell } from "../components/NotebookRunner";
import { useT } from "../i18n";

interface Item { id: string; title: string; summary: string; level: string; minutes: number; tasks: number; done: number; outcomes?: string[]; requires?: string[] }
interface Detail { meta: Item & { stages?: string[] }; cells: NbCell[]; groups: NbGroup[]; saved: SavedCell[] }

const LEVEL: Record<string, string> = { beginner: "border-good text-good", intermediate: "border-accent text-accent", advanced: "border-warn text-warn", expert: "border-bad text-bad" };

export function ProjectsPage() {
  const { t } = useT();
  const { data, error } = useApi<Item[]>("/api/projects");
  if (error) return <p className="text-bad">{error}</p>;
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  return (
    <div className="max-w-5xl space-y-6">
      <div><h1 className="text-3xl font-bold tracking-tight">{t("proj.title")}</h1><p className="text-muted">{t("proj.sub")}</p></div>
      {data.length === 0 && <p className="text-muted">{t("proj.none")}</p>}
      <div className="grid gap-4 md:grid-cols-2">
        {data.map((p) => (
          <Link key={p.id} to={`/projects/${p.id}`} className="card block p-5 transition hover:border-accent">
            <div className="flex items-center gap-2"><span className={`rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase ${LEVEL[p.level] ?? ""}`}>{t(`level.${p.level}` as "level.beginner")}</span><span className="num text-sm text-muted">{p.minutes} {t("common.min")}</span></div>
            <h2 className="mt-2 text-lg font-semibold leading-snug">{p.title}</h2>
            <p className="mt-1 line-clamp-3 text-sm text-muted">{p.summary}</p>
            <div className="mt-3 h-1.5 rounded bg-surface2"><div className="h-1.5 rounded bg-good" style={{ width: `${p.tasks ? (p.done / p.tasks) * 100 : 0}%` }} /></div>
            <p className="mt-1 text-xs text-muted"><span className="num">{p.done}/{p.tasks}</span> {t("proj.tasksDone")}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

export function ProjectPage() {
  const { id = "" } = useParams();
  const { t } = useT();
  const { data, error } = useApi<Detail>(`/api/projects/${id}`);
  if (error) return <p className="text-bad">{error}</p>;
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  const m = data.meta;
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Link to="/projects" className="text-sm text-muted hover:underline">← {t("nav.projects")}</Link>
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{m.title}</h1>
        <p className="mt-1 text-lg text-muted">{m.summary}</p>
        <p className="mt-2 text-sm"><span className="chip">{t(`level.${m.level}` as "level.beginner")}</span> <span className="num ml-2 text-muted">{m.minutes} {t("common.min")}</span></p>
        {m.outcomes && m.outcomes.length > 0 && <div className="card mt-3 p-3 text-sm"><p className="font-semibold">{t("proj.outcomes")}</p><Markdown source={m.outcomes.map((o) => `- ${o}`).join("\n")} /></div>}
      </div>
      <NotebookRunner
        cells={data.cells}
        groups={data.groups}
        saved={data.saved}
        onState={(cells) => api(`/api/projects/${m.id}/state`, { method: "PUT", body: { cells } }).catch(() => {})}
        getSolution={async (index) => (await api<{ solution: string }>(`/api/projects/${m.id}/solution/${index}`, { method: "POST", body: {} })).solution}
      />
    </div>
  );
}
