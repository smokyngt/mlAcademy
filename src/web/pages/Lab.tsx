import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { CodeEditor, usePyState } from "../components/Exercise";
import { pyRunner, type CellResult } from "../runner/pyodide";
import { useT } from "../i18n";

interface Cell { id: string; source: string }
interface NbMeta { id: string; title: string; updatedAt: number }
interface RunnerStatus { available: boolean; reason: string; image: string }

const uid = () => Math.random().toString(36).slice(2, 10);
const STARTER: Cell[] = [
  { id: uid(), source: "import numpy as np\n\nA = np.array([[1., 2.], [3., 4.]])\nx = np.array([1., 1.])\nA @ x" },
  { id: uid(), source: "# Try the lesson-5 gradient check here\nf = lambda v: v[0]**2 + 3*v[1]**2\nh = 1e-5\nx0 = np.array([1., 1.])\nnp.array([(f(x0 + h*e) - f(x0 - h*e)) / (2*h) for e in np.eye(2)])" },
];

export function LabPage() {
  const { t } = useT();
  const [list, setList] = useState<NbMeta[]>([]);
  const [id, setId] = useState<string | null>(null);
  const [title, setTitle] = useState(t("lab.untitled"));
  const [cells, setCells] = useState<Cell[]>(STARTER);
  const [outs, setOuts] = useState<Record<string, CellResult | undefined>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [backend, setBackend] = useState<"pyodide" | "docker">("pyodide");
  const [runner, setRunner] = useState<RunnerStatus | null>(null);
  const [saved, setSaved] = useState<"" | "saved" | "saving">("");
  const py = usePyState();
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const refresh = () => api<NbMeta[]>("/api/notebooks").then(setList);
  useEffect(() => { refresh(); api<RunnerStatus>("/api/runner/status").then(setRunner); }, []);

  function persist(nid: string, ttl: string, c: Cell[]) {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => { await api(`/api/notebooks/${nid}`, { method: "PUT", body: { title: ttl, cells: c } }); setSaved("saved"); refresh(); }, 700);
    setSaved("saving");
  }
  const update = (c: Cell[], ttl = title) => { setCells(c); if (id) persist(id, ttl, c); };

  async function open(nid: string) {
    const n = await api<{ id: string; title: string; cells: Cell[] }>(`/api/notebooks/${nid}`);
    setId(n.id); setTitle(n.title); setCells(n.cells); setOuts({}); await pyRunner.reset().catch(() => {});
  }
  async function create() {
    const nid = uid();
    await api(`/api/notebooks/${nid}`, { method: "PUT", body: { title, cells } });
    setId(nid); refresh();
  }
  async function remove() {
    if (!id || !confirm(t("lab.confirmDelete"))) return;
    await api(`/api/notebooks/${id}`, { method: "DELETE" });
    setId(null); setTitle(t("lab.untitled")); setCells(STARTER); setOuts({}); refresh();
  }

  async function run(index: number) {
    const cell = cells[index];
    setBusy(cell.id);
    try {
      if (backend === "pyodide") {
        const r = await pyRunner.runCell(cell.source);
        setOuts((o) => ({ ...o, [cell.id]: r }));
      } else {
        // Docker replays every cell up to this one in a fresh container: state is rebuilt, not persisted.
        const r = await api<{ outputs?: CellResult[]; unavailable?: string }>("/api/notebooks/run", { body: { cells: cells.slice(0, index + 1).map((c) => c.source) } });
        if (r.unavailable) setOuts((o) => ({ ...o, [cell.id]: { ok: false, stdout: "", value: null, error: r.unavailable! } }));
        else setOuts((o) => { const n = { ...o }; r.outputs!.forEach((x, i) => { n[cells[i].id] = x; }); return n; });
      }
    } catch (e) { setOuts((o) => ({ ...o, [cell.id]: { ok: false, stdout: "", value: null, error: String((e as Error).message) } })); }
    setBusy(null);
  }
  async function runAll() { for (let i = 0; i < cells.length; i++) { await run(i); } }

  return (
    <div className="grid gap-6 lg:grid-cols-[15rem_1fr]">
      <aside className="space-y-3">
        <h1 className="text-2xl font-bold">{t("nav.lab")}</h1>
        <p className="text-sm text-muted">{t("lab.intro")}</p>
        <div className="card space-y-2 p-3 text-sm">
          <p className="font-semibold">{t("lab.runtime")}</p>
          <label className="flex items-start gap-2"><input type="radio" checked={backend === "pyodide"} onChange={() => setBackend("pyodide")} /><span>{t("lab.numpyBox")}<br /><span className="text-xs text-muted">{py.state === "loading" ? t("lab.pyLoading") : py.state === "ready" ? t("lab.pyReady") : py.state === "error" ? py.detail : t("lab.pyIdle")}</span></span></label>
          <label className="flex items-start gap-2"><input type="radio" checked={backend === "docker"} onChange={() => setBackend("docker")} /><span>{t("lab.torchBox")}<br /><span className={`text-xs ${runner?.available ? "text-good" : "text-warn"}`}>{runner ? (runner.available ? t("lab.runnerReady") : runner.reason) : t("lab.checking")}</span></span></label>
        </div>
        <div className="card p-3 text-sm">
          <p className="mb-1 font-semibold">{t("lab.saved")}</p>
          {list.length === 0 && <p className="text-xs text-muted">{t("lab.none")}</p>}
          <ul className="space-y-1">{list.map((n) => <li key={n.id}><button className={`w-full truncate text-left hover:text-accent ${n.id === id ? "font-semibold text-accent" : ""}`} onClick={() => open(n.id)}>{n.title}</button></li>)}</ul>
          <button className="btn mt-2 w-full justify-center" onClick={() => { setId(null); setTitle(t("lab.untitled")); setCells(STARTER); setOuts({}); pyRunner.reset().catch(() => {}); }}>{t("lab.newNb")}</button>
        </div>
      </aside>
      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <input className="input flex-1 text-lg font-semibold" value={title} aria-label={t("lab.titleAria")} onChange={(e) => { setTitle(e.target.value); if (id) persist(id, e.target.value, cells); }} />
          {!id ? <button className="btn btn-primary" onClick={create}>{t("lab.saveNb")}</button> : <><span className="text-xs text-muted">{saved ? t(saved === "saved" ? "lab.stateSaved" : "lab.stateSaving") : ""}</span><button className="btn" onClick={remove}>{t("lab.delete")}</button></>}
          <button className="btn" onClick={runAll} disabled={!!busy}>{t("lab.runAll")}</button>
          <button className="btn" onClick={() => { setOuts({}); pyRunner.reset().catch(() => {}); }}>{t("proj.restart")}</button>
        </div>
        {cells.map((c, i) => {
          const o = outs[c.id];
          return (
            <div key={c.id} className="card p-3">
              <div className="mb-2 flex items-center gap-2 text-xs text-muted"><span className="font-mono">[{i + 1}]</span>
                <button className="btn !py-0.5 text-xs btn-primary" onClick={() => run(i)} disabled={busy === c.id}>{busy === c.id ? t("ex.running") : `▶ ${t("proj.run")}`}</button>
                <button className="btn !py-0.5 text-xs" onClick={() => update(cells.filter((x) => x.id !== c.id))} disabled={cells.length === 1}>{t("lab.deleteCell")}</button>
                <button className="btn !py-0.5 text-xs" onClick={() => { const n = [...cells]; n.splice(i + 1, 0, { id: uid(), source: "" }); update(n); }}>{t("lab.cellBelow")}</button>
              </div>
              <CodeEditor rows={Math.max(3, c.source.split("\n").length + 1)} value={c.source} onChange={(v) => update(cells.map((x) => (x.id === c.id ? { ...x, source: v } : x)))} onRun={() => run(i)} />
              {o && (
                <div className="mt-2 space-y-1 text-xs">
                  {o.stdout && <pre className="whitespace-pre-wrap rounded-md border border-line bg-codebg p-2">{o.stdout}</pre>}
                  {o.value !== null && o.ok && <pre className="whitespace-pre-wrap rounded-md border border-line bg-codebg p-2 text-accent">{o.value}</pre>}
                  {!o.ok && <pre className="whitespace-pre-wrap rounded-md border border-bad/50 bg-bad/10 p-2">{o.error}</pre>}
                </div>
              )}
            </div>
          );
        })}
        <button className="btn" onClick={() => update([...cells, { id: uid(), source: "" }])}>{t("lab.addCell")}</button>
        <p className="text-xs text-muted">{t("lab.footer")}</p>
      </section>
    </div>
  );
}
