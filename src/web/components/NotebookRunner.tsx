import { useEffect, useRef, useState } from "react";
import type { ExerciseResult } from "@shared/types";
import { CodeEditor } from "./CodeEditor";
import { Markdown } from "./Markdown";
import { pyRunner, type CellResult } from "../runner/pyodide";
import { usePyState } from "./Exercise";
import { useT } from "../i18n";

export interface NbCell { index: number; kind: string; title: string; body: string }
export interface NbGroup { task: number; check: number | null; solution: number | null; hints: number[] }
export interface SavedCell { id: string; source: string }

interface Props {
  cells: NbCell[];
  groups: NbGroup[];
  saved: SavedCell[];
  /** Persist learner state (task code, reports, completion). */
  onState: (cells: SavedCell[]) => void;
  /** Resolves the reference solution of a solution cell (server-gated for projects). */
  getSolution: (index: number) => Promise<string>;
}

type Out = { cell?: CellResult; check?: ExerciseResult; error?: string; running?: "run" | "check" };

/** A notebook of cells: prose, runnable code, tasks with checks/hints/solution, and free-text report prompts. */
export function NotebookRunner({ cells, groups, saved, onState, getSolution }: Props) {
  const { t } = useT();
  const py = usePyState();
  const [state, setState] = useState<Record<string, string>>(() => Object.fromEntries(saved.map((s) => [s.id, s.source])));
  const [out, setOut] = useState<Record<number, Out>>({});
  const [hints, setHints] = useState<Record<number, number>>({});
  const [solutions, setSolutions] = useState<Record<number, string>>({});
  const [busyAll, setBusyAll] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const groupOf = new Map(groups.map((g) => [g.task, g]));

  useEffect(() => { pyRunner.reset().catch(() => {}); }, []);

  const put = (id: string, v: string) => {
    setState((s) => {
      const next = { ...s, [id]: v };
      clearTimeout(timer.current);
      timer.current = setTimeout(() => onState(Object.entries(next).map(([k, source]) => ({ id: k, source }))), 700);
      return next;
    });
  };
  const code = (c: NbCell) => state[`code-${c.index}`] ?? state[`task-${c.index}`] ?? c.body;
  const setOutFor = (i: number, o: Out) => setOut((p) => ({ ...p, [i]: { ...p[i], ...o } }));

  async function run(c: NbCell) {
    setOutFor(c.index, { running: "run", error: undefined });
    try { setOutFor(c.index, { cell: await pyRunner.runCell(code(c)), running: undefined }); }
    catch (e) { setOutFor(c.index, { error: String((e as Error).message), running: undefined }); }
  }
  async function check(c: NbCell) {
    const g = groupOf.get(c.index);
    if (!g || g.check === null) return;
    setOutFor(c.index, { running: "check", error: undefined, check: undefined });
    try {
      const r = await pyRunner.runCell(code(c));
      setOutFor(c.index, { cell: r });
      if (!r.ok) { setOutFor(c.index, { running: undefined }); return; }
      const res = await pyRunner.runChecks(cells[g.check].body);
      const passed = res.phase === "ok" && res.results.length > 0 && res.results.every((x) => x.passed);
      setOutFor(c.index, { check: res, running: undefined });
      if (passed) put(`done-${c.index}`, "1");
    } catch (e) { setOutFor(c.index, { error: String((e as Error).message), running: undefined }); }
  }
  async function runAll() {
    setBusyAll(true);
    await pyRunner.reset().catch(() => {});
    for (const c of cells) {
      if (c.kind !== "code" && c.kind !== "task") continue;
      try { const r = await pyRunner.runCell(code(c)); setOutFor(c.index, { cell: r }); if (!r.ok) break; } catch (e) { setOutFor(c.index, { error: String((e as Error).message) }); break; }
    }
    setBusyAll(false);
  }

  const doneCount = groups.filter((g) => state[`done-${g.task}`] === "1").length;

  return (
    <div className="space-y-5">
      <div className="card sticky top-2 z-10 flex flex-wrap items-center gap-3 p-3 text-sm">
        <b className="num">{doneCount}/{groups.length}</b> {t("proj.tasksDone")}
        <div className="h-1.5 min-w-24 flex-1 rounded bg-surface2"><div className="h-1.5 rounded bg-good" style={{ width: `${groups.length ? (doneCount / groups.length) * 100 : 0}%` }} /></div>
        <button className="btn" onClick={runAll} disabled={busyAll}>{busyAll ? t("ex.running") : t("proj.runAll")}</button>
        <button className="btn" onClick={() => { pyRunner.reset().catch(() => {}); setOut({}); }}>{t("proj.restart")}</button>
        {py.state === "loading" && <span className="text-xs text-muted">{t("ex.loadingPy")}</span>}
      </div>

      {cells.map((c) => {
        const o = out[c.index] ?? {};
        if (c.kind === "md") return <Markdown key={c.index} source={c.title ? `## ${c.title}\n\n${c.body}` : c.body} />;
        if (c.kind === "check" || c.kind === "hint" || c.kind === "solution") return null;
        if (c.kind === "report")
          return (
            <section key={c.index} className="card p-4">
              <p className="mb-2 font-semibold">{t("proj.report")}</p>
              <div className="mb-2 text-sm text-muted"><Markdown source={c.body} /></div>
              <textarea className="input w-full" rows={5} value={state[`report-${c.index}`] ?? ""} onChange={(e) => put(`report-${c.index}`, e.target.value)} aria-label={t("proj.report")} />
            </section>
          );
        const isTask = c.kind === "task";
        const g = groupOf.get(c.index);
        const solved = state[`done-${c.index}`] === "1";
        return (
          <section key={c.index} className={`card space-y-3 p-4 ${isTask ? "border-l-4 !border-l-accent" : ""}`}>
            {isTask && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="chip !border-accent !text-accent">{t("proj.task")}</span>
                <h3 className="font-semibold">{c.title}</h3>
                {solved && <span className="chip !border-good !text-good">✓ {t("ex.solved")}</span>}
              </div>
            )}
            {!isTask && c.title && <p className="text-sm font-semibold text-muted">{c.title}</p>}
            <CodeEditor value={code(c)} onChange={(v) => put(isTask ? `task-${c.index}` : `code-${c.index}`, v)} onRun={() => (isTask ? check(c) : run(c))} rows={code(c).split("\n").length + 1} />
            <div className="flex flex-wrap items-center gap-2">
              <button className="btn" onClick={() => run(c)} disabled={!!o.running}>{o.running === "run" ? t("ex.running") : `▶ ${t("proj.run")}`}</button>
              {isTask && g && g.check !== null && <button className="btn btn-primary" onClick={() => check(c)} disabled={!!o.running}>{o.running === "check" ? t("ex.running") : t("proj.check")}</button>}
              {isTask && g && g.hints.length > 0 && <button className="btn" onClick={() => setHints((h) => ({ ...h, [c.index]: Math.min((h[c.index] ?? 0) + 1, g.hints.length) }))} disabled={(hints[c.index] ?? 0) >= g.hints.length}>{t("ex.hint", { n: hints[c.index] ?? 0, max: g.hints.length })}</button>}
              {isTask && g && g.solution !== null && <button className="btn" onClick={async () => { if (!confirm(t("ex.confirmSolution"))) return; const r = await getSolution(g.solution!); setSolutions((s) => ({ ...s, [c.index]: r })); }}>{t("ex.solution")}</button>}
            </div>
            {isTask && g && (hints[c.index] ?? 0) > 0 && <ol className="list-decimal space-y-1 pl-5 text-sm">{g.hints.slice(0, hints[c.index]).map((h) => <li key={h}>{cells[h].body}</li>)}</ol>}
            {o.error && <pre className="whitespace-pre-wrap rounded-lg border border-bad/50 bg-bad/10 p-2 text-xs">{o.error}</pre>}
            {o.cell?.stdout && <pre className="whitespace-pre-wrap rounded-lg border border-line bg-codebg p-2 text-xs">{o.cell.stdout}</pre>}
            {o.cell?.ok && o.cell.value != null && <pre className="whitespace-pre-wrap rounded-lg border border-line bg-codebg p-2 text-xs text-accent">{o.cell.value}</pre>}
            {o.cell && !o.cell.ok && <pre className="whitespace-pre-wrap rounded-lg border border-bad/50 bg-bad/10 p-2 text-xs">{o.cell.error}</pre>}
            {o.check && (
              <div className="text-sm" role="status">
                {o.check.phase !== "ok" && <pre className="whitespace-pre-wrap rounded-lg border border-bad/50 bg-bad/10 p-2 text-xs">{o.check.error}</pre>}
                <p className="font-semibold">{t("ex.passed", { p: o.check.results.filter((r) => r.passed).length, n: o.check.results.length })}</p>
                <ul className="space-y-0.5">{o.check.results.map((r, i) => <li key={i} className={r.passed ? "text-good" : "text-bad"}>{r.passed ? "✓" : "✗"} <span className="text-ink">{r.name}</span>{!r.passed && r.message ? <span className="text-muted"> — {r.message}</span> : null}</li>)}</ul>
              </div>
            )}
            {solutions[c.index] && <details open><summary className="cursor-pointer text-sm font-semibold">{t("ex.refSolution")}</summary><div className="mt-2"><CodeEditor value={solutions[c.index]} readOnly rows={solutions[c.index].split("\n").length} /></div></details>}
          </section>
        );
      })}
    </div>
  );
}
