import { useEffect, useRef, useState } from "react";
import type { ExercisePublic, ExerciseResult } from "@shared/types";
import { api } from "../api";
import { Markdown } from "./Markdown";
import { pyRunner } from "../runner/pyodide";
import { useT } from "../i18n";

export function usePyState() {
  const [s, setS] = useState({ state: pyRunner.state, detail: pyRunner.detail });
  useEffect(() => pyRunner.subscribe((state, detail) => setS({ state, detail: detail ?? "" })), []);
  return s;
}

import { CodeEditor } from "./CodeEditor";
export { CodeEditor };

export function ExerciseCard({ ex, onSolved }: { ex: ExercisePublic; onSolved?: () => void }) {
  const { t } = useT();
  const [code, setCode] = useState(ex.savedCode ?? ex.starter);
  const [result, setResult] = useState<ExerciseResult | null>(null);
  const [msg, setMsg] = useState("");
  const [running, setRunning] = useState(false);
  const [hints, setHints] = useState<string[]>([]);
  const [solution, setSolution] = useState<string | null>(null);
  const [solved, setSolved] = useState(ex.solved);
  const [runner, setRunner] = useState<{ available: boolean; reason: string } | null>(null);
  const py = usePyState();
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => { if (ex.hintsUsed) api<{ hints: string[] }>(`/api/exercises/${ex.id}/hints`).then((r) => setHints(r.hints)); }, [ex.id]);
  useEffect(() => { if (ex.runtime === "docker") api<{ available: boolean; reason: string }>("/api/runner/status").then(setRunner); }, [ex.runtime]);

  const edit = (v: string) => {
    setCode(v);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => api(`/api/exercises/${ex.id}/code`, { method: "PUT", body: { code: v } }).catch(() => {}), 800);
  };

  async function run() {
    setRunning(true); setMsg(""); setResult(null);
    try {
      let res: ExerciseResult;
      if (ex.runtime === "pyodide") {
        res = await pyRunner.runExercise(code, ex.tests);
        const passed = res.phase === "ok" && res.results.length > 0 && res.results.every((r) => r.passed);
        const out = await api<{ firstSolve: boolean }>(`/api/exercises/${ex.id}/attempt`, { body: { code, passed, output: JSON.stringify(res) } });
        if (passed) { setSolved(true); if (out.firstSolve) onSolved?.(); }
      } else {
        const r = await api<{ result?: ExerciseResult; passed?: boolean; firstSolve?: boolean; unavailable?: string }>(`/api/exercises/${ex.id}/run`, { body: { code } });
        if (r.unavailable) { setMsg(r.unavailable); setRunning(false); return; }
        res = r.result!;
        if (r.passed) { setSolved(true); if (r.firstSolve) onSolved?.(); }
      }
      setResult(res);
    } catch (e) { setMsg(String((e as Error).message)); }
    setRunning(false);
  }

  async function hint() {
    const r = await api<{ hints: string[] }>(`/api/exercises/${ex.id}/hint`, { method: "POST", body: {} });
    setHints(r.hints);
  }
  async function reveal() {
    if (!confirm(t("ex.confirmSolution"))) return;
    const r = await api<{ solution: string }>(`/api/exercises/${ex.id}/solution`, { method: "POST", body: {} });
    setSolution(r.solution);
  }

  const total = result?.results.length ?? 0;
  const passedN = result?.results.filter((r) => r.passed).length ?? 0;
  const dockerDown = ex.runtime === "docker" && runner && !runner.available;

  return (
    <section className="card p-4" aria-label={ex.title}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold">{ex.title}</h3>
        <span className="chip">{t("ex.difficulty")} {"★".repeat(ex.difficulty)}{"☆".repeat(5 - ex.difficulty)}</span>
        <span className="chip">{ex.runtime === "pyodide" ? t("ex.numpy") : t("ex.torch")}</span>
        {solved && <span className="chip !border-good !text-good">{t("ex.solved")}</span>}
      </div>
      <Markdown source={ex.prompt} />
      {dockerDown && <p className="my-2 rounded-md border border-warn/50 bg-warn/10 p-2 text-sm">{t("ex.runnerDown", { reason: runner!.reason })}</p>}
      <CodeEditor value={code} onChange={edit} onRun={run} />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" onClick={run} disabled={running || !!dockerDown}>{running ? t("ex.running") : t("ex.run")}</button>
        <button className="btn" onClick={hint} disabled={hints.length >= ex.hintCount}>{t("ex.hint", { n: hints.length, max: ex.hintCount })}</button>
        <button className="btn" onClick={reveal}>{t("ex.solution")}</button>
        <button className="btn" onClick={() => { if (confirm(t("ex.confirmReset"))) edit(ex.starter); }}>{t("ex.reset")}</button>
        {ex.runtime === "pyodide" && py.state === "loading" && <span className="text-xs text-muted">{t("ex.loadingPy")}</span>}
        {ex.runtime === "pyodide" && py.state === "error" && <span className="text-xs text-bad">{py.detail}</span>}
      </div>
      {msg && <p className="mt-2 text-sm text-bad" role="alert">{msg}</p>}
      {hints.length > 0 && <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm">{hints.map((h, i) => <li key={i}>{h}</li>)}</ol>}
      {result && (
        <div className="mt-3 space-y-2 text-sm" role="status">
          {result.phase === "user_error" && <pre className="whitespace-pre-wrap rounded-lg border border-bad/50 bg-bad/10 p-3 text-xs">{result.error}</pre>}
          {result.phase === "test_error" && <pre className="whitespace-pre-wrap rounded-lg border border-bad/50 bg-bad/10 p-3 text-xs">{t("ex.harnessError")}{"\n"}{result.error}</pre>}
          {result.stdout && <pre className="whitespace-pre-wrap rounded-lg border border-line bg-codebg p-3 text-xs">{result.stdout}</pre>}
          {total > 0 && (
            <>
              <p className="font-semibold">{t("ex.passed", { p: passedN, n: total })}{passedN === total && result.phase === "ok" ? " 🎉" : ""}</p>
              <ul className="space-y-1">
                {result.results.map((r, i) => (
                  <li key={i} className={r.passed ? "text-good" : "text-bad"}>
                    {r.passed ? "✓" : "✗"} <span className="text-ink">{r.name}</span>{!r.passed && r.message ? <span className="text-muted"> — {r.message}</span> : null}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
      {solution && (
        <details open className="mt-3">
          <summary className="cursor-pointer text-sm font-semibold">{t("ex.refSolution")}</summary>
          <pre className="mt-2 overflow-x-auto rounded-lg border border-line bg-codebg p-3 text-xs">{solution}</pre>
        </details>
      )}
    </section>
  );
}
