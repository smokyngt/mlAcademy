import type { ExerciseResult } from "@shared/types";

export interface CellResult { ok: boolean; stdout: string; value: string | null; error: string }

type Listener = (state: "loading" | "ready" | "error", detail?: string) => void;

/** Main-thread handle to the Pyodide worker. Enforces a wall-clock timeout by terminating the worker. */
class PyRunner {
  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private seq = 0;
  private listeners = new Set<Listener>();
  state: "idle" | "loading" | "ready" | "error" = "idle";
  detail = "";

  subscribe(fn: Listener) { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; }
  private emit(s: "loading" | "ready" | "error", d?: string) { this.state = s; this.detail = d ?? ""; this.listeners.forEach((l) => l(s, d)); }

  private start() {
    if (this.worker) return this.worker;
    this.emit("loading");
    const w = new Worker(new URL("./pyodide.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (ev) => {
      const m = ev.data as { id: number; ok: boolean; result?: any; error?: string };
      if (m.id === -1) { m.ok ? this.emit("ready") : this.emit("error", m.error); return; }
      const p = this.pending.get(m.id);
      if (!p) return;
      clearTimeout(p.timer);
      this.pending.delete(m.id);
      m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
    };
    w.onerror = (e) => this.emit("error", e.message);
    this.worker = w;
    return w;
  }

  private kill(reason: string) {
    this.worker?.terminate();
    this.worker = null;
    this.state = "idle";
    for (const [, p] of this.pending) { clearTimeout(p.timer); p.reject(new Error(reason)); }
    this.pending.clear();
  }

  private call<T>(payload: object, timeoutMs: number): Promise<T> {
    const w = this.start();
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      // The first call also has to download the interpreter, so the clock only starts counting generously.
      const timer = setTimeout(() => this.kill(`Timed out after ${timeoutMs / 1000}s. The interpreter was restarted (notebook variables are lost). Check for an infinite loop.`), timeoutMs + (this.state === "ready" ? 0 : 60_000));
      this.pending.set(id, { resolve, reject, timer });
      w.postMessage({ id, ...payload });
    });
  }

  runExercise(code: string, tests: string): Promise<ExerciseResult> {
    return this.call({ type: "exercise", code, tests }, 20_000);
  }
  runCell(code: string): Promise<CellResult> {
    return this.call({ type: "cell", code }, 30_000);
  }
  /** Runs a project check cell inside the notebook namespace (after the learner ran their task cell). */
  runChecks(code: string): Promise<ExerciseResult> {
    return this.call({ type: "checks", code }, 20_000);
  }
  reset(): Promise<unknown> {
    return this.call({ type: "reset" }, 10_000);
  }
}

export const pyRunner = new PyRunner();
