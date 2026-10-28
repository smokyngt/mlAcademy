/**
 * Server-side code execution in a locked-down Docker container.
 *
 * User code is NEVER executed in this process. Each run gets a throwaway container with:
 *   no network, read-only root filesystem, 16 MB tmpfs scratch, 1 CPU, 768 MB RAM (no swap), 128 pids,
 *   all capabilities dropped, no-new-privileges, an unprivileged uid, and a hard wall-clock timeout.
 * The job directory (code + harness) is mounted read-only.
 *
 * Build the image once:  docker build -t ml-academy-runner src/server/runner
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { HARNESS_PY } from "../../shared/harness";
import type { ExerciseResult } from "../../shared/types";

export const RUNNER_IMAGE = process.env.ACADEMY_RUNNER_IMAGE ?? "ml-academy-runner";
const TIMEOUT_MS = Number(process.env.ACADEMY_RUN_TIMEOUT_MS ?? 30_000);
const MAX_OUTPUT = 1_000_000;

function exec(cmd: string, args: string[], timeoutMs: number): Promise<{ code: number | null; stdout: string; stderr: string; timedOut: boolean }> {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { windowsHide: true });
    let stdout = "", stderr = "", timedOut = false;
    const timer = setTimeout(() => { timedOut = true; p.kill("SIGKILL"); }, timeoutMs);
    p.stdout.on("data", (d) => { if (stdout.length < MAX_OUTPUT) stdout += d; });
    p.stderr.on("data", (d) => { if (stderr.length < MAX_OUTPUT) stderr += d; });
    p.on("error", (e) => { clearTimeout(timer); resolve({ code: -1, stdout, stderr: String(e), timedOut }); });
    p.on("close", (code) => { clearTimeout(timer); resolve({ code, stdout, stderr, timedOut }); });
  });
}

export interface RunnerStatus {
  available: boolean;
  reason: string;
  image: string;
}

export async function runnerStatus(): Promise<RunnerStatus> {
  const v = await exec("docker", ["version", "--format", "{{.Server.Version}}"], 5000);
  if (v.code !== 0 || !v.stdout.trim())
    return { available: false, image: RUNNER_IMAGE, reason: "The Docker daemon is not reachable. Start Docker Desktop (or the docker service), then reload." };
  const i = await exec("docker", ["image", "inspect", RUNNER_IMAGE, "--format", "{{.Id}}"], 5000);
  if (i.code !== 0)
    return { available: false, image: RUNNER_IMAGE, reason: `Docker is running but the image "${RUNNER_IMAGE}" is not built. Run: docker build -t ${RUNNER_IMAGE} src/server/runner` };
  return { available: true, image: RUNNER_IMAGE, reason: "ok" };
}

async function runJob(files: Record<string, string>, entry: string): Promise<{ unavailable: string } | { stdout: string; stderr: string; timedOut: boolean; code: number | null }> {
  const st = await runnerStatus();
  if (!st.available) return { unavailable: st.reason };
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "academy-job-"));
  const name = `academy-${crypto.randomBytes(6).toString("hex")}`;
  try {
    for (const [f, content] of Object.entries(files)) fs.writeFileSync(path.join(dir, f), content);
    const args = [
      "run", "--rm", "--name", name,
      "--network", "none",
      "--read-only", "--tmpfs", "/tmp:rw,size=16m",
      "--memory", "768m", "--memory-swap", "768m", "--cpus", "1", "--pids-limit", "128",
      "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
      "--user", "65534:65534",
      "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=1",
      "-v", `${dir}:/job:ro`,
      RUNNER_IMAGE, "python", "-I", `/job/${entry}`,
    ];
    const r = await exec("docker", args, TIMEOUT_MS);
    if (r.timedOut) await exec("docker", ["kill", name], 5000);
    return r;
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

export async function runInDocker(job: { kind: "exercise"; code: string; tests: string }): Promise<ExerciseResult | { unavailable: string }> {
  const runner = `${HARNESS_PY}\nimport sys\nprint("@@RESULT@@" + run_exercise(open("/job/user.py").read(), open("/job/tests.py").read()))\n`;
  const r = await runJob({ "runner.py": runner, "user.py": job.code, "tests.py": job.tests }, "runner.py");
  if ("unavailable" in r) return r;
  if (r.timedOut) return { phase: "user_error", stdout: "", error: `Timed out after ${TIMEOUT_MS / 1000}s (infinite loop or too much computation).`, results: [] };
  const line = r.stdout.split("\n").find((l) => l.startsWith("@@RESULT@@"));
  if (!line) return { phase: "user_error", stdout: r.stdout.slice(0, 5000), error: (r.stderr || "The runner produced no result (out of memory?).").slice(-4000), results: [] };
  return JSON.parse(line.slice("@@RESULT@@".length)) as ExerciseResult;
}

export interface CellOutput { ok: boolean; stdout: string; value: string | null; error: string }

/** Replays the cells in order in a single container; returns one output per cell. */
export async function runNotebookInDocker(cells: string[]): Promise<{ outputs: CellOutput[] } | { unavailable: string }> {
  const runner = `${HARNESS_PY}\nimport json\nns = {"__name__": "__main__"}\nouts = []\nfor src in json.load(open("/job/cells.json")):\n    r = json.loads(run_cell(src, ns))\n    outs.append(r)\n    if not r["ok"]:\n        break\nprint("@@RESULT@@" + json.dumps(outs))\n`;
  const r = await runJob({ "runner.py": runner, "cells.json": JSON.stringify(cells) }, "runner.py");
  if ("unavailable" in r) return r;
  if (r.timedOut) return { outputs: [{ ok: false, stdout: "", value: null, error: `Timed out after ${TIMEOUT_MS / 1000}s.` }] };
  const line = r.stdout.split("\n").find((l) => l.startsWith("@@RESULT@@"));
  if (!line) return { outputs: [{ ok: false, stdout: r.stdout.slice(0, 5000), value: null, error: (r.stderr || "No result produced.").slice(-4000) }] };
  return { outputs: JSON.parse(line.slice("@@RESULT@@".length)) as CellOutput[] };
}
