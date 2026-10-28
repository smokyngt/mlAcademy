/**
 * Runs every project notebook end to end with the reference solutions substituted for the tasks, and checks that
 *   - every task's checks pass with the solution,
 *   - every task's starter code FAILS its checks (so a starter can never be accidentally complete).
 *   bun scripts/verify-projects.ts [project-id ...]
 * Project code is trusted content authored for this repo and runs with the local `python -I`.
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { HARNESS_PY } from "../src/shared/harness";
import { parseProject, taskGroups } from "../src/shared/notebook";

const dir = path.resolve(import.meta.dir, "..", "content", "projects");
const only = process.argv.slice(2);
const py = process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");
let failures = 0;
const fail = (m: string) => { failures++; console.error("FAIL", m); };

const DRIVER = `
import json, sys, time, io, contextlib, traceback
hns = {}
exec(compile(HARNESS, "<harness>", "exec"), hns)
ns = {"__name__": "__main__", "check": hns["check"]}
results = hns["__results__"]
out = {"tasks": [], "errors": []}

def run(code, label):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, label, "exec"), ns)
        return None
    except BaseException as e:
        return type(e).__name__ + ": " + str(e)

def run_checks(code, label):
    del results[:]
    err = run(code, label)
    return err, [dict(r) for r in results]

for step in STEPS:
    t0 = time.time()
    if step["kind"] == "code":
        err = run(step["code"], step["label"])
        if err: out["errors"].append(step["label"] + ": " + err)
    else:
        rec = {"label": step["label"], "starter": None, "solution": None, "seconds": 0}
        err = run(step["starter"], step["label"] + " (starter)")
        _, res = run_checks(step["check"], step["label"] + " (check on starter)")
        rec["starter"] = {"passed": sum(1 for r in res if r["passed"]), "total": len(res), "error": err}
        err = run(step["solution"], step["label"] + " (solution)")
        if err: out["errors"].append(step["label"] + " solution: " + err)
        cerr, res = run_checks(step["check"], step["label"] + " (check on solution)")
        rec["solution"] = {"passed": sum(1 for r in res if r["passed"]), "total": len(res), "error": cerr,
                           "failed": [r["name"] + " :: " + r["message"] for r in res if not r["passed"]]}
        rec["seconds"] = round(time.time() - t0, 2)
        out["tasks"].append(rec)
print("@@RESULT@@" + json.dumps(out))
`;

let n = 0;
for (const f of fs.existsSync(dir) ? fs.readdirSync(dir).filter((x) => x.endsWith(".project.py")).sort() : []) {
  const { meta, cells } = parseProject(fs.readFileSync(path.join(dir, f), "utf8"));
  if (only.length && !only.includes(meta.id)) continue;
  n++;
  const groups = taskGroups(cells);
  const taskIdx = new Set(groups.map((g) => g.task));
  const steps: object[] = [];
  const solved = new Set<number>();
  cells.forEach((c, i) => {
    if (c.kind === "code") steps.push({ kind: "code", code: c.body, label: `cell ${i} (${c.title || "code"})` });
    else if (taskIdx.has(i)) {
      const g = groups.find((x) => x.task === i)!;
      if (g.check === null) { fail(`${meta.id}: task "${c.title}" has no check cell`); return; }
      if (g.solution === null) { fail(`${meta.id}: task "${c.title}" has no solution cell`); return; }
      if ((cells[g.check].body.match(/\bcheck\(/g) ?? []).length < 3) fail(`${meta.id}: task "${c.title}" needs >= 3 checks`);
      steps.push({ kind: "task", label: `task ${i} (${c.title})`, starter: c.body, check: cells[g.check].body, solution: cells[g.solution].body });
      solved.add(i);
    }
  });
  const script = `HARNESS = ${JSON.stringify(HARNESS_PY)}\nSTEPS = ${JSON.stringify(steps)}\n${DRIVER}`;
  const tmp = path.join(os.tmpdir(), `proj-${meta.id.replace(/\W/g, "_")}.py`);
  fs.writeFileSync(tmp, script);
  const t0 = Date.now();
  const r = spawnSync(py, ["-I", tmp], { encoding: "utf8", timeout: 300_000 });
  fs.rmSync(tmp, { force: true });
  const line = (r.stdout ?? "").split("\n").find((l) => l.startsWith("@@RESULT@@"));
  if (!line) { fail(`${meta.id}: crashed: ${(r.stderr ?? "").slice(-500)}`); continue; }
  const out = JSON.parse(line.slice("@@RESULT@@".length)) as { tasks: any[]; errors: string[] };
  out.errors.forEach((e) => fail(`${meta.id}: ${e}`));
  let checks = 0;
  for (const t of out.tasks) {
    checks += t.solution.total;
    if (t.solution.error || t.solution.passed !== t.solution.total || t.solution.total === 0) fail(`${meta.id} ${t.label}: solution ${t.solution.passed}/${t.solution.total} ${t.solution.error ?? ""} ${t.solution.failed.join("; ")}`);
    if (t.starter.total > 0 && t.starter.passed === t.starter.total) fail(`${meta.id} ${t.label}: the starter already passes every check`);
    if (t.seconds > 8) fail(`${meta.id} ${t.label}: took ${t.seconds}s (limit 8s)`);
  }
  const total = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`${out.errors.length || out.tasks.some((t: any) => t.solution.passed !== t.solution.total) ? "✗" : "✓"} ${meta.id}: ${out.tasks.length} tasks, ${checks} checks, ${total}s`);
}
console.log(`\n${n} project(s) checked, ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
