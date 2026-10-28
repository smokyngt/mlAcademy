/**
 * Content verification: every reference solution must pass its tests, every starter must fail them,
 * and every [[lesson-id]] / ```viz``` reference in lesson bodies must resolve.
 *
 *   bun run verify:content
 *
 * Exercise code is trusted content written for this repo, so it runs with the local Python (`python -I`).
 * Learner code never goes through this path.
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { HARNESS_PY } from "../src/shared/harness";
import { ACADEMY_ROOT } from "../src/server/db";
import { VIZ_NAMES } from "../src/shared/viz-names";

const authored = path.join(ACADEMY_ROOT, "content", "authored");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "academy-verify-"));
const py = process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");

function parseSections(src: string) {
  const parts: Record<string, string> = {};
  let cur = "";
  for (const line of src.replace(/\r\n/g, "\n").split("\n")) {
    const m = line.match(/^#@@ (\w+)\s*$/);
    if (m) { cur = m[1]; parts[cur] = ""; } else if (cur) parts[cur] += line + "\n";
  }
  return parts;
}

function runPython(userCode: string, tests: string, label: string) {
  const script = path.join(tmp, `${label.replace(/[^\w.-]/g, "_")}.py`);
  fs.writeFileSync(
    script,
    `${HARNESS_PY}\nimport json as _j\nprint("@@RESULT@@" + run_exercise(${JSON.stringify(userCode)}, ${JSON.stringify(tests)}))\n`,
  );
  const r = spawnSync(py, ["-I", script], { encoding: "utf8", timeout: 120_000 });
  const line = (r.stdout ?? "").split("\n").find((l) => l.startsWith("@@RESULT@@"));
  if (!line) return { phase: "crash", error: (r.stderr ?? "").slice(-400), results: [] as any[] };
  return JSON.parse(line.slice("@@RESULT@@".length)) as { phase: string; error: string; results: { name: string; passed: boolean; message: string }[] };
}

let failures = 0;
const fail = (m: string) => { failures++; console.error("FAIL", m); };

const ids = new Set<string>();
// Link targets are checked against EVERY course even when ONLY narrows which exercises run.
for (const c of fs.readdirSync(authored)) {
  const cd = path.join(authored, c);
  if (!fs.statSync(cd).isDirectory()) continue;
  for (const l of fs.readdirSync(cd)) {
    const f = path.join(cd, l, "lesson.md");
    if (fs.existsSync(f)) { const m = fs.readFileSync(f, "utf8").match(/^id:\s*(\S+)/m); if (m) ids.add(m[1]); }
  }
}
const bodies: { id: string; body: string }[] = [];
let exerciseCount = 0;

for (const course of fs.readdirSync(authored).filter((c) => !process.env.ONLY || c === process.env.ONLY)) {
  if (!fs.statSync(path.join(authored, course)).isDirectory()) continue;
  for (const lesson of fs.readdirSync(path.join(authored, course)).filter((d) => fs.existsSync(path.join(authored, course, d, "lesson.md"))).sort()) {
    const dir = path.join(authored, course, lesson);
    const md = fs.readFileSync(path.join(dir, "lesson.md"), "utf8");
    const id = md.match(/^id:\s*(\S+)/m)?.[1];
    if (!id) { fail(`${lesson}: no id`); continue; }
    ids.add(id);
    bodies.push({ id, body: md });
    const exDir = path.join(dir, "exercises");
    if (!fs.existsSync(exDir)) continue;
    for (const f of fs.readdirSync(exDir).filter((x) => x.endsWith(".exercise.txt"))) {
      exerciseCount++;
      const s = parseSections(fs.readFileSync(path.join(exDir, f), "utf8"));
      const meta = JSON.parse(s.meta);
      const sol = runPython(s.solution, s.tests, `${meta.id}-solution`);
      const bad = sol.results.filter((r) => !r.passed);
      if (sol.phase !== "ok" || bad.length || sol.results.length === 0)
        fail(`${meta.id} solution: phase=${sol.phase} ${sol.error ?? ""} ${bad.map((b) => `${b.name} (${b.message})`).join("; ")}`);
      const st = runPython(s.starter, s.tests, `${meta.id}-starter`);
      if (st.phase === "ok" && st.results.length > 0 && st.results.every((r) => r.passed)) fail(`${meta.id}: the starter already passes all tests`);
      console.log(`${bad.length || sol.phase !== "ok" ? "✗" : "✓"} ${meta.id}  solution ${sol.results.length - bad.length}/${sol.results.length}  starter ${st.phase === "ok" ? st.results.filter((r) => r.passed).length : "error"}/${st.results.length}`);
    }
  }
}

for (const { id, body } of bodies) {
  for (const m of body.matchAll(/\[\[([\w.-]+)\]\]/g)) if (!ids.has(m[1])) fail(`${id}: broken lesson link [[${m[1]}]]`);
  for (const m of body.matchAll(/```viz\s*\n([\w-]+)\s*\n```/g)) if (!VIZ_NAMES.includes(m[1])) fail(`${id}: unknown viz "${m[1]}"`);
  const open = (body.match(/```/g) ?? []).length;
  if (open % 2) fail(`${id}: unbalanced code fences`);
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\n${ids.size} lessons, ${exerciseCount} exercises checked, ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
