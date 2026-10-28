import { describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";

/**
 * The academy must stand on its own: nothing in it may import code or data from another folder of the repository.
 * (The repository root is the parent of the academy folder.)
 */
const ROOT = path.resolve(import.meta.dir, "..");
const SKIP = new Set(["node_modules", "dist", "generated", ".git", "data", "__pycache__"]);
const CODE = /\.(ts|tsx|js|mjs|cjs)$/;

function walk(dir: string, out: string[] = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(e.name)) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

const codeFiles = walk(ROOT).filter((f) => CODE.test(f));
const rel = (f: string) => path.relative(ROOT, f).replaceAll("\\", "/");

/** Every module specifier in a source file: static imports, re-exports, dynamic import() and require(). */
function specifiers(src: string) {
  const out: string[] = [];
  const re = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)["']([^"']+)["']/g;
  for (let m; (m = re.exec(src)); ) out.push(m[1]);
  return out;
}

describe("academy isolation: static imports", () => {
  test("the scan really covers the app", () => {
    expect(codeFiles.length).toBeGreaterThan(40);
    expect(codeFiles.some((f) => rel(f) === "src/server/app.ts")).toBe(true);
    expect(codeFiles.some((f) => rel(f) === "src/web/App.tsx")).toBe(true);
    expect(codeFiles.some((f) => rel(f) === "scripts/build-catalog.ts")).toBe(true);
  });

  test("no relative import resolves outside the academy folder", () => {
    const escaping: string[] = [];
    for (const f of codeFiles) {
      for (const spec of specifiers(fs.readFileSync(f, "utf8"))) {
        if (!spec.startsWith(".")) continue;
        const target = path.resolve(path.dirname(f), spec);
        if (path.relative(ROOT, target).startsWith("..")) escaping.push(`${rel(f)} -> ${spec}`);
      }
    }
    expect(escaping).toEqual([]);
  });

  test("no import uses an absolute path or reaches into a sibling folder by name", () => {
    const siblings = fs.readdirSync(path.dirname(ROOT)).filter((n) => n !== path.basename(ROOT) && !n.startsWith("."));
    const offenders: string[] = [];
    for (const f of codeFiles) {
      for (const spec of specifiers(fs.readFileSync(f, "utf8"))) {
        if (path.isAbsolute(spec) || /^[A-Za-z]:[\\/]/.test(spec)) offenders.push(`${rel(f)} -> ${spec}`);
        if (siblings.some((s) => spec === s || spec.startsWith(s + "/"))) offenders.push(`${rel(f)} -> ${spec}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  test("every bare import resolves to the academy's own node_modules", () => {
    const builtin = (s: string) => s.startsWith("node:") || s.startsWith("bun:") || s === "bun";
    const outside: string[] = [];
    for (const f of codeFiles) {
      for (const spec of specifiers(fs.readFileSync(f, "utf8"))) {
        if (spec.startsWith(".") || builtin(spec) || spec.startsWith("@shared/")) continue;
        let resolved: string;
        try { resolved = Bun.resolveSync(spec, path.dirname(f)); } catch { outside.push(`${spec} (${rel(f)}): not installed`); continue; }
        if (path.relative(ROOT, resolved).startsWith("..")) outside.push(`${spec} (${rel(f)}): resolved to ${resolved}`);
      }
    }
    expect(outside).toEqual([]);
  });

  test("tsconfig, vite and drizzle configs stay inside the academy", () => {
    const tsconfig = fs.readFileSync(path.join(ROOT, "tsconfig.json"), "utf8");
    expect(tsconfig).not.toMatch(/\.\.\//);
    expect(fs.readFileSync(path.join(ROOT, "drizzle.config.ts"), "utf8")).not.toMatch(/["']\.\.\//);
    // vite builds from src/web, so its outDir "../../dist" is the academy's own dist folder
    const vite = fs.readFileSync(path.join(ROOT, "vite.config.ts"), "utf8");
    const outDir = vite.match(/outDir:\s*["']([^"']+)["']/)![1];
    expect(path.resolve(ROOT, "src/web", outDir)).toBe(path.join(ROOT, "dist"));
  });
});

describe("academy isolation: runtime reads of the surrounding repository", () => {
  /**
   * Not imports, but file-system reads of sibling folders. They are known and deliberate; this list is the
   * whole allowance, so any new place that touches the parent folder fails here until it is reviewed.
   */
  const ALLOWED = [
    "scripts/build-catalog.ts", // opt-in only (IMPORT_SOURCES=1): imports library notebooks as "source" courses
    "src/server/app.ts", // /api/sources: checks that cited library paths exist (a boolean, never reads their content)
  ];

  test("only the known files resolve the parent of the academy folder", () => {
    const touching = codeFiles
      .filter((f) => !f.endsWith(".test.ts") && !f.endsWith(".test.tsx"))
      .filter((f) => /ACADEMY_ROOT,\s*["']\.\.["']/.test(fs.readFileSync(f, "utf8")))
      .map(rel)
      .sort();
    expect(touching).toEqual([...ALLOWED].sort());
  });

  test("importing sibling content is opt-in: without IMPORT_SOURCES the loaders are never called", () => {
    const src = fs.readFileSync(path.join(ROOT, "scripts", "build-catalog.ts"), "utf8");
    expect(src).toContain('process.env.IMPORT_SOURCES === "1"');
    expect(src).toMatch(/if \(IMPORT\) loadAiEngineering\(\)/);
    expect(src.match(/if \(IMPORT\) loadNotebookBook/g)?.length).toBe(2);
    expect(src).not.toMatch(/^loadAiEngineering\(\)/m);
  });

  test("the authored catalog is self-contained: no lesson, quiz or exercise comes from a sibling folder", async () => {
    const dbFile = path.join(fs.mkdtempSync(path.join(require("node:os").tmpdir(), "academy-iso-")), "iso.db");
    const r = Bun.spawnSync(["bun", path.join(ROOT, "scripts", "build-catalog.ts")], { env: { ...process.env, ACADEMY_DB: dbFile, IMPORT_SOURCES: "0" }, cwd: ROOT });
    expect(r.exitCode).toBe(0);
    const { Database } = await import("bun:sqlite");
    const db = new Database(dbFile, { readonly: true });
    expect(db.query("SELECT COUNT(*) AS n FROM courses WHERE kind <> 'authored'").get()).toEqual({ n: 0 });
    expect(db.query("SELECT COUNT(*) AS n FROM lessons WHERE kind <> 'authored'").get()).toEqual({ n: 0 });
    expect((db.query("SELECT COUNT(*) AS n FROM lessons").get() as { n: number }).n).toBeGreaterThan(0);
    db.close();
  });
});
