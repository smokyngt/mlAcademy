import { describe, expect, test } from "bun:test";
import { HARNESS_PY } from "./harness";
import { parseCells, parseProject, taskGroups } from "./notebook";

describe("notebook cell grammar", () => {
  const src = [
    "# %% meta",
    '{"id":"p1","title":"T","summary":"S","level":"beginner","minutes":5}',
    "# %% md Intro",
    "Some text",
    "# %% task Build it",
    "Write f",
    "# %% hint",
    "Think",
    "# %% check",
    "check('ok', lambda: True)",
    "# %% solution",
    "def f(): pass",
    "# %% code Next",
    "print(1)",
  ].join("\n");

  test("cells are split on `# %%` lines and keep kind, title and trimmed body", () => {
    const cells = parseCells(src);
    expect(cells.map((c) => c.kind)).toEqual(["meta", "md", "task", "hint", "check", "solution", "code"]);
    expect(cells[1]).toEqual({ kind: "md", title: "Intro", body: "Some text" });
    expect(cells[6].body).toBe("print(1)");
  });

  test("a bare `# %%` is a code cell and an unknown kind stays part of the previous body", () => {
    const cells = parseCells("# %%\nx = 1\n# %% bogus y\nz = 2");
    expect(cells).toHaveLength(1);
    expect(cells[0].kind).toBe("code");
    expect(cells[0].body).toContain("# %% bogus y");
  });

  test("CRLF files parse the same as LF files", () => {
    expect(parseCells(src.replaceAll("\n", "\r\n"))).toEqual(parseCells(src));
  });

  test("text before the first marker is ignored", () => {
    expect(parseCells("junk\n# %% md\nhi")).toEqual([{ kind: "md", title: "", body: "hi" }]);
  });

  test("a project must start with a meta cell, and meta is split from the body", () => {
    const { meta, cells } = parseProject(src);
    expect(meta).toMatchObject({ id: "p1", level: "beginner", minutes: 5 });
    expect(cells[0].kind).toBe("md");
    expect(() => parseProject("# %% md\nhi")).toThrow(/meta/);
    expect(() => parseProject("")).toThrow();
  });

  test("task groups attach the check, hints and solution that follow a task", () => {
    const { cells } = parseProject(src);
    expect(taskGroups(cells)).toEqual([{ task: 1, check: 3, solution: 4, hints: [2] }]);
  });

  test("a task with nothing after it has an empty group; two tasks never share cells", () => {
    const cells = parseCells("# %% task a\nA\n# %% task b\nB\n# %% check\nc");
    expect(taskGroups(cells)).toEqual([
      { task: 0, check: null, solution: null, hints: [] },
      { task: 1, check: 2, solution: null, hints: [] },
    ]);
  });
});

describe("python harness (the same text runs in Pyodide, Docker and verification)", () => {
  const python = (code: string) => {
    const r = Bun.spawnSync(["python", "-I", "-c", `${HARNESS_PY}\nimport sys\n${code}`], { stderr: "pipe" });
    expect(r.stderr.toString()).toBe("");
    return JSON.parse(r.stdout.toString());
  };
  const exercise = (user: string, tests: string) => python(`print(run_exercise(${JSON.stringify(user)}, ${JSON.stringify(tests)}))`);

  test("passing and failing checks are reported one by one, with hints", () => {
    const r = exercise("def double(x): return 2 * x", "check('two', lambda: double(1) == 2)\ncheck('three', lambda: double(1) == 3, 'wrong value')");
    expect(r.phase).toBe("ok");
    expect(r.results).toEqual([
      { name: "two", passed: true, message: "" },
      { name: "three", passed: false, message: "wrong value" },
    ]);
  });

  test("an exception inside a check fails that check without stopping the others", () => {
    const r = exercise("", "check('boom', lambda: 1 / 0)\ncheck('fine', lambda: True)");
    expect(r.results[0]).toMatchObject({ name: "boom", passed: false });
    expect(r.results[0].message).toContain("ZeroDivisionError");
    expect(r.results[1].passed).toBe(true);
  });

  test("errors in the learner's code are phase user_error with captured stdout", () => {
    const r = exercise("print('before')\nraise ValueError('nope')", "check('x', lambda: True)");
    expect(r.phase).toBe("user_error");
    expect(r.stdout).toBe("before\n");
    expect(r.error).toContain("ValueError: nope");
    expect(r.results).toEqual([]);
  });

  test("errors in the tests are phase test_error, not blamed on the learner", () => {
    const r = exercise("x = 1", "undefined_name");
    expect(r.phase).toBe("test_error");
    expect(r.error).toContain("NameError");
  });

  test("results do not leak between runs", () => {
    const out = python(`a = json.loads(run_exercise("", "check('a', lambda: True)"))\nb = json.loads(run_exercise("", "check('b', lambda: True)"))\nprint(json.dumps([len(a['results']), len(b['results']), b['results'][0]['name']]))`);
    expect(out).toEqual([1, 1, "b"]);
  });

  test("run_cell keeps state between cells and returns the last expression", () => {
    const out = python(`ns = {}\nprint(json.dumps([json.loads(run_cell("x = 20\\nprint('hi')", ns)), json.loads(run_cell("x + 22", ns)), json.loads(run_cell("1/0", ns))]))`);
    expect(out[0]).toEqual({ ok: true, stdout: "hi\n", value: null, error: "" });
    expect(out[1]).toMatchObject({ ok: true, value: "42" });
    expect(out[2].ok).toBe(false);
    expect(out[2].error).toContain("ZeroDivisionError");
  });

  test("run_checks evaluates a project check cell against the notebook namespace", () => {
    const out = python(`ns = {"answer": 7}\nprint(run_checks("check('is seven', lambda: answer == 7)", ns))`);
    expect(out.phase).toBe("ok");
    expect(out.results).toEqual([{ name: "is seven", passed: true, message: "" }]);
  });
});
