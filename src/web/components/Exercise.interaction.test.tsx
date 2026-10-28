import { dom } from "../testing/dom"; // must stay first: installs jsdom before react-dom/client loads
import { afterEach, beforeAll, beforeEach, describe, expect, mock, test } from "bun:test";
import { act } from "react";
import type { Root } from "react-dom/client";
import type { ExercisePublic, ExerciseResult } from "@shared/types";
import { buttonByText, click, flush, mockServer } from "../testing/helpers";

// CodeMirror needs a real layout engine; the exercise logic only needs something that holds the code and can be edited.
mock.module("./CodeEditor", () => ({
  CodeEditor: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => <textarea data-testid="editor" value={value} onChange={(e) => onChange(e.target.value)} />,
}));

const { createRoot } = await import("react-dom/client");
const { MemoryRouter } = await import("react-router-dom");
const { LangProvider } = await import("../i18n");
const { ExerciseCard } = await import("./Exercise");
const { pyRunner } = await import("../runner/pyodide");

const base: ExercisePublic = {
  id: "ex1", title: "Double it", difficulty: 2, runtime: "pyodide", prompt: "Write `double(x)`.", starter: "def double(x): pass",
  tests: "check('t', lambda: True)", hintCount: 2, conceptIds: [], savedCode: null, solved: false, hintsUsed: 0, solutionRevealed: false,
};
const pass: ExerciseResult = { phase: "ok", stdout: "", error: "", results: [{ name: "returns 2x", passed: true, message: "" }, { name: "negatives", passed: true, message: "" }] };
const fail: ExerciseResult = { phase: "ok", stdout: "debug line", error: "", results: [{ name: "returns 2x", passed: true, message: "" }, { name: "negatives", passed: false, message: "got 0" }] };

let root: Root;
let host: HTMLElement;
let server: ReturnType<typeof mockServer>;
const realRun = pyRunner.runExercise.bind(pyRunner);
let ran: { code: string; tests: string }[] = [];

function mount(ex: Partial<ExercisePublic> = {}, onSolved?: () => void) {
  act(() => { root.render(<MemoryRouter><LangProvider><ExerciseCard ex={{ ...base, ...ex }} onSolved={onSolved} /></LangProvider></MemoryRouter>); });
}
const usePyResult = (res: ExerciseResult | Error) => {
  (pyRunner as any).runExercise = async (code: string, tests: string) => { ran.push({ code, tests }); if (res instanceof Error) throw res; return res; };
};
const run = () => buttonByText(host, /^run|exécuter|lancer/i);
const editor = () => host.querySelector("textarea") as HTMLTextAreaElement;

beforeAll(() => { dom.window.localStorage.setItem("academy.lang", "en"); (globalThis as any).confirm = () => true; });
beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); ran = []; });
afterEach(() => { act(() => root.unmount()); host.remove(); server?.restore(); (pyRunner as any).runExercise = realRun; });

describe("pyodide exercise", () => {
  test("starts from the starter code, or from the learner's saved code when there is some", () => {
    mount();
    expect(editor().value).toBe("def double(x): pass");
    act(() => root.unmount()); root = createRoot(host);
    mount({ savedCode: "def double(x): return 2*x" });
    expect(editor().value).toBe("def double(x): return 2*x");
  });

  test("running sends the code to the interpreter and records a passing attempt; the first solve notifies the lesson", async () => {
    server = mockServer(() => ({ json: { firstSolve: true } }));
    usePyResult(pass);
    let solved = 0;
    mount({}, () => solved++);
    click(run());
    await flush();
    expect(ran).toEqual([{ code: "def double(x): pass", tests: "check('t', lambda: True)" }]);
    const attempt = server.requests.find((r) => r.url.startsWith("/api/exercises/ex1/attempt"))!;
    expect(attempt.method).toBe("POST");
    expect(attempt.body).toMatchObject({ code: "def double(x): pass", passed: true });
    expect(solved).toBe(1);
    expect(host.textContent).toContain("returns 2x");
    expect(host.querySelector("[role=status]")!.textContent).toContain("🎉");
    expect(host.textContent).toMatch(/solved|résolu/i);
  });

  test("a later solve does not notify the lesson again", async () => {
    server = mockServer(() => ({ json: { firstSolve: false } }));
    usePyResult(pass);
    let solved = 0;
    mount({ solved: true }, () => solved++);
    click(run());
    await flush();
    expect(solved).toBe(0);
  });

  test("a failing test is listed with its message, is recorded as not passed, and is not marked solved", async () => {
    server = mockServer(() => ({ json: { firstSolve: false } }));
    usePyResult(fail);
    mount();
    click(run());
    await flush();
    const status = host.querySelector("[role=status]")!;
    expect(status.textContent).toContain("got 0");
    expect(status.textContent).toContain("debug line");
    expect(status.textContent).not.toContain("🎉");
    expect(server.requests.find((r) => r.url.includes("/attempt"))!.body.passed).toBe(false);
    expect(host.textContent).not.toMatch(/\bsolved\b/i);
  });

  test("an exception in the learner's code is shown and counts as a failed attempt", async () => {
    server = mockServer(() => ({ json: { firstSolve: false } }));
    usePyResult({ phase: "user_error", stdout: "", error: "NameError: name 'y' is not defined", results: [] });
    mount();
    click(run());
    await flush();
    expect(host.querySelector("pre")!.textContent).toContain("NameError");
    expect(server.requests.find((r) => r.url.includes("/attempt"))!.body.passed).toBe(false);
  });

  test("a harness failure is not blamed on the learner and never counts as passed", async () => {
    server = mockServer(() => ({ json: { firstSolve: false } }));
    usePyResult({ phase: "test_error", stdout: "", error: "KeyError: 'x'", results: [] });
    mount();
    click(run());
    await flush();
    expect(host.querySelector("pre")!.textContent).toContain("KeyError");
    expect(server.requests.find((r) => r.url.includes("/attempt"))!.body.passed).toBe(false);
  });

  test("an interpreter timeout is surfaced as an alert and the Run button comes back", async () => {
    server = mockServer(() => ({ json: {} }));
    usePyResult(new Error("Timed out after 20s."));
    mount();
    click(run());
    await flush();
    expect(host.querySelector("[role=alert]")!.textContent).toContain("Timed out");
    expect(run().disabled).toBe(false);
    expect(server.requests.some((r) => r.url.includes("/attempt"))).toBe(false);
  });

  test("editing the code is autosaved to the server after a pause", async () => {
    server = mockServer(() => ({ json: { ok: true } }));
    mount();
    act(() => {
      Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value")!.set!.call(editor(), "def double(x): return x*2");
      editor().dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    });
    expect(server.requests).toHaveLength(0); // debounced
    await act(async () => { await new Promise((r) => setTimeout(r, 900)); });
    expect(server.requests).toHaveLength(1);
    expect(server.requests[0]).toMatchObject({ method: "PUT", body: { code: "def double(x): return x*2" } });
    expect(server.requests[0].url).toStartWith("/api/exercises/ex1/code");
  });
});

describe("hints, solution and reset", () => {
  test("hints are revealed one by one and the button stops at the maximum", async () => {
    let n = 0;
    server = mockServer(() => ({ json: { hints: ["first", "second"].slice(0, ++n) } }));
    mount();
    const hintBtn = () => buttonByText(host, /hint|indice/i);
    click(hintBtn()); await flush();
    expect([...host.querySelectorAll("ol li")].map((l) => l.textContent)).toEqual(["first"]);
    expect(hintBtn().disabled).toBe(false);
    click(hintBtn()); await flush();
    expect([...host.querySelectorAll("ol li")].map((l) => l.textContent)).toEqual(["first", "second"]);
    expect(hintBtn().disabled).toBe(true);
  });

  test("the solution is only requested after the learner confirms", async () => {
    server = mockServer(() => ({ json: { solution: "def double(x): return 2 * x" } }));
    (globalThis as any).confirm = () => false;
    mount();
    click(buttonByText(host, /solution/i)); await flush();
    expect(server.requests).toHaveLength(0);
    expect(host.querySelector("details")).toBeNull();
    (globalThis as any).confirm = () => true;
    click(buttonByText(host, /solution/i)); await flush();
    expect(server.requests[0].url).toStartWith("/api/exercises/ex1/solution");
    expect(host.querySelector("details pre")!.textContent).toBe("def double(x): return 2 * x");
  });

  test("reset restores the starter code only after confirmation", async () => {
    server = mockServer(() => ({ json: { ok: true } }));
    mount({ savedCode: "my work" });
    (globalThis as any).confirm = () => false;
    click(buttonByText(host, /reset|réinitialiser/i));
    expect(editor().value).toBe("my work");
    (globalThis as any).confirm = () => true;
    click(buttonByText(host, /reset|réinitialiser/i));
    expect(editor().value).toBe(base.starter);
  });
});

describe("docker exercise", () => {
  test("when the runner is unavailable the reason is shown and Run is disabled", async () => {
    server = mockServer(() => ({ json: { available: false, reason: "docker not installed" } }));
    mount({ runtime: "docker" });
    await flush();
    expect(host.textContent).toContain("docker not installed");
    expect(run().disabled).toBe(true);
  });

  test("a passing server-side run marks it solved without touching the in-browser interpreter", async () => {
    server = mockServer((r) => r.url.includes("/runner/status") ? { json: { available: true, reason: "" } } : { json: { result: pass, passed: true, firstSolve: true } });
    usePyResult(new Error("must not be called"));
    let solved = 0;
    mount({ runtime: "docker" }, () => solved++);
    await flush();
    click(run());
    await flush();
    expect(ran).toEqual([]);
    expect(server.requests.some((r) => r.url.startsWith("/api/exercises/ex1/run"))).toBe(true);
    expect(solved).toBe(1);
  });

  test("an 'unavailable' answer from the server is shown as a message", async () => {
    server = mockServer((r) => r.url.includes("/runner/status") ? { json: { available: true, reason: "" } } : { json: { unavailable: "runner crashed" } });
    mount({ runtime: "docker" });
    await flush();
    click(run());
    await flush();
    expect(host.querySelector("[role=alert]")!.textContent).toBe("runner crashed");
    expect(run().disabled).toBe(false);
  });
});
