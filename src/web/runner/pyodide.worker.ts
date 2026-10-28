/// <reference lib="webworker" />
/**
 * Runs Python (CPython compiled to WebAssembly) inside a Web Worker. This is a real sandbox:
 * no filesystem, no sockets from Python, and a separate thread that the page can terminate on timeout.
 * NumPy is available; PyTorch is not (those exercises use the Docker runner).
 */
import { HARNESS_PY } from "../../shared/harness";

const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

type Req =
  | { id: number; type: "exercise"; code: string; tests: string }
  | { id: number; type: "cell"; code: string }
  | { id: number; type: "checks"; code: string }
  | { id: number; type: "reset" };

let ready: Promise<any> | null = null;

function boot() {
  if (!ready) {
    ready = (async () => {
      const mod = await import(/* @vite-ignore */ PYODIDE_BASE + "pyodide.mjs");
      const py = await mod.loadPyodide({ indexURL: PYODIDE_BASE });
      await py.loadPackage("numpy");
      py.runPython(HARNESS_PY);
      py.runPython("__nb_ns = {'__name__': '__main__'}");
      return py;
    })();
  }
  return ready;
}

self.onmessage = async (ev: MessageEvent<Req>) => {
  const msg = ev.data;
  try {
    const py = await boot();
    let out: string;
    if (msg.type === "exercise") {
      py.globals.set("_user_code", msg.code);
      py.globals.set("_test_code", msg.tests);
      out = py.runPython("run_exercise(_user_code, _test_code)");
    } else if (msg.type === "cell") {
      py.globals.set("_cell_code", msg.code);
      out = py.runPython("run_cell(_cell_code, __nb_ns)");
    } else if (msg.type === "checks") {
      py.globals.set("_chk_code", msg.code);
      out = py.runPython("run_checks(_chk_code, __nb_ns)");
    } else {
      py.runPython("__nb_ns.clear(); __nb_ns['__name__'] = '__main__'");
      out = "{}";
    }
    (self as unknown as Worker).postMessage({ id: msg.id, ok: true, result: JSON.parse(out) });
  } catch (e) {
    (self as unknown as Worker).postMessage({ id: msg.id, ok: false, error: String((e as Error)?.message ?? e) });
  }
};

// Tell the page as soon as the interpreter is loaded so it can show progress.
boot().then(() => (self as unknown as Worker).postMessage({ id: -1, ok: true, result: { ready: true } })).catch((e) => (self as unknown as Worker).postMessage({ id: -1, ok: false, error: "Could not load Pyodide: " + String(e?.message ?? e) }));
