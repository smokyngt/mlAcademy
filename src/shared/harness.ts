/**
 * Python harness shared by every execution backend (Pyodide worker, Docker runner, local content verification).
 * It is plain Python with no dependencies beyond the standard library, so the same text runs everywhere.
 *
 * run_exercise(user_code, test_code) -> JSON string
 *   { phase: "ok" | "user_error" | "test_error", stdout, error, results: [{name, passed, message}] }
 * run_cell(code, ns) -> JSON string  { ok, stdout, value, error }
 * run_checks(code, ns) -> JSON string (same shape as run_exercise), executes a project check cell in a notebook namespace
 */
export const HARNESS_PY = String.raw`
import ast, contextlib, io, json, traceback

__results__ = []

def check(name, fn, hint=""):
    """Register one test. fn is a zero-argument callable returning truthy/falsy (numpy arrays are reduced with .all())."""
    try:
        out = fn()
        ok = bool(out.all()) if hasattr(out, "all") else bool(out)
        msg = "" if ok else (hint or "condition was false")
    except Exception as e:
        ok, msg = False, type(e).__name__ + ": " + str(e)
    __results__.append({"name": name, "passed": ok, "message": msg})

def _tb(e):
    return "".join(traceback.format_exception_only(type(e), e)).strip()

def run_exercise(user_code, test_code):
    del __results__[:]
    ns = {"__name__": "__main__", "check": check}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(user_code, "<your code>", "exec"), ns)
    except BaseException as e:
        return json.dumps({"phase": "user_error", "stdout": buf.getvalue(), "error": _tb(e), "results": []})
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(test_code, "<tests>", "exec"), ns)
    except BaseException as e:
        return json.dumps({"phase": "test_error", "stdout": buf.getvalue(), "error": _tb(e), "results": list(__results__)})
    return json.dumps({"phase": "ok", "stdout": buf.getvalue(), "error": "", "results": list(__results__)})

def run_checks(code, ns):
    """Run a project 'check' cell in the notebook namespace; same result shape as run_exercise."""
    del __results__[:]
    ns["check"] = check
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<check>", "exec"), ns)
    except BaseException as e:
        return json.dumps({"phase": "test_error", "stdout": buf.getvalue(), "error": _tb(e), "results": list(__results__)})
    return json.dumps({"phase": "ok", "stdout": buf.getvalue(), "error": "", "results": list(__results__)})

def run_cell(code, ns):
    buf = io.StringIO()
    value = None
    try:
        tree = ast.parse(code, "<cell>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), ns)
                if v is not None:
                    value = repr(v)
        return json.dumps({"ok": True, "stdout": buf.getvalue(), "value": value, "error": ""})
    except BaseException as e:
        return json.dumps({"ok": False, "stdout": buf.getvalue(), "value": None, "error": _tb(e)})
`;
