import { useEffect, useMemo, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { githubDark, githubLight } from "@uiw/codemirror-theme-github";
import { EditorView, keymap } from "@codemirror/view";
import { Prec } from "@codemirror/state";
import { useT } from "../i18n";

/** Follows the page theme: Fumadocs toggles a `dark`/`light` class on <html>, otherwise the OS preference applies. */
function useIsDark() {
  const read = () => {
    const c = document.documentElement.classList;
    return c.contains("dark") || (!c.contains("light") && matchMedia("(prefers-color-scheme: dark)").matches);
  };
  const [dark, setDark] = useState(read);
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const on = () => setDark(read());
    mq.addEventListener("change", on);
    return () => { obs.disconnect(); mq.removeEventListener("change", on); };
  }, []);
  return dark;
}

/** Python editor: syntax highlighting, bracket matching, indentation, Ctrl/Cmd+Enter to run. */
export function CodeEditor({ value, onChange, onRun, rows = 14, readOnly = false }: { value: string; onChange?: (v: string) => void; onRun?: () => void; rows?: number; readOnly?: boolean }) {
  const dark = useIsDark();
  const { t } = useT();
  const extensions = useMemo(
    () => [
      python(),
      EditorView.lineWrapping,
      Prec.highest(keymap.of([{ key: "Mod-Enter", run: () => { onRun?.(); return true; } }])),
      EditorView.theme({ "&": { fontSize: "0.86rem", borderRadius: "0.6rem", border: "1px solid var(--border)" }, ".cm-scroller": { fontFamily: 'ui-monospace, "Cascadia Code", Consolas, monospace' } }),
    ],
    [onRun],
  );
  const lines = Math.max(rows, 3);
  return (
    <div aria-label={t("editor.aria")}>
      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={extensions}
        theme={dark ? githubDark : githubLight}
        readOnly={readOnly}
        minHeight={`${Math.min(lines, 6) * 1.5 + 1}rem`}
        maxHeight="34rem"
        basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: !readOnly, autocompletion: false }}
      />
    </div>
  );
}
