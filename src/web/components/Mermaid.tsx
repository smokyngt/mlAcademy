import { useEffect, useRef, useState } from "react";
import { useT } from "../i18n";

/** Renders a Mermaid diagram client-side (lazy chunk). Falls back to the source text if the diagram is invalid. */
export default function Mermaid({ code }: { code: string }) {
  const { t } = useT();
  const ref = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    import("mermaid").then(async ({ default: mermaid }) => {
      const c = document.documentElement.classList;
      const dark = c.contains("dark") || (!c.contains("light") && matchMedia("(prefers-color-scheme: dark)").matches);
      mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral", securityLevel: "strict", flowchart: { htmlLabels: true, curve: "basis" }, fontFamily: "Inter Variable, Inter, system-ui, sans-serif" });
      try {
        const { svg } = await mermaid.render("m" + Math.random().toString(36).slice(2), code);
        if (alive && ref.current) { ref.current.innerHTML = svg; setErr(null); }
      } catch (e) { if (alive) setErr(String(e)); }
    });
    return () => { alive = false; };
  }, [code]);
  if (err) return <pre className="text-xs">{code}</pre>;
  return <div ref={ref} className="my-2 overflow-x-auto text-center [&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-w-full" role="img" aria-label={t("mermaid.aria")} />;
}
