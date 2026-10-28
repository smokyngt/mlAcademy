import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Link } from "react-router-dom";
import type { ConceptNode } from "@shared/types";
import { useApi } from "../api";
import { useT } from "../i18n";

const STAGES = ["I", "II", "III", "IV", "V", "VI", "VII"];
const color = (m: number | null) => (m === null ? "var(--surface-2)" : m >= 0.8 ? "var(--good)" : m >= 0.5 ? "var(--warn)" : "var(--bad)");

export function ConceptsPage() {
  const { t } = useT();
  const { data } = useApi<ConceptNode[]>("/api/concepts");
  const [sp] = useSearchParams();
  const wanted = sp.get("stage");
  const [sel, setSel] = useState<string | null>("nn.backprop");
  // Pan/zoom: view = translate(x,y) scale(k) applied to the graph group, in SVG user units.
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ px: number; py: number; vx: number; vy: number; moved: boolean } | null>(null);
  const authored = useMemo(() => (data ?? []).filter((c) => !c.id.startsWith("src.")), [data]);
  const layout = useMemo(() => {
    // Layer = longest path from a root, so every prerequisite sits to the left of the concept that needs it.
    const byId = new Map(authored.map((c) => [c.id, c]));
    const depth = new Map<string, number>();
    const d = (id: string): number => {
      const known = depth.get(id);
      if (known !== undefined) return known;
      depth.set(id, 0);
      const v = Math.max(-1, ...(byId.get(id)?.prerequisites ?? []).map(d)) + 1;
      depth.set(id, v);
      return v;
    };
    authored.forEach((c) => d(c.id));
    const cols = new Map<number, ConceptNode[]>();
    authored.forEach((c) => cols.set(depth.get(c.id)!, [...(cols.get(depth.get(c.id)!) ?? []), c]));
    const pos = new Map<string, { x: number; y: number }>();
    for (const [col, list] of cols) list.forEach((c, i) => pos.set(c.id, { x: 20 + col * 190, y: 20 + i * 52 }));
    return { pos, w: 40 + (Math.max(...cols.keys(), 0) + 1) * 190, h: 40 + Math.max(...[...cols.values()].map((l) => l.length), 1) * 52 };
  }, [authored]);
  useEffect(() => {
    if (!data || !wanted) return;
    const first = data.find((c) => c.stage === wanted && !c.id.startsWith("src."));
    if (first) setSel(first.id);
  }, [data, wanted]);
  // Wheel zooms toward the cursor. Needs a non-passive listener to stop the page from scrolling.
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const scale = Math.min(r.width / layout.w, r.height / layout.h); // meet: px per user unit
      const mx = (e.clientX - r.left - (r.width - layout.w * scale) / 2) / scale;
      const my = (e.clientY - r.top) / scale;
      setView((v) => {
        const k = Math.min(6, Math.max(0.3, v.k * Math.exp(-e.deltaY * 0.0015)));
        return { k, x: mx - ((mx - v.x) / v.k) * k, y: my - ((my - v.y) / v.k) * k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [layout, data]);
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  const node = authored.find((c) => c.id === sel);
  const path = new Set<string>();
  const walk = (id: string) => { if (path.has(id)) return; path.add(id); authored.find((c) => c.id === id)?.prerequisites.forEach(walk); };
  if (sel) walk(sel);
  return (
    <div className="space-y-6">
      <div><h1 className="text-3xl font-bold tracking-tight">{t("concepts.title")}</h1><p className="text-muted">{t("concepts.intro")}</p></div>
      <div className="card relative left-1/2 w-screen max-w-none -translate-x-1/2 rounded-none border-x-0 p-2">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted">
          <span>{t("concepts.zoomHint")}</span>
          <button className="btn !px-2 !py-0.5" onClick={() => setView((v) => ({ ...v, k: Math.min(6, v.k * 1.3) }))} aria-label={t("concepts.zoomIn")}>+</button>
          <button className="btn !px-2 !py-0.5" onClick={() => setView((v) => ({ ...v, k: Math.max(0.3, v.k / 1.3) }))} aria-label={t("concepts.zoomOut")}>−</button>
          <button className="btn !px-2 !py-0.5" onClick={() => setView({ x: 0, y: 0, k: 1 })}>{t("concepts.reset")}</button>
        </div>
        <svg ref={svgRef} viewBox={`0 0 ${layout.w} ${layout.h}`} preserveAspectRatio="xMidYMin meet" className="block w-full select-none" style={{ height: "calc(100vh - 9rem)", minHeight: 420, cursor: "grab", touchAction: "none" }}
          onPointerDown={(e) => { drag.current = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y, moved: false }; }}
          onPointerMove={(e) => {
            const d = drag.current, el = svgRef.current;
            if (!d || !el) return;
            const dx = e.clientX - d.px, dy = e.clientY - d.py;
            if (!d.moved && Math.hypot(dx, dy) < 4) return;
            d.moved = true;
            const r = el.getBoundingClientRect();
            const scale = Math.min(r.width / layout.w, r.height / layout.h);
            setView((v) => ({ ...v, x: d.vx + dx / scale, y: d.vy + dy / scale }));
          }}
          onPointerUp={() => { setTimeout(() => { drag.current = null; }, 0); }}
          onPointerLeave={() => { drag.current = null; }}
          onClickCapture={(e) => { if (drag.current?.moved) { e.stopPropagation(); e.preventDefault(); } }}
          role="img" aria-label={t("concepts.graphAria")}>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          <defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 8 4 0 8z" fill="var(--muted)" /></marker></defs>
          {authored.flatMap((c) => c.prerequisites.map((p) => {
            const a = layout.pos.get(p), b = layout.pos.get(c.id);
            if (!a || !b) return null;
            const hot = sel && path.has(c.id) && path.has(p);
            return <path key={p + c.id} d={`M${a.x + 170} ${a.y + 17} C ${a.x + 185} ${a.y + 17}, ${b.x - 15} ${b.y + 17}, ${b.x} ${b.y + 17}`} fill="none" stroke={hot ? "var(--accent)" : "var(--border)"} strokeWidth={hot ? 2 : 1.2} markerEnd="url(#arr)" />;
          }))}
          {authored.map((c) => {
            const p = layout.pos.get(c.id)!;
            return (
              <g key={c.id} transform={`translate(${p.x},${p.y})`} onClick={() => setSel(c.id)} style={{ cursor: "pointer" }} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && setSel(c.id)}>
                <rect width={170} height={34} rx={8} fill="var(--surface)" stroke={c.id === sel ? "var(--accent)" : path.has(c.id) ? "var(--accent)" : "var(--border)"} strokeWidth={c.id === sel ? 2.5 : 1.2} />
                <rect width={6} height={34} rx={3} fill={color(c.mastery)} />
                <text x={14} y={21} fontSize={11} fill="var(--text)">{c.title.length > 25 ? c.title.slice(0, 24) + "…" : c.title}</text>
              </g>
            );
          })}
          </g>
        </svg>
      </div>
      {node && (
        <div className="card p-4">
          <p className="text-xs text-muted">{t("cat.stage", { s: node.stage })} · {node.id}</p>
          <h2 className="text-xl font-semibold">{node.title}</h2>
          <p className="mt-1">{node.description}</p>
          <p className="mt-2 text-sm">{t("concepts.mastery")} <b>{node.mastery === null ? t("concepts.noEvidence") : `${Math.round(node.mastery * 100)}%`}</b> {t("concepts.evidence", { n: node.attempts })}</p>
          {node.prerequisites.length > 0 && <p className="mt-1 text-sm">{t("concepts.requires")} {node.prerequisites.map((p, i) => <span key={p}>{i ? ", " : ""}<button className="text-accent underline" onClick={() => setSel(p)}>{authored.find((c) => c.id === p)?.title}</button></span>)}</p>}
          <p className="mt-1 text-sm">{t("concepts.taught")} {node.lessonIds.map((l) => <Link key={l} className="mr-2 text-accent underline" to={`/lesson/${l}`}>{l}</Link>)}</p>
        </div>
      )}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STAGES.map((s) => <div key={s} className={`card p-3 text-sm ${wanted === s ? "!border-accent ring-2 ring-accent/30" : ""}`}><p className="font-semibold">{t("cat.stage", { s })}</p>{authored.filter((c) => c.stage === s).length === 0 && <p className="mt-1 text-muted">{t("concepts.emptyStage")} <Link className="text-accent underline" to="/roadmap">{t("concepts.seeMissing")}</Link>.</p>}<ul className="mt-1 space-y-1">{authored.filter((c) => c.stage === s).map((c) => <li key={c.id} className="flex justify-between"><button className="text-left hover:text-accent" onClick={() => setSel(c.id)}>{c.title}</button><span className="font-mono text-xs text-muted">{c.mastery === null ? "–" : Math.round(c.mastery * 100) + "%"}</span></li>)}</ul></div>)}
      </div>
    </div>
  );
}
