import { useEffect, useRef, useState } from "react";
import * as FP from "function-plot";
import { useT } from "../i18n";

// function-plot is CommonJS (`exports.default = functionPlot`): depending on the bundler the callable sits one or two `default`s deep.
const unwrap = (m: unknown): typeof import("function-plot").default => {
  let v = m as { default?: unknown };
  for (let i = 0; i < 3 && typeof v !== "function" && v?.default !== undefined; i++) v = v.default as { default?: unknown };
  return v as never;
};
const functionPlot = unwrap(FP);

export interface PlotSpec {
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  grid?: boolean;
  title?: string;
  annotations?: unknown[];
  data: unknown[];
}

/** Mathematical plot (function-plot / D3): functions, tangents, vectors, level curves, shaded areas, point paths. */
export default function PlotFigure({ spec }: { spec: PlotSpec }) {
  const { t } = useT();
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || width < 200) return;
    try {
      el.innerHTML = "";
      functionPlot({
        target: el,
        width,
        height: spec.height ?? 320,
        grid: spec.grid ?? true,
        title: spec.title,
        xAxis: { domain: spec.xDomain ?? [-5, 5] },
        yAxis: { domain: spec.yDomain ?? [-5, 5] },
        data: spec.data as never,
        annotations: spec.annotations as never,
      });
      setErr(null);
    } catch (e) { setErr(String(e)); }
  }, [spec, width]);

  return (
    <div className="plot-figure">
      <div ref={ref} className="w-full" role="img" aria-label={spec.title ?? t("plot.aria")} />
      {err && <p className="text-xs text-bad">{t("plot.error", { err })}</p>}
    </div>
  );
}
