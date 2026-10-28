import { useMemo, useState, type ReactNode } from "react";
import { VIZ_NAMES } from "@shared/viz-names";

/* ---------- shared bits ---------- */

function Frame({ title, children, readout, legend }: { title: string; children: ReactNode; readout?: ReactNode; legend?: [string, string][] }) {
  return (
    <figure className="card fig not-prose my-8">
      <figcaption className="mb-4 flex items-center gap-2"><span className="fig-title">{title}</span><span className="chip">interactive</span></figcaption>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="space-y-3">
          {children}
          {legend && <div className="legend">{legend.map(([c, l]) => <span key={l}><i style={{ background: c }} />{l}</span>)}</div>}
        </div>
        <div className="space-y-4">{readout}</div>
      </div>
    </figure>
  );
}

/** One measured value: small label, large number. `hero` marks the single most important result. */
function Stat({ label, value, hero, hint }: { label: string; value: string; hero?: boolean; hint?: string }) {
  return (
    <div className={`stat ${hero ? "stat-hero" : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value num">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}
const StatGrid = ({ children }: { children: ReactNode }) => <div className="grid grid-cols-2 gap-2.5">{children}</div>;

function Slider({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="slider-row">
      <span className="slider-name">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label={label} />
      <span className="slider-val">{fmt ? fmt(value) : value}</span>
    </label>
  );
}

const f2 = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "∞");

/** Square plane SVG with grid; `map` converts math coordinates to pixels. */
function Plane({ range, children, w = 420 }: { range: number; children: (m: (x: number, y: number) => [number, number]) => ReactNode; w?: number }) {
  const m = (x: number, y: number): [number, number] => [w / 2 + (x / range) * (w / 2), w / 2 - (y / range) * (w / 2)];
  const ticks = Array.from({ length: 2 * range + 1 }, (_, i) => i - range);
  return (
    <svg viewBox={`0 0 ${w} ${w}`} className="w-full max-w-xl rounded-xl border border-line bg-surface2" role="img">
      {ticks.map((t) => (
        <g key={t} stroke="var(--border)" strokeWidth={t === 0 ? 1.5 : 0.6}>
          <line x1={m(t, -range)[0]} y1={m(t, -range)[1]} x2={m(t, range)[0]} y2={m(t, range)[1]} stroke={t === 0 ? "var(--muted)" : undefined} />
          <line x1={m(-range, t)[0]} y1={m(-range, t)[1]} x2={m(range, t)[0]} y2={m(range, t)[1]} stroke={t === 0 ? "var(--muted)" : undefined} />
        </g>
      ))}
      {children(m)}
    </svg>
  );
}

function Arrow({ from, to, color, dashed }: { from: [number, number]; to: [number, number]; color: string; dashed?: boolean }) {
  const [x1, y1] = from, [x2, y2] = to;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const h = 8;
  return (
    <g stroke={color} fill={color} strokeWidth={2.2}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} strokeDasharray={dashed ? "5 4" : undefined} />
      <polygon points={`${x2},${y2} ${x2 - h * Math.cos(ang - 0.4)},${y2 - h * Math.sin(ang - 0.4)} ${x2 - h * Math.cos(ang + 0.4)},${y2 - h * Math.sin(ang + 0.4)}`} stroke="none" />
    </g>
  );
}

/* ---------- 1. vectors ---------- */

function VectorPlayground() {
  const [a, setA] = useState<[number, number]>([3, 2]);
  const [b, setB] = useState<[number, number]>([4, -1]);
  const dot = a[0] * b[0] + a[1] * b[1];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const cos = na && nb ? dot / (na * nb) : NaN;
  const lam = nb ? dot / (nb * nb) : 0;
  const p: [number, number] = [lam * b[0], lam * b[1]];
  return (
    <Frame
      title="Dot product, angle and projection"
      legend={[["var(--accent)", "vector a"], ["var(--warn)", "vector b"], ["var(--good)", "projection of a onto b"]]}
      readout={
        <>
          <Slider label="a₁" value={a[0]} min={-5} max={5} step={0.5} onChange={(v) => setA([v, a[1]])} />
          <Slider label="a₂" value={a[1]} min={-5} max={5} step={0.5} onChange={(v) => setA([a[0], v])} />
          <Slider label="b₁" value={b[0]} min={-5} max={5} step={0.5} onChange={(v) => setB([v, b[1]])} />
          <Slider label="b₂" value={b[1]} min={-5} max={5} step={0.5} onChange={(v) => setB([b[0], v])} />
          <Stat hero label="Dot product  a·b = Σ aᵢbᵢ" value={f2(dot, 2)} hint={Math.abs(dot) < 0.05 ? "≈ 0: the arrows are perpendicular" : dot > 0 ? "positive: they point roughly the same way" : "negative: they point roughly opposite ways"} />
          <StatGrid>
            <Stat label="Angle θ" value={`${f2((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI, 1)}°`} />
            <Stat label="cos θ" value={f2(cos)} />
            <Stat label="Length ‖a‖" value={f2(na, 2)} />
            <Stat label="Length ‖b‖" value={f2(nb, 2)} />
          </StatGrid>
          <p className="text-sm text-muted">Check: ‖a‖‖b‖cos θ = <b className="num text-ink">{f2(na * nb * cos, 2)}</b>, the same number computed the geometric way.</p>
        </>
      }
    >
      <Plane range={5}>
        {(m) => (
          <>
            <line x1={m(...a)[0]} y1={m(...a)[1]} x2={m(...p)[0]} y2={m(...p)[1]} stroke="var(--muted)" strokeDasharray="2 3" />
            <Arrow from={m(0, 0)} to={m(...p)} color="var(--good)" dashed />
            <Arrow from={m(0, 0)} to={m(...a)} color="var(--accent)" />
            <Arrow from={m(0, 0)} to={m(...b)} color="var(--warn)" />
          </>
        )}
      </Plane>
    </Frame>
  );
}

/* ---------- 2. matrix transform ---------- */

const PRESETS: Record<string, [number, number, number, number]> = {
  identity: [1, 0, 0, 1],
  "rotate 30°": [Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)],
  "scale (2, .5)": [2, 0, 0, 0.5],
  shear: [1, 1, 0, 1],
  reflect: [1, 0, 0, -1],
  "singular (rank 1)": [1, 2, 0.5, 1],
};

function MatrixTransform() {
  const [M, setM] = useState<[number, number, number, number]>(PRESETS["rotate 30°"]);
  const [a, b, c, d] = M;
  const det = a * d - b * c;
  const T = (x: number, y: number): [number, number] => [a * x + b * y, c * x + d * y];
  const set = (i: number) => (v: number) => setM(M.map((x, j) => (i === j ? v : x)) as typeof M);
  const lines = [-3, -2, -1, 0, 1, 2, 3];
  return (
    <Frame
      title="A matrix as a transformation of the plane"
      legend={[["var(--bad)", "image of e₁ = first column"], ["var(--good)", "image of e₂ = second column"], ["var(--accent)", "image of the unit square"]]}
      readout={
        <>
          <div className="flex flex-wrap gap-1">{Object.keys(PRESETS).map((k) => <button key={k} className="btn !px-2 !py-0.5 text-xs" onClick={() => setM(PRESETS[k])}>{k}</button>)}</div>
          <Slider label="A₁₁" value={a} min={-2} max={2} step={0.1} onChange={set(0)} fmt={(v) => f2(v, 2)} />
          <Slider label="A₁₂" value={b} min={-2} max={2} step={0.1} onChange={set(1)} fmt={(v) => f2(v, 2)} />
          <Slider label="A₂₁" value={c} min={-2} max={2} step={0.1} onChange={set(2)} fmt={(v) => f2(v, 2)} />
          <Slider label="A₂₂" value={d} min={-2} max={2} step={0.1} onChange={set(3)} fmt={(v) => f2(v, 2)} />
          <Stat hero label="Determinant det A (area scale factor)" value={f2(det, 2)} hint={Math.abs(det) < 0.05 ? "≈ 0: the plane is squashed onto a line" : det < 0 ? "negative: orientation is flipped" : "the unit square keeps its orientation"} />
        </>
      }
    >
      <Plane range={4}>
        {(m) => (
          <>
            {lines.map((t) => (
              <g key={t} stroke="var(--accent)" strokeOpacity={0.35} strokeWidth={1}>
                <line x1={m(...T(t, -3))[0]} y1={m(...T(t, -3))[1]} x2={m(...T(t, 3))[0]} y2={m(...T(t, 3))[1]} />
                <line x1={m(...T(-3, t))[0]} y1={m(...T(-3, t))[1]} x2={m(...T(3, t))[0]} y2={m(...T(3, t))[1]} />
              </g>
            ))}
            <polygon points={[T(0, 0), T(1, 0), T(1, 1), T(0, 1)].map((q) => m(...q).join(",")).join(" ")} fill="var(--accent)" fillOpacity={0.25} stroke="var(--accent)" />
            <Arrow from={m(0, 0)} to={m(...T(1, 0))} color="var(--bad)" />
            <Arrow from={m(0, 0)} to={m(...T(0, 1))} color="var(--good)" />
          </>
        )}
      </Plane>
    </Frame>
  );
}

/* ---------- 3. derivative explorer ---------- */

const FUNS: Record<string, { f: (x: number) => number; df: (x: number) => number }> = {
  "x²": { f: (x) => x * x, df: (x) => 2 * x },
  "x³ − x": { f: (x) => x ** 3 - x, df: (x) => 3 * x * x - 1 },
  "sin x": { f: Math.sin, df: Math.cos },
  eˣ: { f: Math.exp, df: Math.exp },
};

function DerivativeExplorer() {
  const [name, setName] = useState("x²");
  const [x0, setX0] = useState(1);
  const [logh, setLogh] = useState(0);
  const { f, df } = FUNS[name];
  const h = 10 ** logh;
  const secant = (f(x0 + h) - f(x0)) / h;
  const central = (f(x0 + h) - f(x0 - h)) / (2 * h);
  const W = 360, X = 3, Y = 6;
  const sx = (x: number) => W / 2 + (x / X) * (W / 2);
  const sy = (y: number) => W / 2 - (Math.max(-Y, Math.min(Y, y)) / Y) * (W / 2);
  const pts = useMemo(() => Array.from({ length: 121 }, (_, i) => { const x = -X + (i * 2 * X) / 120; return `${sx(x)},${sy(f(x))}`; }).join(" "), [name]);
  const line = (slope: number, color: string) => <line x1={sx(-X)} y1={sy(f(x0) + slope * (-X - x0))} x2={sx(X)} y2={sy(f(x0) + slope * (X - x0))} stroke={color} strokeWidth={1.8} />;
  return (
    <Frame
      title="From secant to tangent"
      legend={[["var(--accent)", "tangent (true slope)"], ["var(--warn)", "secant through x₀ and x₀+h"]]}
      readout={
        <>
          <div className="flex gap-1">{Object.keys(FUNS).map((k) => <button key={k} className={`btn !px-2 !py-0.5 text-xs ${k === name ? "!border-accent text-accent" : ""}`} onClick={() => setName(k)}>{k}</button>)}</div>
          <Slider label="x₀" value={x0} min={-2.5} max={2.5} step={0.1} onChange={setX0} fmt={(v) => f2(v, 1)} />
          <Slider label="h (log scale)" value={logh} min={-4} max={0.3} step={0.05} onChange={setLogh} fmt={() => h.toExponential(1)} />
          <Stat hero label="True derivative f′(x₀)" value={f2(df(x0), 4)} hint="slope of the blue tangent" />
          <StatGrid>
            <Stat label="Secant slope" value={f2(secant, 4)} hint={`error ${Math.abs(secant - df(x0)).toExponential(1)}`} />
            <Stat label="Central diff." value={f2(central, 4)} hint={`error ${Math.abs(central - df(x0)).toExponential(1)}`} />
          </StatGrid>
          <p className="text-sm text-muted">Shrink h: the secant converges to the tangent. At the same h the central difference is far more accurate (error O(h²) versus O(h)).</p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${W}`} className="w-full max-w-xl rounded-xl border border-line bg-surface2" role="img">
        <line x1={0} y1={W / 2} x2={W} y2={W / 2} stroke="var(--muted)" /><line x1={W / 2} y1={0} x2={W / 2} y2={W} stroke="var(--muted)" />
        <polyline points={pts} fill="none" stroke="var(--text)" strokeWidth={2} />
        {line(df(x0), "var(--accent)")}
        {line(secant, "var(--warn)")}
        <circle cx={sx(x0)} cy={sy(f(x0))} r={4} fill="var(--accent)" />
        <circle cx={sx(x0 + h)} cy={sy(f(x0 + h))} r={4} fill="var(--warn)" />
      </svg>
    </Frame>
  );
}

/* ---------- 4. gradient descent ---------- */

function GradientDescentViz() {
  const [lr, setLr] = useState(0.15);
  const [kappa, setKappa] = useState(10);
  const [steps, setSteps] = useState(30);
  const start: [number, number] = [-4, 2];
  const loss = (x: number, y: number) => 0.5 * (x * x + kappa * y * y);
  const path = useMemo(() => {
    const p: [number, number][] = [start];
    let [x, y] = start;
    for (let i = 0; i < steps; i++) { x -= lr * x; y -= lr * kappa * y; p.push([x, y]); if (Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break; }
    return p;
  }, [lr, kappa, steps]);
  const last = path[path.length - 1];
  const limit = 2 / kappa;
  const rho = Math.max(Math.abs(1 - lr), Math.abs(1 - lr * kappa));
  const W = 360, Xr = 5, Yr = 3;
  const sx = (x: number) => W / 2 + (x / Xr) * (W / 2);
  const sy = (y: number) => W / 2 - (y / Yr) * (W / 2);
  const levels = [0.5, 2, 5, 9, 14];
  return (
    <Frame
      title="Gradient descent on ½(x² + κy²)"
      legend={[["var(--warn)", "start"], ["var(--accent)", "path of the iterates"], ["var(--border)", "level curves of the loss"]]}
      readout={
        <>
          <Slider label="learning rate η" value={lr} min={0.01} max={1.2} step={0.01} onChange={setLr} fmt={(v) => f2(v, 2)} />
          <Slider label="condition number κ" value={kappa} min={1} max={50} step={1} onChange={setKappa} />
          <Slider label="steps" value={steps} min={1} max={100} step={1} onChange={setSteps} />
          <Stat hero label={`Loss after ${path.length - 1} steps`} value={loss(...last) > 1e9 ? "diverged" : f2(loss(...last), 4)} hint={`started at ${f2(loss(...start), 2)}`} />
          <StatGrid>
            <Stat label="Stability limit 2/κ" value={f2(limit, 3)} hint={lr < limit ? "your η is below it" : "your η is above it"} />
            <Stat label="Shrink factor / step" value={f2(rho, 3)} hint={rho < 1 ? "below 1 = converging" : "≥ 1 = diverging"} />
          </StatGrid>
          <p className={`text-sm font-semibold ${rho < 1 ? "text-good" : "text-bad"}`}>
            {rho >= 1 ? "Diverges: η exceeds 2/κ, the steep axis blows up." : lr > 1 / kappa ? "Converges, oscillating along the steep axis (η > 1/κ)." : "Converges smoothly."}
          </p>
          <p className="text-sm text-muted">Try κ=50: η=0.03 is stable but slow along x, η=0.05 diverges. That is ill-conditioning.</p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${W}`} className="w-full max-w-xl overflow-hidden rounded-xl border border-line bg-surface2" role="img">
        <line x1={0} y1={W / 2} x2={W} y2={W / 2} stroke="var(--muted)" /><line x1={W / 2} y1={0} x2={W / 2} y2={W} stroke="var(--muted)" />
        {levels.map((c) => <ellipse key={c} cx={W / 2} cy={W / 2} rx={(Math.sqrt(2 * c) / Xr) * (W / 2)} ry={(Math.sqrt((2 * c) / kappa) / Yr) * (W / 2)} fill="none" stroke="var(--border)" strokeWidth={1.2} />)}
        <polyline points={path.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
        {path.slice(0, 60).map(([x, y], i) => <circle key={i} cx={sx(x)} cy={sy(y)} r={i === 0 ? 4.5 : 2.4} fill={i === 0 ? "var(--warn)" : "var(--accent)"} />)}
      </svg>
    </Frame>
  );
}

/* ---------- 5. activations ---------- */

const ACTS: Record<string, { f: (z: number) => number; d: (z: number) => number; color: string }> = {
  ReLU: { f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0), color: "var(--accent)" },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }, color: "var(--good)" },
  tanh: { f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2, color: "var(--warn)" },
};

function ActivationPlot() {
  const [on, setOn] = useState<Record<string, boolean>>({ ReLU: true, sigmoid: true, tanh: true });
  const [deriv, setDeriv] = useState(false);
  const [z0, setZ0] = useState(0);
  const W = 360, X = 5, Y = 2;
  const sx = (x: number) => W / 2 + (x / X) * (W / 2);
  const sy = (y: number) => W / 2 - (y / Y) * (W / 2);
  const poly = (fn: (z: number) => number) => Array.from({ length: 101 }, (_, i) => { const z = -X + (i * 2 * X) / 100; return `${sx(z)},${sy(Math.max(-Y, Math.min(Y, fn(z))))}`; }).join(" ");
  return (
    <Frame
      title="Activation functions and their derivatives"
      legend={Object.entries(ACTS).map(([k, a]) => [a.color, k] as [string, string])}
      readout={
        <>
          <div className="flex flex-wrap gap-3 text-xs">
            {Object.keys(ACTS).map((k) => <label key={k} className="flex items-center gap-1"><input type="checkbox" checked={on[k]} onChange={(e) => setOn({ ...on, [k]: e.target.checked })} />{k}</label>)}
            <label className="flex items-center gap-1"><input type="checkbox" checked={deriv} onChange={(e) => setDeriv(e.target.checked)} />show derivative φ′</label>
          </div>
          <Slider label="evaluate at z" value={z0} min={-5} max={5} step={0.1} onChange={setZ0} fmt={(v) => f2(v, 1)} />
          <div className="space-y-2">
            {Object.entries(ACTS).map(([k, a]) => (
              <div key={k} className="stat flex items-center justify-between gap-3">
                <span className="stat-label" style={{ color: a.color }}>{k}</span>
                <span className="num text-right"><span className="text-xs text-muted">φ </span><b className="text-lg">{f2(a.f(z0), 3)}</b><span className="ml-3 text-xs text-muted">φ′ </span><b className="text-lg">{f2(a.d(z0), 3)}</b></span>
              </div>
            ))}
          </div>
          <p className="text-sm text-muted">The sigmoid derivative never exceeds 0.25; tanh peaks at 1; ReLU is exactly 0 or 1 (0 for every negative z: the dead-unit risk).</p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${W}`} className="w-full max-w-xl rounded-xl border border-line bg-surface2" role="img">
        <line x1={0} y1={sy(0)} x2={W} y2={sy(0)} stroke="var(--muted)" /><line x1={sx(0)} y1={0} x2={sx(0)} y2={W} stroke="var(--muted)" />
        {Object.entries(ACTS).map(([k, a]) => on[k] && <polyline key={k} points={poly(deriv ? a.d : a.f)} fill="none" stroke={a.color} strokeWidth={2.2} strokeDasharray={deriv ? "5 3" : undefined} />)}
        <line x1={sx(z0)} y1={0} x2={sx(z0)} y2={W} stroke="var(--muted)" strokeDasharray="2 3" />
      </svg>
    </Frame>
  );
}

/* ---------- registry ---------- */

const REGISTRY: Record<string, () => ReactNode> = {
  "vector-playground": () => <VectorPlayground />,
  "matrix-transform": () => <MatrixTransform />,
  "derivative-explorer": () => <DerivativeExplorer />,
  "gradient-descent": () => <GradientDescentViz />,
  "activation-plot": () => <ActivationPlot />,
};

export function VizByName({ name }: { name: string }) {
  if (!VIZ_NAMES.includes(name) || !REGISTRY[name]) return <p className="text-bad">Unknown visualisation “{name}”.</p>;
  return <>{REGISTRY[name]()}</>;
}
