import { lazy, Suspense } from "react";
import type { FigureData } from "@shared/types";
import { Markdown } from "./Markdown";
import type { PlotSpec } from "./PlotFigure";

const Mermaid = lazy(() => import("./Mermaid"));
const PlotFigure = lazy(() => import("./PlotFigure"));

/** Splits markdown into [intro, section 1, section 2, ...] at top-level `## ` headings (code fences respected). */
export function splitSections(src: string): string[] {
  const parts: string[][] = [[]];
  let code = false;
  for (const line of src.split("\n")) {
    if (line.trim().startsWith("```")) code = !code;
    if (!code && /^## /.test(line)) parts.push([]);
    parts[parts.length - 1].push(line);
  }
  return parts.map((p) => p.join("\n"));
}

function Figure({ f }: { f: FigureData }) {
  return (
    <figure className="not-prose card my-6 overflow-hidden p-4">
      <Suspense fallback={<div className="h-40 animate-pulse rounded bg-surface2" />}>
        {f.kind === "mermaid" ? <Mermaid code={f.code ?? ""} /> : f.kind === "video" ? <video className="mx-auto w-full max-w-3xl rounded" src={f.src} controls loop muted playsInline preload="metadata" aria-label={f.caption} /> : <PlotFigure spec={f.spec as PlotSpec} />}
      </Suspense>
      <figcaption className="mt-3 text-center text-sm text-muted">{f.caption}</figcaption>
    </figure>
  );
}

/** The lesson text with its figures placed after the n-th `##` section (0 = after the intro). */
export function LessonBody({ body, figures, math }: { body: string; figures: FigureData[]; math: boolean }) {
  const sections = splitSections(body);
  return (
    <>
      {sections.map((s, i) => (
        <div key={i}>
          <Markdown source={s} math={math} />
          {figures.filter((f) => Math.min(f.after, sections.length - 1) === i).map((f) => <Figure key={f.id} f={f} />)}
        </div>
      ))}
    </>
  );
}
