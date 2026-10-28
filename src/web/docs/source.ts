import { loader } from "fumadocs-core/source";
import type { CourseSummary, Level, PathPart } from "@shared/types";

/**
 * Builds a Fumadocs source from the catalog served by the API (lessons live in SQLite, not in files).
 * Each course becomes a folder (meta file) and each lesson a page whose slug is the stable lesson id,
 * so Fumadocs generates the sidebar tree, breadcrumbs and previous/next links for us.
 */
/**
 * Sidebar for the learning path: one labelled section per part (Mathematics / Machine Learning), then the modules in
 * teaching order, each prefixed with its level, each holding its lessons.
 */
export function buildPathSource(parts: PathPart[], partLabels: Record<"math" | "ml", string>, levelLabels: Record<Level, string>) {
  const rootPages: string[] = [];
  const files: Parameters<typeof loader>[0]["source"]["files"] = [];
  for (const part of parts) {
    rootPages.push(`---${partLabels[part.id]}---`);
    for (const m of part.modules) {
      rootPages.push(m.id);
      files.push({ type: "meta", path: `${m.id}/meta.json`, data: { title: `${levelLabels[m.level]} · ${m.title}`, pages: m.lessons.map((l) => l.id) } });
      for (const l of m.lessons) files.push({ type: "page", path: `${m.id}/${l.id}.md`, slugs: [l.id], data: { title: l.title } });
    }
  }
  files.unshift({ type: "meta", path: "meta.json", data: { pages: rootPages } });
  return loader({ baseUrl: "/lesson", source: { files } });
}

export function buildDocsSource(catalog: CourseSummary[], partLabels: Record<"math" | "ml", string> = { math: "Mathematics", ml: "Machine Learning" }) {
  // Two clearly separated parts in one always-visible sidebar, using Fumadocs separators.
  const pages = (['math', 'ml'] as const).flatMap((part) => [`---${partLabels[part]}---`, ...catalog.filter((c) => c.part === part).map((c) => c.id)]);
  const files = [
    { type: "meta" as const, path: "meta.json", data: { pages } },
    ...catalog.flatMap((c) => [
      { type: "meta" as const, path: `${c.id}/meta.json`, data: { title: c.title, defaultOpen: c.kind === "authored", pages: c.lessons.map((l) => l.id) } },
      ...c.lessons.map((l) => ({ type: "page" as const, path: `${c.id}/${l.id}.md`, slugs: [l.id], data: { title: l.title } })),
    ]),
  ];
  return loader({ baseUrl: "/lesson", source: { files } });
}
