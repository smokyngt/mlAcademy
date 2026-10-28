import { expect, test } from "bun:test";
import { buildDocsSource } from "./source";
import type { CourseSummary } from "@shared/types";

const catalog: CourseSummary[] = [
  { id: "foundations-to-nn", title: "Path 1", summary: "", stage: "I-IV", kind: "authored", part: "math", lessons: [
    { id: "f2n.01", title: "Functions", minutes: 40, position: 0, status: "new" },
    { id: "f2n.02", title: "Vectors", minutes: 45, position: 1, status: "new" },
  ] },
  { id: "src-x", title: "Library", summary: "", stage: "V", kind: "source", part: "ml", lessons: [{ id: "src.x.01", title: "Intro", minutes: 20, position: 0, status: "new" }] },
];

test("lessons become a Fumadocs page tree with stable lesson-id URLs, in catalog order", () => {
  const s = buildDocsSource(catalog);
  const folders = s.pageTree.children.filter((c: any) => c.type === "folder") as any[];
  expect(folders.map((f) => f.name)).toEqual(["Path 1", "Library"]);
  // The two parts appear as separate labelled sections, in order.
  const seps = s.pageTree.children.filter((c: any) => c.type === "separator").map((c: any) => c.name);
  expect(seps).toEqual(["Mathematics", "Machine Learning"]);
  expect(s.pageTree.children.findIndex((c: any) => c.name === "Path 1")).toBeLessThan(s.pageTree.children.findIndex((c: any) => c.name === "Machine Learning"));
  expect(folders[0].children.map((c: any) => c.url)).toEqual(["/lesson/f2n.01", "/lesson/f2n.02"]);
  expect(s.getPages().length).toBe(3);
});
