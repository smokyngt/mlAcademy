import { useEffect, useMemo, useState } from "react";
import { parseCells, taskGroups } from "@shared/notebook";
import { api } from "../api";
import { NotebookRunner, type NbCell, type SavedCell } from "./NotebookRunner";

const hash = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36); };

/** A small runnable notebook embedded in a lesson with a ```notebook fenced block (same cell grammar as projects). */
export default function InlineNotebook({ source }: { source: string }) {
  const cells = useMemo<NbCell[]>(() => parseCells(source).map((c, index) => ({ index, kind: c.kind, title: c.title, body: c.body })), [source]);
  const groups = useMemo(() => taskGroups(parseCells(source)), [source]);
  const id = `inline.${hash(source)}`;
  const [saved, setSaved] = useState<SavedCell[] | null>(null);
  useEffect(() => { api<{ cells: SavedCell[] }>(`/api/notebooks/${id}`).then((n) => setSaved(n.cells)).catch(() => setSaved([])); }, [id]);
  if (saved === null) return <p className="text-sm text-muted">…</p>;
  return (
    <div className="not-prose my-6 rounded-xl border border-line p-3">
      <NotebookRunner
        cells={cells}
        groups={groups}
        saved={saved}
        onState={(c) => api(`/api/notebooks/${id}`, { method: "PUT", body: { title: "Lesson mini-lab", cells: c } }).catch(() => {})}
        getSolution={async (i) => cells[i]?.body ?? ""}
      />
    </div>
  );
}
