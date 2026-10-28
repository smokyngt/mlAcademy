/**
 * Notebook cell grammar shared by project files (`content/projects/*.project.py`) and inline ```notebook blocks.
 * See docs/PROJECTS.md. A cell starts with a line `# %% <kind> [title]`; its body runs to the next such line.
 */
export type CellKind = "meta" | "md" | "code" | "task" | "check" | "solution" | "hint" | "report";

export interface Cell {
  kind: CellKind;
  title: string;
  body: string;
}

export interface ProjectMeta {
  id: string;
  title: string;
  summary: string;
  level: "beginner" | "intermediate" | "advanced" | "expert";
  minutes: number;
  requires?: string[];
  outcomes?: string[];
  stages?: string[];
}

const KINDS = new Set<CellKind>(["meta", "md", "code", "task", "check", "solution", "hint", "report"]);

export function parseCells(src: string): Cell[] {
  const cells: Cell[] = [];
  let cur: { kind: CellKind; title: string; lines: string[] } | null = null;
  for (const line of src.replace(/\r\n/g, "\n").split("\n")) {
    const m = line.match(/^# %%\s*(\w+)?\s*(.*)$/);
    if (m && (!m[1] || KINDS.has(m[1] as CellKind))) {
      if (cur) cells.push({ kind: cur.kind, title: cur.title, body: cur.lines.join("\n").replace(/^\n+|\s+$/g, "") });
      cur = { kind: (m[1] as CellKind) ?? "code", title: m[2].trim(), lines: [] };
    } else if (cur) cur.lines.push(line);
  }
  if (cur) cells.push({ kind: cur.kind, title: cur.title, body: cur.lines.join("\n").replace(/^\n+|\s+$/g, "") });
  return cells;
}

export function parseProject(src: string): { meta: ProjectMeta; cells: Cell[] } {
  const cells = parseCells(src);
  if (!cells.length || cells[0].kind !== "meta") throw new Error("a project must start with a `# %% meta` cell");
  const meta = JSON.parse(cells[0].body) as ProjectMeta;
  return { meta, cells: cells.slice(1) };
}

/** For each task cell: the indices of its check, hints and solution (the cells that follow it up to the next task/md/code). */
export function taskGroups(cells: Cell[]) {
  const groups: { task: number; check: number | null; solution: number | null; hints: number[] }[] = [];
  for (let i = 0; i < cells.length; i++) {
    if (cells[i].kind !== "task") continue;
    const g = { task: i, check: null as number | null, solution: null as number | null, hints: [] as number[] };
    for (let j = i + 1; j < cells.length && ["check", "hint", "solution"].includes(cells[j].kind); j++) {
      if (cells[j].kind === "check") g.check = j;
      else if (cells[j].kind === "solution") g.solution = j;
      else g.hints.push(j);
    }
    groups.push(g);
  }
  return groups;
}
