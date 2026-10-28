# Project notebooks (complex, multi-stage)

A project is a long guided notebook that a learner works through cell by cell, with milestones that check their work. Projects live in `content/projects/<id>.project.py`, one file per project, using the cell grammar below. Everything runs in the same sandbox as exercises: CPython + NumPy only (Pyodide), seeded, each cell < 5 s, whole project < 60 s.

## Grammar

A file is a sequence of cells. A cell starts with a line beginning `# %%`.

```
# %% meta
{"id": "proj.spirals", "title": "...", "summary": "...", "level": "intermediate", "minutes": 120,
 "requires": ["f2n.10", "dl.01"], "outcomes": ["...", "..."], "stages": ["Data", "Model", "Experiments", "Report"]}
# %% md Stage 1: the data
Markdown text (shown to the learner as prose; may contain $math$ and ```code``` blocks).
# %% code
# ordinary code cell, pre-filled and run as is
import numpy as np
# %% task Implement `forward`
# a code cell the LEARNER must complete; contains TODO stubs; runs in the shared namespace
def forward(params, X):
    raise NotImplementedError   # TODO
# %% check
# runs after the task cell above, using check(name, lambda: ..., hint) as in exercises. Hidden from the learner until they press "Check".
check("forward output shape", lambda: forward(params, X).shape == (len(X), 3), "Expected (N, 3)")
# %% solution
# reference solution for the previous task cell (hidden behind a button)
def forward(params, X): ...
# %% hint
First hint text (plain text). Several `# %% hint` cells in a row are revealed one at a time.
# %% report
Free-text prompt the learner answers in their own words (e.g. "Explain why Adam beat SGD here"); saved with the notebook, never graded.
```

Rules:
- `meta` must be the first cell. `task` cells are followed by their `check`, optional `hint`s and a `solution`.
- Every task needs a `check` with >= 3 checks and a `solution` that passes all of them. A task's starter must FAIL its checks.
- Cells share one namespace in order (like a notebook). Earlier tasks' SOLUTIONS are substituted when verifying later cells, so a project is verifiable end to end.
- Make the project progressive: 5-9 stages, 8-14 tasks, each stage ends with a short `md` summary of what was learned. Include at least one experiment with several seeds and mean ± std, one failure-analysis task, and one `report` cell per stage.
- Only NumPy, the standard library, and functions you define. No files, no network, no plotting (print tables instead; ASCII sparklines are fine).
- Cite the lessons and library sources used in an `md` cell titled "Sources" (only what you actually read).

## Verify

`bun scripts/verify-projects.ts` runs every project with the solutions substituted and asserts that all checks pass and every task starter fails its checks. (Run it before reporting.)
