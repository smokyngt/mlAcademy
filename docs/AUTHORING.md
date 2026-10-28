# Authoring a lesson (read this before writing)

A lesson is a folder `content/authored/<course>/<NN-slug>/` containing:

```
lesson.md              frontmatter + body (English)
quiz.json              6-9 questions
cards.json             12-15 flashcards
exercises/<name>.exercise.txt   1-2 exercises (sections below)
```

Study `content/authored/classical-ml/05-pca/` and `content/authored/math/07-eigenvalues-and-svd/` first: they are the quality bar.

## lesson.md

```markdown
---
id: <prefix>.<NN>                 # globally unique, never reused
title: "Title"                    # quote it if it contains a colon
summary: One sentence.
minutes: 60
teaches: [concept.id, ...]        # concept ids (must exist: see Concepts below)
requires: [concept.id, ...]
sources:
  - "path/relative/to/repo (Sec. X, PDF p.N-M): what it gave you. READ."
---
```

Body sections, in this order (adapt names, keep the spirit):

1. `## Where this fits` — context: which real problem needs this, what it connects to before/after. Never start with a definition.
2. `## The idea in plain words` — analogies and pictures, **no symbols**.
3. Formal definition(s), then `## Derivation` — no unexplained jumps in any important equation.
4. `## Worked example` with explicit numbers that you have **checked by running code**.
5. `## Implementation` (short python block) and `## Shapes` when tensors are involved.
6. `## Common misconceptions`
7. `## Key takeaways`, `## Explain it yourself` (2 prompts), `## Further reading`, `## Mastery criteria`.

Rules:
- Math: `$...$` inline, display math with `$$` ALONE on its own opening and closing lines. Never write `$$\begin{aligned}` on one line (it breaks the renderer). Run `python scripts/fix-math-fences.py --check`.
- Every fenced code block has a language tag. Diagrams: ```mermaid or an interactive figure is optional.
- Do not paste or paraphrase sources at length. Explain in your own words; cite what you used.
- **The `sources` list must contain only material you actually opened and read for this lesson** (page numbers or file paths). Do not cite from memory. If you also use general knowledge, say nothing about it in `sources`.
- State facts you are not sure of as uncertain, or leave them out. Every number in a worked example must be reproducible by the exercise code.

## quiz.json

`[{ "id": "<lessonid>.q1", "concept": "...", "stage": "pre"|"check", "kind": "mcq"|"numeric", "prompt": "...", "options": [...], "answer": <index>|{"value":x,"tol":t}, "explanation": "..." }]`
Use `$...$` math in strings (JSON-escape backslashes: `\\alpha`). Exactly one correct option; vary its position.

## cards.json

`[{ "id": "<lessonid>.c1", "concept": "...", "front": "...", "back": "..." }]`, 12-15 cards: definitions, formulas, "why" questions, shapes, pitfalls.

## exercises/<name>.exercise.txt

Sections, each introduced by a line `#@@ meta|prompt|starter|tests|solution`:

- `meta`: one-line JSON `{"id":"<lessonid>.ex1","title":"...","difficulty":1-5,"runtime":"pyodide","concepts":["..."],"hints":["...","...","..."]}`
- `prompt`: markdown task statement.
- `starter`: python with `raise NotImplementedError` stubs.
- `tests`: python using `check(name, lambda: <truthy>, "hint")`. The user's code is exec'd first, the tests after, in the same namespace. numpy is imported by the user code; import it again in tests.
- `solution`: complete working code.

Runtime `pyodide` = CPython + NumPy only (no torch, no scipy, no sklearn, no matplotlib). Use `runtime: "docker"` only when PyTorch is really needed. Tests must be deterministic (seeded rng) and fast (< 5 s).

## Concepts

Do NOT edit `content/concepts.json`. Put your new concepts in `content/concepts.d/<your-prefix>.json`:
`[{ "id": "dl.adam", "title": "...", "description": "...", "stage": "I".."VII", "prereqs": ["existing.concept.id", ...] }]`.
Existing ids can be listed with `bun scripts/list-concepts.ts`.

## Verify (mandatory) — do not run `bun run sync`

```
python scripts/fix-math-fences.py --check
ONLY=<course-folder> bun scripts/verify-content.ts      # every solution passes, every starter fails
DRY=1 bun scripts/build-catalog.ts                      # frontmatter/quiz/cards/exercise schema check, no database write
```

## Do not touch

`content/courses.json`, `content/path.json`, `content/concepts.json`, anything under `src/`, other agents' folders.
