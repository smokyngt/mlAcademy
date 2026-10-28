# Brief shared by every lesson-writing agent

You write lessons for a personal ML academy. Repo root: `C:\Users\briki\Desktop\side Project\ML\ML-journey`. The app is in `academy/`; run every command from `C:\Users\briki\Desktop\side Project\ML\ML-journey\academy`.

The user's library is the SOURCE MATERIAL. It is read and used to write ORIGINAL lessons; it is never imported verbatim. The goal is a progressive path from beginner to expert, so write for a learner who has only done the lessons before yours: define every term, never assume an unlisted prerequisite, and say which earlier lessons you build on (use `requires` and `[[lesson-id]]` links).

## Read first
1. `docs/AUTHORING.md` (format, rules, mandatory verification).
2. Two finished lessons as the quality bar: `content/authored/math/07-eigenvalues-and-svd/` and `content/authored/classical-ml/05-pca/`.

## Pedagogy (non-negotiable)
- Open with `## Where this fits` (real problem + connections) and `## The idea in plain words` (analogies, no symbols) BEFORE any formula.
- Derive important equations step by step; no unexplained jumps. Explicit shapes for every tensor.
- Worked numeric examples with numbers you verified by running code. Implement from scratch (NumPy) with tests where possible; mention what the production library does.
- Common misconceptions, limits and trade-offs. Key takeaways, "Explain it yourself", mastery criteria.
- Quiz >= 7 questions, flashcards >= 13, 1-2 exercises per lesson (`runtime: "pyodide"` = NumPy only; use `"docker"` only if PyTorch is essential, and then verify the solution with the local `python`, which has torch). Exercises for topics that are not numeric (e.g. ethics, prompt engineering) should still be small runnable tasks: metrics, harnesses, simulators, checkers.

## Reading tools
- PDFs in `..\math\`: `python scripts/pdfpages.py toc "<pdf name>" 3` and `python scripts/pdfpages.py text "<pdf name>" <first> <last> <max_chars>`. Available: `math for ML.pdf` (Deisenroth), `Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf`, `Designing Machine Learning Systems.pdf` (Huyen), French slides `cours-tal-01.pdf`..`cours-tal-06.pdf`, `coursIntroApprentissageAudio.pdf`, `coursAvanceApprentissageAudio.pdf`.
- Repo lessons: `..\ai-engineering-from-scratch\phases\<NN-phase>\<NN-lesson>\docs\en.md` (+ `code\`). List directories to find them.
- Book notebooks (JSON; read markdown and code cells): `..\LLMs-from-scratch\ch0X\01_main-chapter-code\*.ipynb`, `..\reasoning-from-scratch\chXX\...`.
- Your own personal notebooks/projects may exist under `..\ML_CODE` (use only if directly relevant).

## Honesty rules
- In `sources:` list ONLY what you opened and read for the lesson, each ending in `READ`, with PDF page numbers or file paths. Never cite from memory; never claim a source says something you did not read.
- Original words only; no long quotes. If a source disagrees with common knowledge, say so.
- Every fact in worked examples must be reproducible by code in the exercise.

## Concepts
Do NOT edit `content/concepts.json`. Put your concepts in `content/concepts.d/<your-file>.json` (array of `{id,title,description,stage,prereqs}`); `bun scripts/list-concepts.ts` lists existing ids (other agents add theirs concurrently; depend only on ids that already exist when you start, plus your own).

## Verification (mandatory, until clean)
```
python scripts/fix-math-fences.py          # fixes $$ fences in place
ONLY=<course-folder> bun scripts/verify-content.ts
DRY=1 bun scripts/build-catalog.ts
```
NEVER run `bun run sync`, never edit `content/courses.json`, `content/path.json`, `content/concepts.json`, anything in `src/`, or any other agent's folder. Many agents work concurrently.

## Final answer
A SHORT report only: lessons written (id, title), sources actually read, verification results, anything you could not do.
