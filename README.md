# ML Academy

A personal mathematics → machine learning → deep learning → LLM academy, written from the books, slides and lesson sets in this repository.
Bun + Hono + SQLite (Drizzle) on the server; React + Vite + Tailwind + KaTeX + Fumadocs on the client.

## Run it

```bash
cd academy
bun install
bun run sync        # build data/academy.db from content/ (idempotent; never touches your progress)
bun run dev         # API :3001 + Vite :5173 -> http://localhost:5173   (Ctrl+C stops both)
# or production:
bun run build && bun run start   # http://localhost:3001
```

Checks: `bun run typecheck`, `bun test`, `bun run verify:content` (runs every reference solution and every starter of every exercise, and checks links).
If `bun run dev` says port 5173 is taken, an old dev server is still running: close it.

## What is in it

**145 lessons** in two parts (Mathematics, Machine Learning) on a single beginner → intermediate → advanced → expert path (`content/path.json`), with 203 exercises that are run against their tests, 1,175 quiz questions and 1,947 flashcards.

| Area | Courses |
|---|---|
| Mathematics | foundations (programming, algebra, functions, logs, trig, series, limits), vectors/matrices/calculus, eigen/SVD, systems, probability, statistics, Monte Carlo, information theory, convexity, numerics, complex/Fourier/graphs/Markov/integrals |
| Machine learning core | gradient descent, losses, neural networks from scratch, autograd; regression, logistic regression, trees, boosting, SVM, kNN/NB, evaluation, features, GMM/EM, time series, anomalies, tuning, lasso, t-SNE/interpretability; deep-learning toolkit |
| Architectures | CNNs, RNN/LSTM/GRU, embeddings, seq2seq, attention, transformers, BERT/T5/GPT, vision, NLP, generative models, RL, speech/audio, modern architectures |
| LLMs and systems | tokenisation, language models, pre-training, LoRA, alignment, KV cache, quantisation, RAG, evaluation, LLM applications, MLOps, reasoning models, safety/fairness/privacy |

App features: dashboard (real activity only), path page with "maths you need" per ML module, Fumadocs lesson reader (sidebar, TOC, search), KaTeX, five interactive figures, quizzes graded on the server, flashcards (SM-2 variant), exercises with hints and gated solutions, notebook lab, concept map with mastery, bookmarks, cumulative assessment, capstone with written evidence, "What is missing" page (coverage computed from the database), "Sources" page (what each lesson actually read). Interface in English and French.

## Sources: how they were used

Your library is source material, **not** imported text. Each lesson was written in original words and lists in its frontmatter exactly which pages or files were opened (`READ`). The Sources page shows this per lesson and marks the first 17 lessons as "pointer only": they were written from general knowledge and their citations say where your material treats the same topic. Library authors' numbers were checked by code where possible; several discrepancies (CNN parameter count, codec bitrate, scaling-law ratio, FID bias) are called out inside the lessons.

## Code execution (no faking)

User code never runs in the server process.
1. **Browser WebAssembly sandbox** (default): CPython + NumPy via Pyodide in a Web Worker, killed on timeout. First use downloads ~10 MB.
2. **Docker sandbox** (PyTorch exercises, notebook PyTorch mode): `--network none`, read-only root, memory/CPU/pids limits, all capabilities dropped. Build once: `docker build -t ml-academy-runner src/server/runner`. **It has never been run end to end** (the Docker daemon was off when it was written); the PyTorch exercises were verified with a local Python.

Browser results are reported by the client; fine for one local user.

## Architecture

```
content/
  courses.json, path.json (the learning path), concepts.json + concepts.d/*.json, capstones.json, roadmap.json, bibliography.json
  authored/<course>/<NN-slug>/  lesson.md, quiz.json, cards*.json, exercises/*.exercise.txt   (+ optional *.fr.* translations)
scripts/  build-catalog.ts (content -> SQLite), verify-content.ts, fix-math-fences.py, pdfpages.py, dev.ts, inventory.py
src/server  Hono API, Drizzle schema + migrations, mastery.ts, srs.ts, runner/, content-i18n.ts
src/web     React app: pages/, components/, viz/, docs/ (Fumadocs), i18n/, lib/math.ts (math normaliser)
docs/       AUTHORING.md, AGENT_BRIEF.md, SOURCE_INVENTORY.md
```

Adding a lesson: follow `docs/AUTHORING.md`, add the id to `content/path.json`, run `bun run sync && bun run verify:content`.

Mastery: score = exponential average of evidence (quiz 0.3, exercise 0.5, card 0.15); mastered at ≥ 0.8 with ≥ 3 pieces of evidence; two recent misses recommend revisiting weak prerequisites. Failed exercise runs do not lower mastery; only the first solve counts.

## Honest limitations

- The UI has not been seen in a real browser by its author; it type-checks, builds, passes server-rendering and API tests.
- Course text is English only. French exists for the interface and the translation plumbing (`lesson.fr.md`, `quiz.fr.json`, ...) works with English fallback, but nothing is translated yet.
- Docker runner untested end to end; some heavy exercises (CTC, Griffin-Lim, Gibbs, DQN) were timed only in local Python, not in WebAssembly.
- The first 17 lessons still need re-checking against their sources.
- No CUDA/GPU work (CPU-only sandboxes), notebook cells have no plotting or syntax highlighting, no long multi-stage project notebooks yet.
- Regulatory and lab-framework content in the safety course is a dated summary of course notes: verify against official sources.
- Fumadocs is used for the lesson layout, but lessons live in SQLite rather than MDX files.
