# Translating a lesson into French

Run everything from `C:\Users\briki\Desktop\side Project\ML\ML-journey\academy`. The English files are the source of truth and must NOT be modified.

For each lesson folder `content/authored/<course>/<NN-slug>/` create these siblings of the English files:

| English | French file | Content |
|---|---|---|
| `lesson.md` | `lesson.fr.md` | same frontmatter keys: keep `id`, `teaches`, `requires`, `minutes`, `sources` EXACTLY; translate `title` and `summary` (quote them if they contain a colon). Body fully translated. |
| `quiz.json` | `quiz.fr.json` | `[{ "id", "prompt", "options", "explanation" }]` only (no `answer`, `kind`, `concept`): same ids, same order, same number of options in the same order. |
| `cards.json`, `cards-more.json`, ... | `cards.fr.json`, `cards-more.fr.json`, ... | `[{ "id", "front", "back" }]` same ids, same order. |
| `exercises/<name>.exercise.txt` | `exercises/<name>.fr.json` | `{ "id", "title", "prompt", "hints": [...] }`: id must equal the exercise id; same number of hints. Do NOT touch starter/tests/solution (code stays English). |

## Rules (checked automatically)

- Translate prose into clear, natural, pedagogical French (tutoiement is fine, as in "tu"). Keep a teaching tone; do not summarise or shorten.
- **Copy unchanged**: every formula (`$...$`, `$$...$$`), every fenced code block (including ```viz and ```mermaid), every `[[lesson-id]]` link, every URL, file path, identifier, function name, variable name and number. Python comments inside code blocks stay as they are.
- Keep the heading structure (same number of `#`, `##`, `###` headings). Headings are translated.
- French typography is optional; do not translate technical names that are used as-is in French ML writing (e.g. "softmax", "dropout", "batch norm", "embedding", "token", "fine-tuning") unless a standard French term is common (e.g. "descente de gradient", "rétropropagation", "fonction de perte", "valeur propre", "vraisemblance").
- Inside JSON strings, LaTeX backslashes must be doubled (`\\alpha`). Use the Write tool for files with LaTeX: shell heredocs corrupt backslashes.
- Do not add or remove quiz questions, cards, options or hints.

## Verify (mandatory, until clean)

```
bun scripts/verify-translations.ts <course-folder> [<course-folder> ...]
```

It checks ids, option counts, card ids, code blocks, formulas, headings and links against the English originals. Do not run `bun run sync`; do not edit `src/`, `content/*.json` or English lesson files.
