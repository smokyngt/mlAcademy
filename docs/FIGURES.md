# Figures: diagrams and plots inside lessons

Every lesson can have a `figures.json` next to `lesson.md`. Figures are rendered **inside the lesson**, after the n-th second-level section, and are independent of the lesson text (so they never interfere with translations). Two libraries do the drawing:

- **Mermaid** (`kind: "mermaid"`) for architectures, pipelines, data flow, algorithm flow, dependency and timeline diagrams: flowchart, sequence, state, class, ER, mindmap, timeline, `block-beta`, xychart-beta are all available.
- **function-plot** (`kind: "plot"`, D3-based) for mathematics: graphs of functions, derivatives and tangents (interactive), vectors, points, level curves (implicit), shaded regions, parametric curves.

## File format

```json
[
  {
    "id": "f1",
    "after": 1,
    "kind": "mermaid",
    "caption": { "en": "A dense layer: ...", "fr": "Une couche dense : ..." },
    "code": { "en": "flowchart LR\n  X[\"X (B,n)\"] --> L[\"XW + b\"] --> A[\"ReLU\"] --> Y[\"A (B,m)\"]" }
  },
  {
    "id": "f2",
    "after": 2,
    "kind": "plot",
    "caption": { "en": "The sigmoid and its derivative", "fr": "La sigmoïde et sa dérivée" },
    "spec": {
      "xDomain": [-6, 6], "yDomain": [-0.2, 1.2], "height": 320,
      "data": [
        { "fn": "1/(1+exp(-x))", "color": "#2563eb" },
        { "fn": "exp(-x)/(1+exp(-x))^2", "color": "#d97706" }
      ]
    }
  }
]
```

- `after` = number of the `##` section after which the figure appears (0 = after the intro before the first `##`, 1 = after the first `##` section, ...). Use the section where the concept is explained.
- `caption`: required, English; add `fr` for the French reading (otherwise English is shown). Mermaid: `code.en` required, `code.fr` optional (translate node labels; keep ids and structure).
- `plot.spec` options: `xDomain`, `yDomain` (defaults `[-5,5]`), `height` (220-420), `grid` (default true), `title`, and `data`: a list of function-plot items. Colours: use `#2563eb` (blue), `#d97706` (orange), `#16a34a` (green), `#dc2626` (red), `#7c3aed` (purple).

## function-plot item cheat sheet

```json
{ "fn": "x^2" }                                           explicit function of x (mathjs syntax: ^, exp, log, sin, cos, sqrt, abs)
{ "fn": "x^2", "derivative": { "fn": "2*x", "updateOnMouseMove": true } }    draggable tangent line
{ "fn": "x^2", "range": [0, 3], "closed": true }           area under the curve (shaded)
{ "fnType": "implicit", "fn": "x*x + 4*y*y - 4" }          level curves, ellipses, decision boundaries
{ "fnType": "parametric", "x": "cos(t)", "y": "sin(t)", "range": [0, 6.283] }
{ "fnType": "vector", "vector": [2, 1], "offset": [0, 0], "graphType": "polyline", "color": "#2563eb" }    an arrow from offset
{ "fnType": "points", "graphType": "scatter", "color": "#dc2626", "attr": { "r": 4 }, "points": [[1, 1], [2, 3]] }
{ "fnType": "points", "graphType": "polyline", "points": [[0, 0], [1, 2], [3, 1]] }    a path (e.g. gradient-descent iterates)
```
Add text with `"annotations": [{ "x": 1, "text": "minimum" }, { "y": 0.5, "text": "p = 0.5" }]` at the top level of the spec. Keep expressions free of anything but `x` (explicit), `x`/`y` (implicit) or `t` (parametric).

## What to draw (guidance)

- **Math lessons**: a graph for every function family introduced; geometric pictures (vectors, projections, angles, level curves, tangent/secant, the unit circle, eigenvector directions, areas under curves, distributions with shaded tails, convex vs non-convex shapes, loss surfaces as level curves with the descent path).
- **Architecture lessons**: one Mermaid diagram of the whole architecture with tensor shapes on the edges or in the node labels; one per sub-block (attention head, LSTM cell with gates, encoder-decoder, U-Net skips, residual block); training/inference pipelines as flowcharts; algorithm loops (EM, k-means, Lloyd, value iteration) as flow or state diagrams; sequence diagrams for protocols (RAG request, tool-calling loop, speculative decoding handshake).
- 2 to 5 figures per lesson (more for architecture-heavy ones). A figure must teach something the text alone does not; do not decorate. Label axes/edges. Do not duplicate an existing ```mermaid block inside the lesson text.

## Verify (mandatory)

```
bun scripts/verify-figures.ts [course-folder ...]
```
It validates JSON, ids, the `after` range, parses every Mermaid diagram with Mermaid's own parser, and renders every plot with function-plot (bad expressions fail). Fix every error. Do not run `bun run sync`.
