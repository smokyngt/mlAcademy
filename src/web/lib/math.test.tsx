import { expect, test } from "bun:test";
import { normalizeMath } from "./math";
import { renderToString } from "react-dom/server";
import { Markdown } from "../components/Markdown";
import { MemoryRouter } from "react-router-dom";

test("single-line $$...$$ becomes a display block", () => {
  expect(normalizeMath("a\n$$x=1 \\tag{1}$$\nb")).toBe("a\n$$\nx=1 \\tag{1}\n$$\nb");
});
test("opening $$ with content and closing with content are split", () => {
  expect(normalizeMath("$$\\begin{aligned}\nx&=1\\\\\ny&=2\n\\end{aligned}$$\nafter")).toBe("$$\n\\begin{aligned}\nx&=1\\\\\ny&=2\n\\end{aligned}\n$$\nafter");
});
test("code fences are untouched", () => {
  const s = "```python\nprice = '$$5$$'\n```";
  expect(normalizeMath(s)).toBe(s);
});
test("already-correct blocks are unchanged", () => {
  const s = "$$\nx\n$$";
  expect(normalizeMath(s)).toBe(s);
});
test("rendered output has no raw KaTeX source and no KaTeX error nodes", () => {
  const src = "Before\n\n$$\\|a-b\\|^2=\\|a\\|^2-2a\\cdot b \\tag{1}$$\n\n$$\\begin{aligned}\nx&=1\\\\\ny&=2\n\\end{aligned}$$\n\nAfter paragraph with $z$.";
  const html = renderToString(<MemoryRouter><Markdown source={src} /></MemoryRouter>);
  expect(html).toContain("katex-display");
  expect(html).not.toContain("katex-error");
  expect(html).toContain('class="tag"'); // the equation number (1) is really typeset
  expect(html).toContain("After paragraph");
});
