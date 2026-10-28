"""Normalise display-math fences: a multi-line $$ block must have `$$` alone on its opening and closing lines.
(remark-math treats text after an opening `$$` as fence metadata, so `$$\\begin{aligned}` never closes and swallows the page.)
Single-line `$$...$$` is left alone. Usage: python scripts/fix-math-fences.py [--check]"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "content" / "authored"
check_only = "--check" in sys.argv


def fix(text: str):
    out, in_block, in_code, changed = [], False, False, False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out.append(line); continue
        if in_code:
            out.append(line); continue
        if in_block:
            if s == "$$":
                in_block = False; out.append(line)
            elif s.endswith("$$"):
                out += [line[: line.rindex("$$")].rstrip(), "$$"]; in_block = False; changed = True
            else:
                out.append(line)
            continue
        if s == "$$":
            in_block = True; out.append(line); continue
        if s.startswith("$$"):
            rest = s[2:]
            changed = True
            if rest.endswith("$$") and len(rest) > 2:
                out += ["$$", rest[:-2].strip(), "$$"]
            else:
                out += ["$$", rest]; in_block = True
            continue
        out.append(line)
    return "\n".join(out), changed, in_block


bad = 0
for p in sorted(ROOT.rglob("*.md")):
    new, changed, unclosed = fix(p.read_text(encoding="utf8"))
    if unclosed:
        print("UNCLOSED $$ block in", p); bad += 1
    if changed:
        print(("would fix " if check_only else "fixed ") + str(p.relative_to(ROOT)))
        bad += check_only
        if not check_only:
            p.write_text(new, encoding="utf8")
sys.exit(1 if bad else 0)
