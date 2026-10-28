"""Read-only helpers for the PDFs in ../math.
  python scripts/pdfpages.py toc  "<pdf>" [max_level]        print the table of contents with PDF page numbers
  python scripts/pdfpages.py text "<pdf>" <first> <last> [max_chars]   print the text of PDF pages first..last (1-based)
The pdf argument may be a file name inside ../math (e.g. "math for ML.pdf")."""
import pathlib, re, sys, warnings

warnings.filterwarnings("ignore")
import pymupdf  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
MATH = pathlib.Path(__file__).resolve().parent.parent.parent / "math"
cmd, name = sys.argv[1], sys.argv[2]
doc = pymupdf.open(MATH / name if not pathlib.Path(name).exists() else name)
if cmd == "toc":
    lvl = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for l, t, p in doc.get_toc():
        if l <= lvl:
            print("  " * (l - 1) + f"{t[:80]}  [p{p}]")
elif cmd == "text":
    a, b = int(sys.argv[3]), int(sys.argv[4])
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 12000
    out = [f"[p{i + 1}] " + re.sub(r"\s+", " ", doc[i].get_text()) for i in range(a - 1, min(b, len(doc)))]
    print("\n".join(out)[:cap])
