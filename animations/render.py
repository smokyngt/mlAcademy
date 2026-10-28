"""Render every scene in scenes.py to src/web/public/manim/<Scene>.mp4 (480p15 by default; --hq for 1080p60)."""
import subprocess, sys, shutil
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "src" / "web" / "public" / "manim"
SCENES = ["LinearMap", "Eigenvectors", "GradientDescent", "Derivative"]
q = ("-qh", "1080p60") if "--hq" in sys.argv else ("-ql", "480p15")
OUT.mkdir(parents=True, exist_ok=True)
for s in SCENES:
    subprocess.run([sys.executable, "-m", "manim", q[0], "--media_dir", str(HERE / "media"), str(HERE / "scenes.py"), s], check=True)
    shutil.copy(HERE / "media" / "videos" / "scenes" / q[1] / f"{s}.mp4", OUT / f"{s}.mp4")
    print("ok", s)
