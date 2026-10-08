"""Single-command launcher for the PTP Intelligence web app.

    python webapp/start.py                 # build the frontend if needed, then serve app + API on :8000
    python webapp/start.py --dev           # backend only (run `npm run dev` in webapp/frontend for hot reload on :5173)
    python webapp/start.py --torch-site <path-to-site-packages-with-CUDA-torch>

Open http://127.0.0.1:8000 afterwards.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "webapp" / "frontend"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--dev", action="store_true")
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--torch-site", default=os.environ.get("PTP_TORCH_SITE"))
ap.add_argument("--precision", default=os.environ.get("PTP_SLM_PRECISION", "nf4"), choices=["nf4", "fp16", "int8"])
a = ap.parse_args()

env = dict(os.environ, PYTHONIOENCODING="utf-8", PTP_SLM_PRECISION=a.precision)
if a.torch_site:
    env["PTP_TORCH_SITE"] = a.torch_site

if not a.dev and (a.rebuild or not (FRONT / "dist" / "index.html").exists()):
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        sys.exit("npm not found: install Node.js 18+ or run with --dev and an existing build")
    if not (FRONT / "node_modules").exists():
        subprocess.check_call([npm, "install"], cwd=FRONT)
    subprocess.check_call([npm, "run", "build"], cwd=FRONT)

print(f"Serving on http://127.0.0.1:{a.port}  (Qwen precision {a.precision})")
subprocess.call([sys.executable, "-m", "uvicorn", "webapp.backend.app:app", "--host", "127.0.0.1",
                 "--port", str(a.port)], cwd=ROOT, env=env)
