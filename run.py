"""Run the presentation version: python run.py app"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "app":
        raise SystemExit("Usage: python run.py app")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
                                     cwd=Path(__file__).resolve().parent))
