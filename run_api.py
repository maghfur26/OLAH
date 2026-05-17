"""
OLAH - Entry Point Utama
Jalankan dari ROOT project:
    python run_api.py
"""
import sys
import os

# Tambahkan root project ke sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

# Import app dari api package (api/__init__.py → api/api.py)
from api import app  # noqa: E402
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
