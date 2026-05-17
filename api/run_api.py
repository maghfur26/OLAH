"""
OLAH - API Launcher
Jalankan: uv run run_api.py
"""
import sys
import os

# Set path SEBELUM import apapun
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set PYTHONPATH environment variable agar proses child uvicorn juga mendapat path
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

# Import app langsung di sini (bukan lewat string "api:app")
# sehingga sys.path sudah benar saat import terjadi
from api import app  # noqa: E402
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,                   # objek langsung, bukan string
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=False karena pakai objek langsung
    )