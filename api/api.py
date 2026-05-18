"""
OLAH - REST API dengan FastAPI
AI Engineer | Coding Camp 2026 - CC26-PSU127

Jalankan dari ROOT project:
    python run_api.py
    ATAU
    uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

Swagger UI otomatis: http://localhost:8000/docs
"""

import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Path Setup ─────────────────────────────────────────────────────────────
# Tambahkan folder model/ ke sys.path agar bisa import inference.py
_API_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../api/
_ROOT_DIR  = os.path.dirname(_API_DIR)                     # .../project root/
_MODEL_DIR = os.path.join(_ROOT_DIR, "model")              # .../model/

if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from inference import OlahInferenceEngine  # noqa: E402

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "OLAH Recipe Recommender API",
    description = "Masukkan bahan yang kamu punya → OLAH rekomendasikan resepnya!",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # ganti domain spesifik di production
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

# ── Paths absolut untuk saved_model (tidak bergantung pada cwd) ───────────
_SAVED_MODEL_DIR = os.path.join(_ROOT_DIR, "saved_model")

# Singleton — load sekali saat startup
engine = OlahInferenceEngine(
    model_path      = os.getenv("MODEL_PATH",
                        os.path.join(_SAVED_MODEL_DIR, "olah_recommender.keras")),
    embeddings_path = os.getenv("EMBEDDINGS_PATH",
                        os.path.join(_SAVED_MODEL_DIR, "recipe_embeddings.npy")),
    metadata_path   = os.getenv("METADATA_PATH",
                        os.path.join(_SAVED_MODEL_DIR, "recipe_metadata.json")),
    encoder_path    = os.getenv("ENCODER_PATH",
                        os.path.join(_SAVED_MODEL_DIR, "label_encoder.pkl")),
)


@app.on_event("startup")
async def startup_event():
    print("[API] Loading OLAH model...")
    engine.load()
    print("[API] ✓ Ready!")


# ── Schemas ──────────────────────────────────────────────────────────────────
VALID_CATEGORIES = ["ayam", "sapi", "ikan", "kambing", "tahu", "tempe", "telur", "udang"]


class RecommendRequest(BaseModel):
    ingredients:     List[str]      = Field(..., min_length=1, example=["ayam", "bawang merah", "kemiri"])
    top_k:           int            = Field(default=10, ge=1, le=50)
    category_filter: Optional[str] = Field(default=None, example="ayam")
    min_similarity:  float          = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, v):
        cleaned = [i.strip() for i in v if i.strip()]
        if not cleaned:
            raise ValueError("Minimal satu bahan harus diisi")
        return cleaned

    @field_validator("category_filter")
    @classmethod
    def validate_category(cls, v):
        if v and v not in VALID_CATEGORIES:
            raise ValueError(f"Category harus salah satu dari: {VALID_CATEGORIES}")
        return v


class SimilarRequest(BaseModel):
    recipe_name: str = Field(..., min_length=2, example="opor ayam")
    top_k:       int = Field(default=5, ge=1, le=20)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status":        "ok",
        "model_loaded":  engine._is_loaded,
        "total_recipes": len(engine.metadata) if engine.metadata else 0,
        "version":       "1.0.0",
    }


@app.get("/categories", tags=["Recipes"])
async def get_categories():
    return {"status": "success", "categories": VALID_CATEGORIES}


@app.post("/recommend", tags=["Recipes"])
async def recommend_recipes(req: RecommendRequest):
    """
    **Endpoint utama** — Rekomendasi resep berdasarkan bahan yang dimiliki.
    """
    if not engine._is_loaded:
        raise HTTPException(503, "Model sedang loading, coba lagi sebentar")

    t0     = time.time()
    result = engine.recommend(
        ingredients     = req.ingredients,
        top_k           = req.top_k,
        category_filter = req.category_filter,
        min_similarity  = req.min_similarity,
    )
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 2)

    if result["status"] == "error":
        raise HTTPException(400, result.get("message", "Error"))
    return result


@app.post("/similar", tags=["Recipes"])
async def get_similar(req: SimilarRequest):
    """Cari resep mirip — untuk fitur 'Resep Serupa' di halaman detail."""
    if not engine._is_loaded:
        raise HTTPException(503, "Model sedang loading")
    result = engine.get_similar_recipes(req.recipe_name, req.top_k)
    if result["status"] == "error":
        raise HTTPException(404, result.get("message", "Tidak ditemukan"))
    return result


@app.get("/recipe/popular", tags=["Recipes"])
async def get_popular(n: int = 10, category: Optional[str] = None):
    """Resep populer berdasarkan love_count."""
    if not engine._is_loaded:
        raise HTTPException(503, "Model sedang loading")
    data = engine.metadata or []
    if category:
        data = [m for m in data if m.get("category") == category]
    data = sorted(data, key=lambda x: x.get("love_count", 0), reverse=True)[:n]
    return {"status": "success", "total": len(data), "recipes": data}


@app.get("/recipe/random", tags=["Recipes"])
async def get_random(n: int = 10, category: Optional[str] = None):
    """Resep random — untuk homepage / discovery."""
    import random
    if not engine._is_loaded:
        raise HTTPException(503, "Model sedang loading")
    data = engine.metadata or []
    if category:
        data = [m for m in data if m.get("category") == category]
    sample = random.sample(data, min(n, len(data)))
    return {"status": "success", "total": len(sample), "recipes": sample}
