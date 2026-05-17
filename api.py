"""
OLAH - REST API dengan FastAPI
AI Engineer | Coding Camp 2026 - CC26-PSU127

Jalankan:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Swagger UI otomatis: http://localhost:8000/docs
"""

import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.append(os.path.dirname(__file__))
from inference import OlahInferenceEngine

# ── App ──────────────────────────────────────
app = FastAPI(
    title       = "OLAH Recipe Recommender API",
    description = "Masukkan bahan yang kamu punya → OLAH rekomendasikan resepnya!",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # ganti domain spesifik di production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    allow_credentials = True,
)

# Singleton — load sekali saat startup
engine = OlahInferenceEngine(
    model_path      = os.getenv("MODEL_PATH",      "./saved_model/olah_recommender.keras"),
    embeddings_path = os.getenv("EMBEDDINGS_PATH", "./saved_model/recipe_embeddings.npy"),
    metadata_path   = os.getenv("METADATA_PATH",   "./saved_model/recipe_metadata.json"),
    encoder_path    = os.getenv("ENCODER_PATH",    "./saved_model/label_encoder.pkl"),
)

@app.on_event("startup")
async def startup_event():
    print("[API] Loading OLAH model...")
    engine.load()
    print("[API] ✓ Ready!")


# ── Schemas ──────────────────────────────────
VALID_CATEGORIES = ["ayam", "sapi", "ikan", "kambing", "tahu", "tempe", "telur", "udang"]

class RecommendRequest(BaseModel):
    ingredients:     List[str]       = Field(..., min_length=1, example=["ayam", "bawang merah", "kemiri"])
    top_k:           int             = Field(default=10, ge=1, le=50)
    category_filter: Optional[str]  = Field(default=None, example="ayam")
    min_similarity:  float           = Field(default=0.1, ge=0.0, le=1.0)

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


# ── Endpoints ────────────────────────────────
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
async def get_random(n: int = 5, category: Optional[str] = None):
    """Resep random — untuk homepage / discovery."""
    import random
    if not engine._is_loaded:
        raise HTTPException(503, "Model sedang loading")
    data = engine.metadata or []
    if category:
        data = [m for m in data if m.get("category") == category]
    sample = random.sample(data, min(n, len(data)))
    return {"status": "success", "total": len(sample), "recipes": sample}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)