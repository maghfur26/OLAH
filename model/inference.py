"""
OLAH - Recipe Recommender Inference Engine
AI Engineer | Coding Camp 2026 - CC26-PSU127

Load model terlatih → preprocessing input bahan → rekomendasi resep
Siap dikonsumsi oleh FastAPI (api.py).
"""

import os
import re
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict, Any

# Import custom components agar load_model bisa mengenali layer kustom
from recipe_recommender import (
    IngredientAttentionLayer,
    L2NormalizeLayer,
    RecommendationLoss,
    CUSTOM_OBJECTS,
)

INGREDIENT_SYNONYMS = {
    "telor": "telur", "cabe": "cabai", "cabe rawit": "cabai rawit",
    "cabe merah": "cabai merah", "merica": "lada", "merica bubuk": "lada bubuk",
    "bawang bombay": "bawang bombei", "minyak goreng": "minyak",
    "kecap asin": "kecap", "kecap manis": "kecap",
    "ketumbar bubuk": "ketumbar", "santan kara": "santan",
    "santan instan": "santan", "gula pasir": "gula", "gula putih": "gula",
    "garam dapur": "garam", "tepung terigu": "tepung", "tepung maizena": "maizena",
}


class OlahInferenceEngine:
    """
    Engine inference utama.

    Cara pakai:
        engine = OlahInferenceEngine()
        engine.load()
        results = engine.recommend(["ayam", "bawang merah", "kemiri"])
    """

    def __init__(
        self,
        model_path:      str = "./saved_model/olah_recommender.keras",
        embeddings_path: str = "./saved_model/recipe_embeddings.npy",
        metadata_path:   str = "./saved_model/recipe_metadata.json",
        encoder_path:    str = "./saved_model/label_encoder.pkl",
    ):
        self.model_path      = model_path
        self.embeddings_path = embeddings_path
        self.metadata_path   = metadata_path
        self.encoder_path    = encoder_path

        self.model           = None
        self.embedding_model = None
        self.all_embeddings  = None
        self.metadata        = None
        self.vocab           = None
        self.label_encoder   = None
        self.config          = None
        self._is_loaded      = False

    def load(self):
        """Load semua artefak model dari disk."""
        print("[INFERENCE] Loading model components...")

        self.model = keras.models.load_model(
            self.model_path,
            custom_objects=CUSTOM_OBJECTS,
        )

        # Sub-model: hanya embedding output (untuk cosine similarity)
        self.embedding_model = keras.Model(
            inputs  = self.model.input,
            outputs = self.model.get_layer("embedding_output").output,
            name    = "embedding_extractor",
        )

        self.all_embeddings = np.load(self.embeddings_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(self.encoder_path, "rb") as f:
            saved = pickle.load(f)
        self.vocab         = saved["vocab"]
        self.label_encoder = saved["label_encoder"]
        self.config        = saved["config"]

        self._is_loaded = True
        print(f"[INFERENCE] ✓ Ready | {len(self.metadata)} recipes loaded")

    def _normalize(self, text: str) -> str:
        """Normalisasi satu bahan: lowercase, sinonim, hapus satuan."""
        text = text.lower().strip()
        for wrong, correct in INGREDIENT_SYNONYMS.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        text = re.sub(
            r"\b(gr|gram|kg|ml|liter|sdm|sdt|lembar|batang|siung|buah|ruas|iris|genggam|\d+)\b",
            "", text,
        )
        return re.sub(r"\s+", " ", text).strip()

    def preprocess(self, ingredients: List[str]) -> np.ndarray:
        """
        Konversi list bahan → integer sequence (1, max_seq_length).
        Sama persis dengan DataPreprocessor.text_to_sequence().
        """
        max_len  = self.config["max_seq_length"]
        combined = " ".join(self._normalize(ing) for ing in ingredients)
        tokens   = combined.split()
        seq      = [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return np.array([seq], dtype=np.int32)

    def recommend(
        self,
        ingredients:     List[str],
        top_k:           int = 10,
        category_filter: Optional[str] = None,
        min_similarity:  float = 0.1,
    ) -> Dict[str, Any]:
        """
        Rekomendasikan resep berdasarkan bahan yang dimiliki user.

        Returns dict dengan key:
            status, query_ingredients, total_results, recommendations
        """
        if not self._is_loaded:
            raise RuntimeError("Panggil engine.load() dahulu.")
        if not ingredients:
            return {"status": "error", "message": "Ingredients kosong", "recommendations": []}

        seq             = self.preprocess(ingredients)
        query_embedding = self.embedding_model(seq, training=False).numpy()
        similarities    = cosine_similarity(query_embedding, self.all_embeddings)[0]

        # Turunkan score resep di luar filter kategori
        if category_filter:
            for i, meta in enumerate(self.metadata):
                if meta.get("category") != category_filter:
                    similarities[i] *= 0.2

        top_indices     = np.argsort(similarities)[::-1]
        recommendations = []

        for idx in top_indices:
            if len(recommendations) >= top_k:
                break
            score = float(similarities[idx])
            if score < min_similarity:
                break

            rec  = self.metadata[idx].copy()
            rec["similarity_score"] = round(score, 4)

            # Hitung matched ingredients
            recipe_ings = rec.get("ingredients_cleaned", "").lower()
            matched     = [ing for ing in ingredients
                           if self._normalize(ing) in recipe_ings]
            rec["matched_ingredients"]   = matched
            rec["match_count"]           = len(matched)
            rec["total_user_ingredients"]= len(ingredients)
            rec["match_percentage"]      = round(len(matched) / max(len(ingredients), 1) * 100, 1)
            recommendations.append(rec)

        return {
            "status":           "success",
            "query_ingredients": ingredients,
            "category_filter":  category_filter,
            "total_results":    len(recommendations),
            "recommendations":  recommendations,
        }

    def get_similar_recipes(self, recipe_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Cari resep mirip berdasarkan nama. Untuk fitur 'Resep Serupa' di UI."""
        if not self._is_loaded:
            raise RuntimeError("Panggil engine.load() dahulu.")

        target_idx = next(
            (i for i, m in enumerate(self.metadata)
             if m["recipe_name"].lower() == recipe_name.lower()),
            None,
        )
        if target_idx is None:
            return {"status": "error", "message": f"Resep '{recipe_name}' tidak ditemukan", "recommendations": []}

        query_emb   = self.all_embeddings[target_idx: target_idx + 1]
        sims        = cosine_similarity(query_emb, self.all_embeddings)[0]
        sims[target_idx] = -1  # exclude diri sendiri

        top_indices     = np.argsort(sims)[::-1][:top_k]
        recommendations = []
        for idx in top_indices:
            rec = self.metadata[idx].copy()
            rec["similarity_score"] = round(float(sims[idx]), 4)
            recommendations.append(rec)

        return {
            "status":         "success",
            "source_recipe":  recipe_name,
            "total_results":  len(recommendations),
            "recommendations": recommendations,
        }


# ── Quick test ──
if __name__ == "__main__":
    engine = OlahInferenceEngine()
    engine.load()

    print("\n[TEST 1] Rekomendasi bahan: ayam, bawang merah, kemiri, santan")
    r = engine.recommend(["ayam", "bawang merah", "bawang putih", "kemiri", "santan"], top_k=5)
    print(f"Status: {r['status']} | Results: {r['total_results']}")
    for i, rec in enumerate(r["recommendations"], 1):
        print(f"  {i}. {rec['recipe_name']} [{rec['category']}] sim={rec['similarity_score']:.3f} match={rec['match_percentage']}%")

    print("\n[TEST 2] Filter kategori telur")
    r2 = engine.recommend(["telur", "bawang putih", "garam", "minyak"], top_k=5, category_filter="telur")
    for i, rec in enumerate(r2["recommendations"], 1):
        print(f"  {i}. {rec['recipe_name']} [{rec['category']}] sim={rec['similarity_score']:.3f}")

    print("\n[DONE]")