"""
OLAH - Generative AI Feature (Side Quest)
AI Engineer | Coding Camp 2026 - CC26-PSU127

Menggunakan Google Gemini API untuk fitur tambahan:
1. Tips memasak otomatis berdasarkan resep
2. Estimasi masa simpan bahan
3. Substitusi bahan jika bahan tertentu tidak ada
4. Deskripsi resep yang menarik untuk UI

Cara pakai:
    pip install google-generativeai
    export GEMINI_API_KEY="your-key-here"
"""

import os
import json
from typing import List, Optional
import google.generativeai as genai


# ─────────────────────────────────────────────
# SETUP GEMINI
# ─────────────────────────────────────────────
def setup_gemini(api_key: Optional[str] = None):
    """Inisialisasi Gemini API."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY tidak ditemukan. "
            "Set environment variable: export GEMINI_API_KEY='your-key'"
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ─────────────────────────────────────────────
# FITUR GENERATIF
# ─────────────────────────────────────────────


class OlahGenerativeFeatures:
    """
    Kumpulan fitur berbasis Generative AI untuk aplikasi OLAH.
    Semua metode menggunakan Gemini untuk menghasilkan konten
    yang membantu pengguna memasak lebih baik.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.model = setup_gemini(api_key)
        self._system_context = (
            "Kamu adalah asisten memasak untuk aplikasi OLAH yang membantu "
            "mengurangi food waste. Berikan jawaban dalam Bahasa Indonesia yang "
            "ramah, singkat, dan praktis. Format respons selalu dalam JSON valid."
        )

    def _call_gemini(self, prompt: str) -> str:
        """Wrapper call ke Gemini API dengan error handling."""
        try:
            full_prompt = f"{self._system_context}\n\n{prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return json.dumps({"error": str(e), "status": "failed"})

    def get_cooking_tips(self, recipe_name: str, ingredients: List[str]) -> dict:
        """
        Generate tips memasak otomatis untuk resep tertentu.
        
        Digunakan di: halaman detail resep → section "Tips dari OLAH"
        """
        prompt = f"""
Berikan 3 tips memasak praktis untuk resep "{recipe_name}" 
dengan bahan utama: {", ".join(ingredients[:8])}.

Respons dalam JSON dengan format:
{{
  "recipe_name": "{recipe_name}",
  "tips": [
    {{"tip": "...", "why": "..."}},
    {{"tip": "...", "why": "..."}},
    {{"tip": "...", "why": "..."}}
  ],
  "difficulty": "mudah|sedang|sulit",
  "estimated_time_minutes": 30
}}
"""
        raw = self._call_gemini(prompt)
        # Parse JSON dari response
        try:
            # Ekstrak JSON dari markdown code block jika ada
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"recipe_name": recipe_name, "tips": [], "raw_response": raw}

    def get_ingredient_substitutes(
        self, missing_ingredient: str, recipe_name: str
    ) -> dict:
        """
        Saran substitusi bahan jika pengguna tidak punya bahan tertentu.
        Fitur kunci untuk mengurangi food waste dan tetap bisa masak.
        
        Digunakan di: halaman rekomendasi → "Tidak punya X? Ganti dengan..."
        """
        prompt = f"""
Pengguna tidak memiliki "{missing_ingredient}" untuk resep "{recipe_name}".
Berikan 2-3 alternatif substitusi yang mudah ditemukan.

Respons dalam JSON:
{{
  "missing_ingredient": "{missing_ingredient}",
  "recipe_name": "{recipe_name}",
  "substitutes": [
    {{
      "ingredient": "nama bahan pengganti",
      "ratio": "perbandingan penggunaan, misal: 1:1 atau 1 sdm = 2 sdm",
      "note": "catatan penggunaan"
    }}
  ],
  "can_skip": true/false,
  "skip_note": "catatan jika bahan bisa dilewati"
}}
"""
        raw = self._call_gemini(prompt)
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"missing_ingredient": missing_ingredient, "substitutes": [], "raw": raw}

    def estimate_shelf_life(self, ingredients: List[str]) -> dict:
        """
        Estimasi masa simpan bahan makanan yang dimiliki user.
        Mendukung fitur pengingat masa simpan (expire reminder) di aplikasi.
        
        Digunakan di: halaman stok bahan → "Segera gunakan sebelum..."
        """
        if not ingredients:
            return {"ingredients": [], "status": "no_ingredients"}

        ingredients_str = ", ".join(ingredients[:10])
        prompt = f"""
Estimasikan masa simpan bahan-bahan berikut dalam kondisi penyimpanan normal:
{ingredients_str}

Respons dalam JSON:
{{
  "estimations": [
    {{
      "ingredient": "nama bahan",
      "shelf_life_days": 7,
      "storage_method": "kulkas|freezer|suhu ruang",
      "storage_tips": "cara menyimpan agar tahan lama",
      "urgency": "segera|normal|tahan lama"
    }}
  ],
  "priority_use": ["bahan yang paling cepat habis masa simpannya"]
}}

Urutkan dari yang paling cepat habis ke yang paling lama.
"""
        raw = self._call_gemini(prompt)
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"estimations": [], "raw": raw}

    def generate_recipe_description(self, recipe_name: str, ingredients: List[str], category: str) -> dict:
        """
        Generate deskripsi resep yang menarik untuk tampilan kartu resep di UI.
        
        Digunakan di: recipe card → deskripsi singkat yang menggugah selera
        """
        prompt = f"""
Buat deskripsi singkat (2 kalimat) yang menarik untuk resep "{recipe_name}" 
kategori {category} dengan bahan: {", ".join(ingredients[:6])}.

Respons dalam JSON:
{{
  "recipe_name": "{recipe_name}",
  "description": "deskripsi singkat menggugah selera...",
  "tagline": "tagline pendek 5-7 kata",
  "emoji": "emoji yang relevan"
}}
"""
        raw = self._call_gemini(prompt)
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"recipe_name": recipe_name, "description": "", "raw": raw}

    def generate_shopping_list(
        self, recipe_name: str, all_ingredients: List[str], user_has: List[str]
    ) -> dict:
        """
        Generate daftar belanja otomatis — bahan yang perlu dibeli.
        Fitur tambahan: daftar belanja terintegrasi sesuai project plan.
        
        Digunakan di: fitur "Tambah ke Daftar Belanja"
        """
        missing = [ing for ing in all_ingredients if ing.lower() not in [u.lower() for u in user_has]]

        if not missing:
            return {
                "recipe_name": recipe_name,
                "shopping_list": [],
                "message": "Kamu sudah punya semua bahan! 🎉"
            }

        prompt = f"""
Untuk memasak "{recipe_name}", pengguna perlu membeli bahan berikut:
{", ".join(missing)}

Buat daftar belanja yang terorganisir dengan estimasi harga dan jumlah.

Respons dalam JSON:
{{
  "recipe_name": "{recipe_name}",
  "shopping_list": [
    {{
      "ingredient": "nama bahan",
      "quantity": "jumlah yang perlu dibeli",
      "estimated_price_idr": 5000,
      "notes": "tips beli, misal: pilih yang segar"
    }}
  ],
  "total_estimated_price_idr": 25000,
  "store_section": {{
    "sayuran": ["..."],
    "bumbu": ["..."],
    "protein": ["..."]
  }}
}}
"""
        raw = self._call_gemini(prompt)
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"recipe_name": recipe_name, "shopping_list": [], "raw": raw}


# ─────────────────────────────────────────────
# FASTAPI ROUTES — tambahkan ke api.py
# ─────────────────────────────────────────────
"""
Tambahkan import dan routes berikut ke api.py untuk mengaktifkan
fitur Generative AI:

from generative_features import OlahGenerativeFeatures
from pydantic import BaseModel

gen_features = OlahGenerativeFeatures()

class TipsRequest(BaseModel):
    recipe_name: str
    ingredients: List[str]

class SubstituteRequest(BaseModel):
    missing_ingredient: str
    recipe_name: str

class ShelfLifeRequest(BaseModel):
    ingredients: List[str]

@app.post("/ai/tips", tags=["AI Features"])
async def get_cooking_tips(request: TipsRequest):
    return gen_features.get_cooking_tips(request.recipe_name, request.ingredients)

@app.post("/ai/substitute", tags=["AI Features"])
async def get_substitutes(request: SubstituteRequest):
    return gen_features.get_ingredient_substitutes(
        request.missing_ingredient, request.recipe_name
    )

@app.post("/ai/shelf-life", tags=["AI Features"])
async def get_shelf_life(request: ShelfLifeRequest):
    return gen_features.estimate_shelf_life(request.ingredients)
"""


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Pastikan GEMINI_API_KEY sudah di-set
    if not os.getenv("GEMINI_API_KEY"):
        print("[WARNING] GEMINI_API_KEY belum di-set.")
        print("Set dengan: export GEMINI_API_KEY='your-key'")
        print("Atau tambahkan ke file .env")
        exit(1)

    gen = OlahGenerativeFeatures()

    print("\n[TEST 1] Cooking Tips")
    tips = gen.get_cooking_tips("ayam goreng", ["ayam", "bawang putih", "kunyit"])
    print(json.dumps(tips, indent=2, ensure_ascii=False))

    print("\n[TEST 2] Substitusi Bahan")
    sub = gen.get_ingredient_substitutes("santan", "rendang ayam")
    print(json.dumps(sub, indent=2, ensure_ascii=False))

    print("\n[TEST 3] Masa Simpan")
    shelf = gen.estimate_shelf_life(["ayam segar", "tahu", "bayam", "tomat"])
    print(json.dumps(shelf, indent=2, ensure_ascii=False))
