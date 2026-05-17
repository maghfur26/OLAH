# 🍳 OLAH — AI Engineer Module
**Coding Camp 2026 | CC26-PSU127 | Sustainable Living & Responsible Consumption**

> *Punya sisa bahan makanan? di-OLAH aja!*

---

## 📁 Struktur Folder

```
olah_ai/
├── model/
│   ├── recipe_recommender.py   # Model utama + training pipeline
│   └── inference.py            # Engine inference & similarity search
├── api/
│   ├── api.py                  # FastAPI REST API
│   └── generative_features.py  # Gemini AI features (side quest)
├── notebooks/
│   └── olah_training.ipynb     # Notebook EDA + training
├── data/
│   └── Recipes_Clean_Sampled.csv  # Dataset (copy ke sini)
├── saved_model/                # Diisi otomatis setelah training
│   ├── olah_recommender.keras
│   ├── recipe_embeddings.npy
│   ├── recipe_metadata.json
│   └── label_encoder.pkl
├── tensorboard_logs/           # Diisi otomatis saat training
├── requirements.txt
└── README.md
```

---

## ✅ Checklist Main Quest

| Quest | Status | File |
|-------|--------|------|
| TensorFlow Functional API | ✅ | `model/recipe_recommender.py` → `build_olah_model()` |
| Custom Layer | ✅ | `IngredientAttentionLayer` |
| Custom Loss Function | ✅ | `RecommendationLoss` |
| Custom Callback | ✅ | `BestModelCallback` |
| Save model `.keras` | ✅ | `saved_model/olah_recommender.keras` |
| Inference code | ✅ | `model/inference.py` |

## ✅ Checklist Side Quest

| Quest | Status | File |
|-------|--------|------|
| REST API (FastAPI) | ✅ | `api/api.py` |
| `tf.GradientTape` training loop | ✅ | `OlahTrainer` class |
| Generative AI (Gemini) | ✅ | `api/generative_features.py` |
| TensorBoard integration | ✅ | `OlahTrainer` → `summary_writer` |
| Accuracy ≥ 85% | 🎯 | Target — evaluasi setelah training |

---

## 🚀 Cara Menjalankan

### 1. Setup Environment
```bash
# Clone repo & masuk ke folder AI
cd olah_ai

# Install dependencies
pip install -r requirements.txt

# Copy dataset
cp /path/to/Recipes_Clean_Sampled.csv data/
```

### 2. Training Model
```bash
cd model
python recipe_recommender.py
```

Output yang diharapkan:
```
[DATA] Loaded 5615 recipes
[DATA] Categories: ['ayam', 'ikan', 'kambing', 'sapi', 'tahu', 'telur', 'tempe', 'udang']
...
Epoch  1/30 — loss: 1.8234 — acc: 0.3210 — val_loss: 1.5432 — val_acc: 0.4521
...
[CALLBACK] ✓ Best model saved at epoch 12 | val_acc=0.8754
...
Test Accuracy : 0.8812 (88.12%)
Target        : >= 0.85 (85%)
Status        : ✓ PASSED
```

### 3. Jalankan API
```bash
cd api
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Akses dokumentasi API: **http://localhost:8000/docs**

### 4. Monitor dengan TensorBoard
```bash
tensorboard --logdir olah_ai/tensorboard_logs
# Buka: http://localhost:6006
```

### 5. Setup Gemini AI (Opsional)
```bash
export GEMINI_API_KEY="your-api-key-here"
# Dapatkan API key gratis di: https://aistudio.google.com/
```

---

## 🔌 API Endpoints

### Core Endpoints
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Cek status API & model |
| `GET` | `/categories` | List kategori tersedia |
| `POST` | `/recommend` | **Rekomendasi resep utama** |
| `POST` | `/similar` | Resep mirip berdasarkan nama |
| `GET` | `/recipe/popular` | Resep populer |
| `GET` | `/recipe/random` | Resep random |

### AI Feature Endpoints (Gemini)
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/ai/tips` | Tips memasak otomatis |
| `POST` | `/ai/substitute` | Substitusi bahan |
| `POST` | `/ai/shelf-life` | Estimasi masa simpan |

### Contoh Request `/recommend`
```json
POST /recommend
{
  "ingredients": ["ayam", "bawang merah", "bawang putih", "kemiri", "santan"],
  "top_k": 10,
  "category_filter": "ayam",
  "min_similarity": 0.1
}
```

### Contoh Response
```json
{
  "status": "success",
  "query_ingredients": ["ayam", "bawang merah", "bawang putih", "kemiri", "santan"],
  "total_results": 10,
  "processing_time_ms": 45.2,
  "recommendations": [
    {
      "recipe_name": "opor ayam",
      "category": "ayam",
      "similarity_score": 0.9234,
      "match_count": 5,
      "match_percentage": 100.0,
      "matched_ingredients": ["ayam", "bawang merah", "bawang putih", "kemiri", "santan"],
      "love_count": 245,
      "ingredients_cleaned": "ayam, bawang merah, bawang putih, kemiri, ..."
    }
  ]
}
```

---

## 🏗️ Arsitektur Model

```
Input: ingredient_sequence (batch, 128)
    │
    ▼
Embedding Layer (vocab_size, 128)
    │
    ▼
SpatialDropout1D
    │
    ▼
BiLSTM (256 → 128)  ← tangkap konteks urutan bahan
    │
    ▼
BiLSTM (128 → 64)
    │
    ▼
IngredientAttentionLayer [CUSTOM LAYER]  ← bobot bahan penting
    │
    ▼
Dense(256) + BatchNorm + Dropout
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
embedding_output (128, L2-norm)   category_output (8, softmax)
[untuk similarity search]          [untuk training supervised]
```

**Custom Components:**
- **`IngredientAttentionLayer`** — Self-attention untuk menentukan bahan mana yang paling informatif
- **`RecommendationLoss`** — CrossEntropy + contrastive term untuk embedding yang lebih baik
- **`BestModelCallback`** — Auto-save model terbaik + adaptive early stopping

---

## 🔗 Integrasi dengan Full-Stack Team

### Endpoint yang dikonsumsi Frontend:
```javascript
// Rekomendasi resep
const response = await fetch('http://api-url/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ingredients: userIngredients,  // dari state bahan user
    top_k: 10,
    category_filter: selectedCategory  // null jika tidak ada filter
  })
});
const { recommendations } = await response.json();
```

### Environment Variables untuk Backend (Node.js/Express):
```env
AI_API_URL=http://localhost:8000
AI_API_TIMEOUT=10000
```

### Environment Variables untuk AI Module:
```env
GEMINI_API_KEY=your-gemini-api-key
MODEL_PATH=./saved_model/olah_recommender.keras
EMBEDDINGS_PATH=./saved_model/recipe_embeddings.npy
METADATA_PATH=./saved_model/recipe_metadata.json
ENCODER_PATH=./saved_model/label_encoder.pkl
```

---

## 📊 Target Performa

| Metrik | Target | Keterangan |
|--------|--------|------------|
| Test Accuracy | ≥ 85% | Klasifikasi kategori resep |
| Inference Time | < 100ms | Per request rekomendasi |
| Top-10 Relevance | ≥ 80% | Resep relevan dengan bahan input |

---

## 👥 Tim AI Engineer

- **CACC193D6Y0545** — Maghfur Hasani
- **CACC319D6X0506** — Angelin Viona Lumban Tobing

---

*Dibuat untuk Coding Camp 2026 powered by DBS Foundation*
