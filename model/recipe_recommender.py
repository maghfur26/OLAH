"""
OLAH - Recipe Recommendation System
AI Engineer | Coding Camp 2026 - CC26-PSU127

Model: Deep Learning dengan TensorFlow Functional API
Task: Content-based Recipe Recommendation berbasis bahan (ingredients)

Arsitektur:
- Embedding + BiLSTM untuk encoding bahan makanan
- Custom Layer (IngredientAttentionLayer, L2NormalizeLayer)
- Custom Loss Function (RecommendationLoss)
- Custom Callback (BestModelCallback)
- Training dengan tf.GradientTape (manual training loop)
- Visualisasi training dengan Matplotlib & Seaborn
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 0. KONFIGURASI
# ─────────────────────────────────────────────

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Styling global matplotlib/seaborn
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12",
           "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]

CONFIG = {
    "data_path":            "../data/Recipes_Clean_Sampled.csv",
    "model_save_path":      "../saved_model/olah_recommender.keras",
    "embeddings_save_path": "../saved_model/recipe_embeddings.npy",
    "metadata_save_path":   "../saved_model/recipe_metadata.json",
    "encoder_save_path":    "../saved_model/label_encoder.pkl",
    "plots_dir":            "../saved_model/plots",  # semua gambar output
    "max_seq_length": 64,
    "embedding_dim":  64,
    "hidden_dim":    128,
    "num_categories":  8,
    "dropout_rate":  0.4,
    "learning_rate": 2e-4,
    "batch_size":     32,
    "epochs":         50,
}


# ─────────────────────────────────────────────
# 1. PREPROCESSING DATA
# ─────────────────────────────────────────────

class DataPreprocessor:
    def __init__(self, config):
        self.config         = config
        self.vocab          = {}
        self.label_encoder  = LabelEncoder()
        self.max_vocab_size = 5000

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        synonyms = {
            "telor": "telur", "cabe": "cabai", "merica": "lada",
            "bawang bombay": "bawang bombei", "minyak goreng": "minyak",
            "ketumbar bubuk": "ketumbar", "santan kara": "santan",
            "gula pasir": "gula", "garam dapur": "garam",
        }
        for wrong, correct in synonyms.items():
            text = text.replace(wrong, correct)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_vocab(self, texts):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in sorted_words[: self.max_vocab_size - 2]:
            vocab[word] = len(vocab)
        self.vocab = vocab
        return vocab

    def text_to_sequence(self, text, max_len=None):
        max_len = max_len or self.config["max_seq_length"]
        tokens  = text.split()
        seq     = [self.vocab.get(t, 1) for t in tokens]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def load_and_prepare(self, data_path):
        df = pd.read_csv(data_path, encoding="latin-1")
        print(f"[DATA] Loaded {len(df)} recipes dari {data_path}")
        df["ingredients_text"] = df["ingredients_cleaned"].apply(self.clean_text)
        df["recipe_name_clean"] = df["recipe_name"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else ""
        )
        df = df[df["ingredients_text"].str.len() > 5].reset_index(drop=True)
        print(f"[DATA] After cleaning: {len(df)} recipes")
        df["category_encoded"] = self.label_encoder.fit_transform(df["category"])
        print(f"[DATA] Categories: {list(self.label_encoder.classes_)}")
        self.build_vocab(df["ingredients_text"].tolist())
        print(f"[DATA] Vocab size: {len(self.vocab)}")
        df["ingredient_seq"] = df["ingredients_text"].apply(self.text_to_sequence)
        return df

    def save_metadata(self, df):
        metadata = []
        for _, row in df.iterrows():
            metadata.append({
                "recipe_name":         row["recipe_name"],
                "category":            row["category"],
                "ingredients_cleaned": row.get("ingredients_cleaned", ""),
                "total_ingredients":   int(row.get("total_ingredients", 0)),
                "love_count":          int(row.get("love_count", 0)),
                "steps":               row.get("steps", ""),
                "url":                 row.get("url", ""),
            })
        os.makedirs(os.path.dirname(self.config["metadata_save_path"]), exist_ok=True)
        with open(self.config["metadata_save_path"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[DATA] Metadata saved: {len(metadata)} recipes")


# ─────────────────────────────────────────────
# 2. CUSTOM COMPONENTS
# ─────────────────────────────────────────────

class IngredientAttentionLayer(keras.layers.Layer):
    """
    CUSTOM LAYER — Ingredient-aware Self-Attention

    Menerima tuple (lstm_out, raw_ids):
      - lstm_out : (batch, seq_len, hidden_dim) — output BiLSTM
      - raw_ids  : (batch, seq_len) — token integer, untuk buat padding mask

    Membuat padding mask sendiri sehingga tidak bergantung mask_zero
    Embedding dan tidak ada mask yang bocor ke BatchNormalization.
    Output: (batch, units) — representasi resep hasil masked mean-pooling.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.W_query = keras.layers.Dense(units, use_bias=False)
        self.W_key   = keras.layers.Dense(units, use_bias=False)
        self.W_value = keras.layers.Dense(units, use_bias=False)
        self.scale   = tf.math.sqrt(tf.cast(units, tf.float32))

    def call(self, inputs, training=False):
        lstm_out, raw_ids = inputs

        # Padding mask: 1.0 di posisi token nyata, 0.0 di PAD (token == 0)
        pad_mask = tf.cast(tf.not_equal(raw_ids, 0), tf.float32)  # (batch, seq)

        Q = self.W_query(lstm_out)  # (batch, seq, units)
        K = self.W_key(lstm_out)    # (batch, seq, units)
        V = self.W_value(lstm_out)  # (batch, seq, units)

        # Scaled dot-product self-attention
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale    # (batch, seq, seq)
        scores += (1.0 - pad_mask[:, tf.newaxis, :]) * (-1e9)      # mask PAD columns

        attn_weights = tf.nn.softmax(scores, axis=-1)               # (batch, seq, seq)
        context      = tf.matmul(attn_weights, V)                   # (batch, seq, units)

        # Masked mean-pooling — rata-rata hanya token nyata
        mask_3d = pad_mask[:, :, tf.newaxis]                        # (batch, seq, 1)
        sum_ctx = tf.reduce_sum(context * mask_3d, axis=1)          # (batch, units)
        count   = tf.reduce_sum(mask_3d, axis=1) + 1e-9             # (batch, 1)
        return sum_ctx / count                                       # (batch, units)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class L2NormalizeLayer(keras.layers.Layer):
    """
    CUSTOM LAYER — L2 Normalization untuk embedding output.
    Membuat cosine similarity = dot product → efisien saat inference.
    Fully serializable ke .keras (tidak pakai Lambda).
    """

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


class RecommendationLoss(keras.losses.Loss):
    """
    CUSTOM LOSS FUNCTION — Classification + Confidence Regularization

    L = CrossEntropy(y_true, y_pred)
      + alpha * (−mean_std(y_pred))

    Term kedua mendorong distribusi prediksi lebih tajam (confident),
    sehingga embedding antar kategori menjadi lebih terpisah.
    """

    def __init__(self, alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.alpha   = alpha
        self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        ce          = self.ce_loss(y_true, y_pred)
        pred_std    = tf.math.reduce_std(y_pred, axis=-1)
        contrastive = -tf.reduce_mean(pred_std)
        return ce + self.alpha * contrastive

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class BestModelCallback(keras.callbacks.Callback):
    """
    CUSTOM CALLBACK — Save Best Model + Early Stopping

    - Simpan model terbaik (val_accuracy tertinggi) ke disk secara otomatis
    - Set flag stop_training setelah `patience` epoch tanpa improvement
    - Ekspor training history ke JSON (dipakai ulang untuk plotting)

    Catatan: flag stop_training dibaca oleh OlahTrainer, bukan Keras internal,
    karena custom GradientTape loop tidak menggunakan .fit().
    """

    def __init__(self, save_path, patience=10, min_delta=0.001):
        super().__init__()
        self.save_path     = save_path
        self.patience      = patience
        self.min_delta     = min_delta
        self.best_val_acc  = 0.0
        self.wait          = 0
        self.best_epoch    = 0
        self.history_log   = []
        self.stop_training = False   

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)

        self.history_log.append({
            "epoch":      epoch + 1,
            "train_acc":  float(logs.get("accuracy", 0.0)),
            "train_loss": float(logs.get("loss", 0.0)),
            "val_acc":    float(val_acc),
            "val_loss":   float(logs.get("val_loss", 0.0)),
        })

        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch + 1
            self.wait         = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            print(f"\n[CALLBACK] ✓ Best model saved epoch {epoch+1} "
                  f"| val_acc={val_acc:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                print(f"\n[CALLBACK] Early stop epoch {epoch+1}. "
                      f"Best: epoch {self.best_epoch} val_acc={self.best_val_acc:.4f}")

    def on_train_end(self, logs=None):
        print(f"\n[CALLBACK] Selesai. Best val_acc: {self.best_val_acc:.4f}")
        log_dir      = os.path.dirname(self.save_path)
        history_path = os.path.join(log_dir, "training_history.json")
        os.makedirs(log_dir, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.history_log, f, indent=2)


# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE (TF Functional API)
# ─────────────────────────────────────────────

def build_olah_model(vocab_size, num_categories,
                     max_seq_length=64, embedding_dim=64,
                     hidden_dim=128, dropout_rate=0.3):
    """
    OLAH Recommender — TensorFlow Functional API

    Input  : ingredient_input (batch, seq_len) — token ids
    Output : [embedding_output (batch, hidden_dim),   ← cosine similarity
              category_output  (batch, num_categories)] ← training supervised
    """

    ingredient_input = keras.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="ingredient_input"
    )

    # Embedding — mask_zero=False, padding mask dikelola manual di Attention layer
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="ingredient_embedding",
    )(ingredient_input)

    x = keras.layers.SpatialDropout1D(dropout_rate)(x)

    # BiLSTM — tangkap konteks urutan bahan
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_dim // 2, return_sequences=True, dropout=0.2),
        name="bilstm",
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_dim // 4, return_sequences=True, dropout=0.2),
        name="bilstm_2",
    )(x)

    # Custom Attention Layer — tuple (lstm_out, raw_ids) untuk padding mask mandiri
    ingredient_repr = IngredientAttentionLayer(
        units=hidden_dim // 2, name="ingredient_attention"
    )([x, ingredient_input])
    # Output: (batch, hidden_dim//2) — tanpa mask bocor ke BatchNorm

    # Dense projection
    ingredient_repr = keras.layers.Dense(hidden_dim, name="projection")(ingredient_repr)
    ingredient_repr = keras.layers.BatchNormalization(name="batch_norm")(ingredient_repr)
    ingredient_repr = keras.layers.Dropout(dropout_rate)(ingredient_repr)
    ingredient_repr = keras.layers.Activation("relu")(ingredient_repr)

    # Embedding output — L2 normalized untuk cosine similarity search
    embedding_output = L2NormalizeLayer(name="embedding_output")(ingredient_repr)

    # Classification head — untuk training supervised
    x_cls = keras.layers.Dense(hidden_dim // 2, activation="relu")(ingredient_repr)
    x_cls = keras.layers.Dropout(dropout_rate)(x_cls)
    category_output = keras.layers.Dense(
        num_categories, activation="softmax", name="category_output"
    )(x_cls)

    model = keras.Model(
        inputs=ingredient_input,
        outputs=[embedding_output, category_output],
        name="OLAH_Recommender",
    )
    return model


# ─────────────────────────────────────────────
# 4. CUSTOM TRAINING LOOP (tf.GradientTape)
# ─────────────────────────────────────────────

class OlahTrainer:
    """
    Custom training loop menggunakan tf.GradientTape.
    Kontrol penuh atas gradient, logging, dan early stopping.
    """

    def __init__(self, model, config):
        self.model     = model
        self.config    = config
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config["learning_rate"], clipnorm=1.0
        )
        self.loss_fn        = RecommendationLoss(alpha=0.3, name="recommendation_loss")
        self.acc_metric     = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            _, category_out = self.model(x_batch, training=True)
            loss = self.loss_fn(y_batch, category_out)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.acc_metric.update_state(y_batch, category_out)
        return loss

    @tf.function
    def val_step(self, x_batch, y_batch):
        _, category_out = self.model(x_batch, training=False)
        val_loss = self.loss_fn(y_batch, category_out)
        self.val_acc_metric.update_state(y_batch, category_out)
        return val_loss

    def train(self, X_train, y_train, X_val, y_val,
              epochs, batch_size, callback):

        train_ds = (tf.data.Dataset
                    .from_tensor_slices((X_train, y_train))
                    .shuffle(len(X_train), seed=RANDOM_SEED)
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        val_ds   = (tf.data.Dataset
                    .from_tensor_slices((X_val, y_val))
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE))

        callback.set_model(self.model)

        print(f"\n{'='*62}")
        print(f"  OLAH Recommender — Custom Training Loop (GradientTape)")
        print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {self.config['learning_rate']}")
        print(f"{'='*62}\n")

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            self.acc_metric.reset_state()
            epoch_losses = []

            for x_batch, y_batch in train_ds:
                loss = self.train_step(x_batch, y_batch)
                epoch_losses.append(float(loss))

            train_loss = float(np.mean(epoch_losses))
            train_acc  = float(self.acc_metric.result())

            self.val_acc_metric.reset_state()
            val_losses = []

            for x_batch, y_batch in val_ds:
                val_loss = self.val_step(x_batch, y_batch)
                val_losses.append(float(val_loss))

            val_loss_mean = float(np.mean(val_losses))
            val_acc       = float(self.val_acc_metric.result())

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss_mean)
            history["val_accuracy"].append(val_acc)

            print(f"Epoch {epoch+1:3d}/{epochs} — "
                  f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss_mean:.4f} | val_acc: {val_acc:.4f}")

            callback.on_epoch_end(epoch, {
                "loss": train_loss, "accuracy": train_acc,
                "val_loss": val_loss_mean, "val_accuracy": val_acc,
            })

            if callback.stop_training:
                break

        callback.on_train_end()
        return history


# ─────────────────────────────────────────────
# 5. VISUALISASI (Matplotlib + Seaborn)
#    Menggantikan TensorBoard sepenuhnya
# ─────────────────────────────────────────────

class TrainingVisualizer:
    """
    Semua visualisasi training & evaluasi menggunakan Matplotlib & Seaborn.
    Setiap plot disimpan otomatis ke CONFIG['plots_dir'] sebagai file PNG.

    Plot yang dihasilkan:
        01_eda_dashboard.png       — distribusi data sebelum training
        02_training_history.png   — akurasi & loss per epoch
        03_confusion_matrix.png   — confusion matrix test set
        04_per_class_metrics.png  — precision, recall, F1 per kategori
        05_summary_report.png     — ringkasan performa model
    """

    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {path}")

    # ── 5a. EDA Dashboard ────────────────────

    def plot_eda(self, df: pd.DataFrame):
        """Dashboard EDA: distribusi kategori, jumlah bahan, top ingredients."""
        from collections import Counter

        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
        fig.suptitle("OLAH — Exploratory Data Analysis (EDA)",
                     fontsize=15, fontweight="bold", y=1.01)

        cat_counts = df["category"].value_counts()

        # 1. Bar chart distribusi kategori
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(cat_counts.index, cat_counts.values,
                       color=PALETTE[:len(cat_counts)], edgecolor="white", linewidth=0.8)
        ax1.set_title("Distribusi Resep per Kategori", fontweight="bold", fontsize=12)
        ax1.set_xlabel("Kategori"); ax1.set_ylabel("Jumlah Resep")
        ax1.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, cat_counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 3, str(val),
                     ha="center", fontsize=9, fontweight="bold")

        # 2. Pie chart proporsi
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.pie(cat_counts.values, labels=cat_counts.index,
                colors=PALETTE[:len(cat_counts)],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9})
        ax2.set_title("Proporsi Kategori", fontweight="bold", fontsize=12)

        # 3. Distribusi jumlah bahan per resep
        ax3 = fig.add_subplot(gs[0, 2])
        col = "total_ingredients" if "total_ingredients" in df.columns else None
        if col:
            ax3.hist(df[col], bins=25, color=PALETTE[1],
                     edgecolor="white", alpha=0.85)
            ax3.axvline(df[col].mean(), color="red", linestyle="--",
                        linewidth=1.5, label=f"Mean={df[col].mean():.1f}")
            ax3.axvline(df[col].median(), color="orange", linestyle="--",
                        linewidth=1.5, label=f"Median={int(df[col].median())}")
            ax3.set_title("Distribusi Jumlah Bahan / Resep",
                          fontweight="bold", fontsize=12)
            ax3.set_xlabel("Jumlah Bahan"); ax3.set_ylabel("Frekuensi")
            ax3.legend(fontsize=9)

        # 4. Top 15 bahan terbanyak
        ax4 = fig.add_subplot(gs[1, :2])
        all_ings = []
        for row in df["ingredients_cleaned"].dropna():
            for ing in str(row).split(","):
                c = ing.strip().lower()
                if len(c) > 2:
                    all_ings.append(c)
        top15  = Counter(all_ings).most_common(15)
        names, counts = zip(*top15)
        ybars = ax4.barh(range(len(names)), counts,
                         color=sns.color_palette("Blues_d", len(names)))
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=9)
        ax4.invert_yaxis()
        ax4.set_xlabel("Frekuensi")
        ax4.set_title("Top 15 Bahan Paling Sering Digunakan",
                      fontweight="bold", fontsize=12)
        for bar, cnt in zip(ybars, counts):
            ax4.text(bar.get_width() + 1,
                     bar.get_y() + bar.get_height() / 2,
                     str(cnt), va="center", fontsize=8)

        # 5. Rata-rata love count per kategori
        ax5 = fig.add_subplot(gs[1, 2])
        if "love_count" in df.columns:
            love = (df.groupby("category")["love_count"]
                      .mean().sort_values(ascending=True))
            ax5.barh(love.index, love.values,
                     color=PALETTE[:len(love)])
            ax5.set_title("Rata-rata Love Count\nper Kategori",
                          fontweight="bold", fontsize=12)
            ax5.set_xlabel("Rata-rata Love Count")

        self._save(fig, "01_eda_dashboard.png")

    # ── 5b. Training History ─────────────────

    def plot_training_history(self, history: dict):
        """
        Plot akurasi & loss tiap epoch (train vs val).
        Pengganti TensorBoard — semua info training dalam satu gambar.
        """
        epochs_ran = list(range(1, len(history["accuracy"]) + 1))
        best_epoch = int(np.argmax(history["val_accuracy"])) + 1
        best_vacc  = max(history["val_accuracy"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("OLAH — Training History (tf.GradientTape)",
                     fontsize=14, fontweight="bold")

        # — Accuracy —
        ax = axes[0]
        ax.plot(epochs_ran, history["accuracy"],
                label="Train Accuracy", color=PALETTE[0],
                linewidth=2, marker="o", markersize=3)
        ax.plot(epochs_ran, history["val_accuracy"],
                label="Val Accuracy", color=PALETTE[1],
                linewidth=2, linestyle="--", marker="s", markersize=3)
        ax.axhline(y=0.85, color="red", linestyle=":", linewidth=1.5,
                   label="Target 85%")
        ax.axvline(x=best_epoch, color="gray", linestyle=":",
                   linewidth=1.2, label=f"Best epoch {best_epoch}")
        # Anotasi titik terbaik
        offset_x = max(1, len(epochs_ran) * 0.05)
        ax.annotate(
            f"Best val_acc\n{best_vacc:.4f}",
            xy=(best_epoch, best_vacc),
            xytext=(best_epoch + offset_x, best_vacc - 0.08),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray",
        )
        ax.set_title("Akurasi per Epoch", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

        # — Loss —
        ax = axes[1]
        ax.plot(epochs_ran, history["loss"],
                label="Train Loss", color=PALETTE[2],
                linewidth=2, marker="o", markersize=3)
        ax.plot(epochs_ran, history["val_loss"],
                label="Val Loss", color=PALETTE[3],
                linewidth=2, linestyle="--", marker="s", markersize=3)
        ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title("Loss per Epoch", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

        plt.tight_layout()
        self._save(fig, "02_training_history.png")

    # ── 5c. Confusion Matrix ─────────────────

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Confusion matrix count & normalized berdampingan."""
        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("OLAH — Confusion Matrix (Test Set)",
                     fontsize=14, fontweight="bold")

        for ax, data, fmt, title, cmap in zip(
            axes,
            [cm, cm_pct],
            ["d", ".2f"],
            ["Count", "Normalized (per baris)"],
            ["Blues", "Greens"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(f"Confusion Matrix — {title}",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        self._save(fig, "03_confusion_matrix.png")

    # ── 5d. Per-class Metrics ────────────────

    def plot_classification_report(self, y_true, y_pred, class_names):
        """Bar chart grouped: Precision, Recall, F1 per kategori."""
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        prec = [report[c]["precision"] for c in class_names]
        rec  = [report[c]["recall"]    for c in class_names]
        f1   = [report[c]["f1-score"]  for c in class_names]

        x = np.arange(len(class_names))
        w = 0.25

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(x - w, prec, w, label="Precision", color=PALETTE[0], alpha=0.85)
        ax.bar(x,     rec,  w, label="Recall",    color=PALETTE[1], alpha=0.85)
        ax.bar(x + w, f1,   w, label="F1-Score",  color=PALETTE[2], alpha=0.85)
        ax.axhline(y=0.85, color="red", linestyle="--",
                   linewidth=1.2, label="Target 85%")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, fontsize=10)
        ax.set_ylim([0, 1.12])
        ax.set_ylabel("Score"); ax.set_xlabel("Kategori")
        ax.set_title("Per-class Metrics — Precision, Recall, F1-Score",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
        # Anotasi nilai F1
        for xi, f in zip(x + w, f1):
            ax.text(xi, f + 0.012, f"{f:.2f}",
                    ha="center", fontsize=8, color="#333")

        plt.tight_layout()
        self._save(fig, "04_per_class_metrics.png")

    # ── 5e. Summary Report ───────────────────

    def plot_summary(self, test_acc, best_val_acc, total_epochs,
                     class_names, y_true, y_pred):
        """Satu halaman ringkasan hasil model — siap masuk laporan."""
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        fig = plt.figure(figsize=(14, 7))
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
        fig.suptitle("OLAH Recommender — Model Summary Report",
                     fontsize=15, fontweight="bold")

        # Panel kiri: tabel metrik utama
        ax_left = fig.add_subplot(gs[0, 0])
        ax_left.axis("off")
        status   = "✓ PASSED" if test_acc >= 0.85 else "✗ Tambah data/epoch"
        metrics  = [
            ("Test Accuracy",     f"{test_acc * 100:.2f}%"),
            ("Best Val Accuracy", f"{best_val_acc * 100:.2f}%"),
            ("Total Epochs",      str(total_epochs)),
            ("Target Accuracy",   "≥ 85.00%"),
            ("Status",            status),
            ("Macro F1",          f"{report['macro avg']['f1-score'] * 100:.2f}%"),
            ("Macro Precision",   f"{report['macro avg']['precision'] * 100:.2f}%"),
            ("Macro Recall",      f"{report['macro avg']['recall'] * 100:.2f}%"),
        ]
        tbl = ax_left.table(
            cellText=metrics,
            colLabels=["Metrik", "Nilai"],
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.2)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2ECC71")
                cell.set_text_props(color="white", fontweight="bold")
            elif r == 5:  # row Status
                cell.set_facecolor(
                    "#D5F5E3" if test_acc >= 0.85 else "#FADBD8"
                )
            else:
                cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "white")
        ax_left.set_title("Ringkasan Performa Model",
                          fontsize=12, fontweight="bold", pad=15)

        # Panel kanan: F1 per kategori horizontal bar
        ax_right = fig.add_subplot(gs[0, 1])
        f1_scores = [report[c]["f1-score"] for c in class_names]
        colors    = [PALETTE[0] if f >= 0.85 else PALETTE[2] for f in f1_scores]
        bars = ax_right.barh(class_names, f1_scores,
                              color=colors, edgecolor="white")
        ax_right.axvline(x=0.85, color="red", linestyle="--",
                          linewidth=1.5, label="Target 85%")
        ax_right.set_xlim([0, 1.12])
        ax_right.set_xlabel("F1-Score")
        ax_right.set_title("F1-Score per Kategori",
                            fontsize=12, fontweight="bold")
        ax_right.legend(fontsize=9)
        ax_right.grid(axis="x", alpha=0.4)
        for bar, val in zip(bars, f1_scores):
            ax_right.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
            )

        self._save(fig, "05_summary_report.png")


# ─────────────────────────────────────────────
# 6. EMBEDDING EXTRACTION
# ─────────────────────────────────────────────

class RecipeEmbeddingEngine:
    def __init__(self, model, config):
        self.model  = model
        self.config = config
        self.embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer("embedding_output").output,
            name="embedding_extractor",
        )

    def extract_all_embeddings(self, X_all, batch_size=64):
        embeddings = []
        for i in range(0, len(X_all), batch_size):
            batch = X_all[i: i + batch_size]
            emb   = self.embedding_model(batch, training=False)
            embeddings.append(emb.numpy())
        return np.vstack(embeddings)

    def save_embeddings(self, embeddings):
        os.makedirs(os.path.dirname(self.config["embeddings_save_path"]), exist_ok=True)
        np.save(self.config["embeddings_save_path"], embeddings)
        print(f"[EMBED] Embeddings saved: shape={embeddings.shape}")


# ─────────────────────────────────────────────
# 7. MAIN — TRAINING PIPELINE
# ─────────────────────────────────────────────

CUSTOM_OBJECTS = {
    "IngredientAttentionLayer": IngredientAttentionLayer,
    "L2NormalizeLayer":         L2NormalizeLayer,
    "RecommendationLoss":       RecommendationLoss,
}


def main():
    print("\n" + "=" * 62)
    print("  OLAH - Recipe Recommender Training Pipeline")
    print("  Coding Camp 2026 | CC26-PSU127")
    print("=" * 62 + "\n")

    viz = TrainingVisualizer(CONFIG["plots_dir"])

    # ── Data ──────────────────────────────────
    preprocessor = DataPreprocessor(CONFIG)
    df           = preprocessor.load_and_prepare(CONFIG["data_path"])

    print("\n[VIZ] Generating EDA plots...")
    viz.plot_eda(df)

    X = np.array(df["ingredient_seq"].tolist(), dtype=np.int32)
    y = df["category_encoded"].values.astype(np.int32)
    print(f"\n[SPLIT] X: {X.shape} | y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train,
        random_state=RANDOM_SEED
    )
    print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Model ──────────────────────────────────
    model = build_olah_model(
        vocab_size     = len(preprocessor.vocab),
        num_categories = CONFIG["num_categories"],
        max_seq_length = CONFIG["max_seq_length"],
        embedding_dim  = CONFIG["embedding_dim"],
        hidden_dim     = CONFIG["hidden_dim"],
        dropout_rate   = CONFIG["dropout_rate"],
    )
    model.summary()

    # ── Training ───────────────────────────────
    callback = BestModelCallback(
        save_path = CONFIG["model_save_path"],
        patience  = 10,
        min_delta = 0.001,
    )
    trainer = OlahTrainer(model, CONFIG)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs     = CONFIG["epochs"],
        batch_size = CONFIG["batch_size"],
        callback   = callback,
    )

    # ── Plot training history ──────────────────
    print("\n[VIZ] Generating training history plot...")
    viz.plot_training_history(history)

    # ── Evaluasi test set ──────────────────────
    print("\n[EVAL] Loading best model untuk evaluasi test set...")
    best_model = keras.models.load_model(
        CONFIG["model_save_path"], custom_objects=CUSTOM_OBJECTS
    )

    _, test_probs = best_model(X_test, training=False)
    y_pred     = np.argmax(test_probs.numpy(), axis=1)
    test_acc   = float(np.mean(y_pred == y_test))
    class_names = list(preprocessor.label_encoder.classes_)

    print(f"\n{'='*47}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Target        : >= 85.00%")
    print(f"  Status        : {'✓ PASSED' if test_acc >= 0.85 else '→ Tambah data/epoch'}")
    print(f"{'='*47}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Plot evaluasi ──────────────────────────
    print("[VIZ] Generating evaluation plots...")
    viz.plot_confusion_matrix(y_test, y_pred, class_names)
    viz.plot_classification_report(y_test, y_pred, class_names)
    viz.plot_summary(
        test_acc     = test_acc,
        best_val_acc = callback.best_val_acc,
        total_epochs = len(history["accuracy"]),
        class_names  = class_names,
        y_true       = y_test,
        y_pred       = y_pred,
    )

    # ── Simpan embeddings & artefak ───────────
    print("\n[EMBED] Mengekstrak embeddings semua resep...")
    engine         = RecipeEmbeddingEngine(best_model, CONFIG)
    all_embeddings = engine.extract_all_embeddings(X)
    engine.save_embeddings(all_embeddings)

    preprocessor.save_metadata(df)

    os.makedirs(os.path.dirname(CONFIG["encoder_save_path"]), exist_ok=True)
    with open(CONFIG["encoder_save_path"], "wb") as f:
        pickle.dump({
            "label_encoder": preprocessor.label_encoder,
            "vocab":         preprocessor.vocab,
            "config":        CONFIG,
        }, f)

    print(f"\n[DONE] Model  : {CONFIG['model_save_path']}")
    print(f"[DONE] Plots  : {CONFIG['plots_dir']}/")
    print(f"         01_eda_dashboard.png")
    print(f"         02_training_history.png")
    print(f"         03_confusion_matrix.png")
    print(f"         04_per_class_metrics.png")
    print(f"         05_summary_report.png")
    print(f"[DONE] Training selesai!\n")
    return history, test_acc


if __name__ == "__main__":
    history, test_acc = main()