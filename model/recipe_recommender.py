"""
OLAH - Recipe Recommendation System
AI Engineer | Coding Camp 2026 - CC26-PSU127

Model: Deep Learning dengan TensorFlow Functional API
Task: Content-based Recipe Recommendation berbasis bahan (ingredients)

Arsitektur:
- Embedding + BiLSTM untuk encoding bahan makanan
- Custom Layer (IngredientAttentionLayer)
- Custom Loss Function (RecommendationLoss)
- Custom Callback (BestModelCallback)
- Training dengan tf.GradientTape (manual training loop)
- TensorBoard logging
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIG = {
    "data_path": "../data/Recipes_Clean_Sampled.csv",
    "model_save_path": "../saved_model/olah_recommender.keras",
    "embeddings_save_path": "../saved_model/recipe_embeddings.npy",
    "metadata_save_path": "../saved_model/recipe_metadata.json",
    "encoder_save_path": "../saved_model/label_encoder.pkl",
    "max_seq_length": 64,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_categories": 8,
    "dropout_rate": 0.3,
    "learning_rate": 2e-4,
    "batch_size": 32,
    "epochs": 50,
    "tensorboard_log_dir": "./tensorboard_logs",
}


# ─────────────────────────────────────────────
# 1. PREPROCESSING DATA
# ─────────────────────────────────────────────

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.vocab = {}
        self.label_encoder = LabelEncoder()
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
        tokens = text.split()
        seq = [self.vocab.get(t, 1) for t in tokens]
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
                "recipe_name": row["recipe_name"],
                "category": row["category"],
                "ingredients_cleaned": row.get("ingredients_cleaned", ""),
                "total_ingredients": int(row.get("total_ingredients", 0)),
                "love_count": int(row.get("love_count", 0)),
                "steps": row.get("steps", ""),
                "url": row.get("url", ""),
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

    Menghitung attention weight untuk setiap token dalam sequence bahan
    makanan. Mask padding dibuat secara manual dari input (token == 0),
    sehingga TIDAK bergantung pada mask_zero dari Embedding dan tidak
    ada mask yang bocor ke layer downstream.

    Input : (batch, seq_len, hidden_dim)  — sudah di-encode oleh BiLSTM
    Output: (batch, units)               — representasi resep hasil pooling
    """

    # supports_masking = False (default) — layer ini TIDAK meneruskan mask
    # ke downstream, sehingga BatchNormalization aman menerima outputnya.

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_query = keras.layers.Dense(units, use_bias=False)
        self.W_key   = keras.layers.Dense(units, use_bias=False)
        self.W_value = keras.layers.Dense(units, use_bias=False)
        self.scale   = tf.math.sqrt(tf.cast(units, tf.float32))

    def call(self, inputs, training=False):
        """
        inputs: tuple (lstm_out, raw_input_ids)
          - lstm_out     : (batch, seq_len, hidden_dim)
          - raw_input_ids: (batch, seq_len) — integer token ids sebelum embedding
        """
        lstm_out, raw_ids = inputs

        # Buat padding mask: True di posisi token nyata, False di posisi PAD
        # Shape: (batch, seq_len)
        pad_mask = tf.cast(tf.not_equal(raw_ids, 0), tf.float32)

        Q = self.W_query(lstm_out)  # (batch, seq, units)
        K = self.W_key(lstm_out)    # (batch, seq, units)
        V = self.W_value(lstm_out)  # (batch, seq, units)

        # Scaled dot-product self-attention
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale  # (batch, seq, seq)

        # Terapkan padding mask: posisi PAD diberi skor sangat negatif
        mask_expanded = pad_mask[:, tf.newaxis, :]        # (batch, 1, seq)
        scores += (1.0 - mask_expanded) * (-1e9)

        attn_weights = tf.nn.softmax(scores, axis=-1)     # (batch, seq, seq)
        context      = tf.matmul(attn_weights, V)          # (batch, seq, units)

        # Masked mean-pooling: rata-rata hanya token nyata
        mask_3d     = pad_mask[:, :, tf.newaxis]           # (batch, seq, 1)
        masked_ctx  = context * mask_3d                    # zero out PAD positions
        sum_ctx     = tf.reduce_sum(masked_ctx, axis=1)    # (batch, units)
        count       = tf.reduce_sum(mask_3d, axis=1) + 1e-9  # (batch, 1)
        output      = sum_ctx / count                       # (batch, units)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class L2NormalizeLayer(keras.layers.Layer):
    """
    CUSTOM LAYER — L2 Normalization untuk output embedding.
    Dipakai untuk cosine similarity search saat inference.
    Menggantikan Lambda agar fully serializable (.keras format).
    """
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


class RecommendationLoss(keras.losses.Loss):
    """
    CUSTOM LOSS FUNCTION — Classification + Confidence Regularization

    L = CrossEntropy(y_true, y_pred)
      + alpha * (-mean_std(y_pred))

    Term kedua mendorong distribusi prediksi lebih tajam (confident),
    yang secara tidak langsung membuat embedding lebih diskriminatif.
    """

    def __init__(self, alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
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

    - Simpan model terbaik (val_accuracy tertinggi) ke disk otomatis
    - Hentikan training jika tidak ada peningkatan setelah `patience` epoch
    - Ekspor training history ke JSON
    """

    def __init__(self, save_path, patience=10, min_delta=0.001):
        super().__init__()
        self.save_path    = save_path
        self.patience     = patience
        self.min_delta    = min_delta
        self.best_val_acc = 0.0
        self.wait         = 0
        self.best_epoch   = 0
        self.history_log  = []
        # Flag dibaca oleh OlahTrainer — tidak pakai model.stop_training
        # karena Keras Functional model tidak expose atribut itu di luar .fit()
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)

        self.history_log.append({
            "epoch":     epoch + 1,
            "train_acc": float(logs.get("accuracy", 0.0)),
            "val_acc":   float(val_acc),
            "val_loss":  float(logs.get("val_loss", 0.0)),
        })

        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch + 1
            self.wait         = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            print(f"\n[CALLBACK] ✓ Best model saved epoch {epoch+1} | val_acc={val_acc:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                print(f"\n[CALLBACK] Early stop epoch {epoch+1}. Best: epoch {self.best_epoch} val_acc={self.best_val_acc:.4f}")

    def on_train_end(self, logs=None):
        print(f"\n[CALLBACK] Selesai. Best val_acc: {self.best_val_acc:.4f}")
        log_dir      = os.path.dirname(self.save_path)
        history_path = os.path.join(log_dir, "training_history.json")
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

    Dua input dipakai agar IngredientAttentionLayer bisa membuat
    padding mask sendiri tanpa bergantung pada mask_zero Embedding
    (yang menyebabkan mask bocor ke BatchNormalization dan crash).

    Input:
        ingredient_input  : (batch, seq_len) — token ids
    Internal:
        [ingredient_input, ingredient_input] dikirim ke Attention
        agar layer bisa akses raw ids untuk membuat padding mask
    Output:
        embedding_output  : (batch, hidden_dim) — L2-normalized, untuk cosine sim
        category_output   : (batch, num_categories) — softmax, untuk training
    """

    ingredient_input = keras.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="ingredient_input"
    )

    # Embedding — TANPA mask_zero agar tidak ada mask yang bocor downstream
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,          # mask dikelola manual di IngredientAttentionLayer
        name="ingredient_embedding",
    )(ingredient_input)

    x = keras.layers.SpatialDropout1D(dropout_rate)(x)

    # BiLSTM untuk tangkap konteks urutan bahan
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_dim // 2, return_sequences=True, dropout=0.2),
        name="bilstm",
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_dim // 4, return_sequences=True, dropout=0.2),
        name="bilstm_2",
    )(x)

    # Custom Attention Layer — menerima (lstm_out, raw_ids) sebagai tuple
    # Raw ids dipakai untuk membuat padding mask (token == 0 → PAD)
    ingredient_repr = IngredientAttentionLayer(
        units=hidden_dim // 2, name="ingredient_attention"
    )([x, ingredient_input])
    # Output: (batch, hidden_dim//2) — TANPA mask, aman untuk BatchNorm

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

        os.makedirs(config["tensorboard_log_dir"], exist_ok=True)
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(config["tensorboard_log_dir"], "train")
        )
        self.val_writer = tf.summary.create_file_writer(
            os.path.join(config["tensorboard_log_dir"], "val")
        )

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
        callback.on_train_begin()

        print(f"\n{'='*60}")
        print(f"  OLAH Recommender — Custom Training Loop (GradientTape)")
        print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {self.config['learning_rate']}")
        print(f"{'='*60}\n")

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

            with self.train_writer.as_default():
                tf.summary.scalar("loss",     train_loss, step=epoch)
                tf.summary.scalar("accuracy", train_acc,  step=epoch)

            with self.val_writer.as_default():
                tf.summary.scalar("loss",     val_loss_mean, step=epoch)
                tf.summary.scalar("accuracy", val_acc,       step=epoch)

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
# 5. EMBEDDING EXTRACTION & SIMILARITY ENGINE
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
# 6. MAIN — TRAINING PIPELINE
# ─────────────────────────────────────────────

CUSTOM_OBJECTS = {
    "IngredientAttentionLayer": IngredientAttentionLayer,
    "L2NormalizeLayer":         L2NormalizeLayer,
    "RecommendationLoss":       RecommendationLoss,
}


def main():
    print("\n" + "="*60)
    print("  OLAH - Recipe Recommender Training Pipeline")
    print("  Coding Camp 2026 | CC26-PSU127")
    print("="*60 + "\n")

    # ── Data ──
    preprocessor = DataPreprocessor(CONFIG)
    df           = preprocessor.load_and_prepare(CONFIG["data_path"])

    X = np.array(df["ingredient_seq"].tolist(), dtype=np.int32)
    y = df["category_encoded"].values.astype(np.int32)

    print(f"\n[SPLIT] X shape: {X.shape} | y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_SEED
    )
    print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Model ──
    model = build_olah_model(
        vocab_size     = len(preprocessor.vocab),
        num_categories = CONFIG["num_categories"],
        max_seq_length = CONFIG["max_seq_length"],
        embedding_dim  = CONFIG["embedding_dim"],
        hidden_dim     = CONFIG["hidden_dim"],
        dropout_rate   = CONFIG["dropout_rate"],
    )
    model.summary()

    # ── Training ──
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

    # ── Evaluasi ──
    print("\n[EVAL] Loading best model untuk evaluasi test set...")
    best_model = keras.models.load_model(
        CONFIG["model_save_path"], custom_objects=CUSTOM_OBJECTS
    )

    _, test_probs = best_model(X_test, training=False)
    test_acc_m = keras.metrics.SparseCategoricalAccuracy()
    test_acc_m.update_state(y_test, test_probs)
    test_acc = float(test_acc_m.result())

    print(f"\n{'='*45}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Target        : >= 85.00%")
    print(f"  Status        : {'✓ PASSED' if test_acc >= 0.85 else '→ Perlu lebih banyak data/epoch'}")
    print(f"{'='*45}\n")

    # ── Simpan Embeddings & Metadata ──
    print("[EMBED] Mengekstrak embeddings semua resep...")
    engine       = RecipeEmbeddingEngine(best_model, CONFIG)
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
    print(f"[DONE] Logs   : tensorboard --logdir {CONFIG['tensorboard_log_dir']}")
    print(f"[DONE] Training selesai!")
    return history, test_acc


if __name__ == "__main__":
    history, test_acc = main()