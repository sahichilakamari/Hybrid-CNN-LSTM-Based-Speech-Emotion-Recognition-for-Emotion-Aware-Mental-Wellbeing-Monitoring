import os
import numpy as np
import tensorflow as tf
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from features.mel import extract_logmel
from models.cnn_bilstm import build_next_model
from config import BATCH_SIZE, EPOCHS

DATA_PATH = "data/audio/RAVDESS"

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

# ================= SAFE MFCC =================
def extract_mfcc_fixed(path):
    y, sr = librosa.load(path, sr=22050, duration=3.0)

    target = int(sr * 3.0)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    # ONLY MFCC (stable)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # FIX TIME
    mfcc = librosa.util.fix_length(mfcc, size=160, axis=1)

    # RESIZE ROWS → 128 (deterministic)
    mfcc_resized = np.zeros((128, 160))
    for i in range(128):
        src = int(i * 40 / 128)
        mfcc_resized[i] = mfcc[src]

    # NORMALIZE
    mfcc_resized = (mfcc_resized - np.mean(mfcc_resized)) / (np.std(mfcc_resized) + 1e-6)

    return mfcc_resized


print("Loading data...")
X, y = [], []

# ================= LOAD DATA =================
for actor in os.listdir(DATA_PATH):
    actor_dir = os.path.join(DATA_PATH, actor)

    for file in os.listdir(actor_dir):
        if not file.endswith(".wav"):
            continue

        emotion_id = file.split("-")[2]
        if emotion_id not in EMOTIONS:
            continue

        path = os.path.join(actor_dir, file)

        mel = extract_logmel(path, augment=True)   # (128,160)
        mfcc = extract_mfcc_fixed(path)            # (128,160)

        # 🔥 GUARANTEED SAME SHAPE
        combined = np.stack([mel, mfcc], axis=-1)

        X.append(combined)
        y.append(EMOTIONS[emotion_id])

# ================= PREPROCESS =================
X = np.array(X)

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# ================= CLASS WEIGHTS =================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
class_weights = dict(enumerate(class_weights))

print("Shape:", X.shape)

# ================= SPLIT =================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# ================= MODEL =================
model = build_next_model((128,160,2), len(le.classes_))
model.summary()

# ================= CALLBACKS =================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.ModelCheckpoint("models/best_next_model.keras", save_best_only=True)
]

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================= FINAL =================
val_acc = model.evaluate(X_val, y_val)[1]
print("Final Validation Accuracy:", val_acc)