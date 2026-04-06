# train_transfer_learning_simple.py
import numpy as np
import tensorflow as tf
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib

# ================= CONFIG =================
DATA_PATH = "data/audio/RAVDESS"
SAMPLE_RATE = 22050
N_MELS = 128
MAX_LEN = 160
BATCH_SIZE = 32
EPOCHS = 50

EMOTIONS = {
    "01": "neutral",
    "02": "calm", 
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

# ================= EXTRACT SIMPLE MEL FEATURES =================
def extract_mel_features(audio_path):
    """Extract mel spectrogram features"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=2048, hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    
    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0,0),(0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]
    
    return mel.flatten()  # Flatten for Dense layers

# ================= LOAD DATA =================
print("Loading and extracting features...")
X, y = [], []

for actor in os.listdir(DATA_PATH):
    actor_dir = os.path.join(DATA_PATH, actor)
    if not os.path.isdir(actor_dir):
        continue
    
    for file in os.listdir(actor_dir):
        if not file.endswith(".wav"):
            continue
        
        emotion_id = file.split("-")[2]
        if emotion_id not in EMOTIONS:
            continue
        
        features = extract_mel_features(os.path.join(actor_dir, file))
        X.append(features)
        y.append(EMOTIONS[emotion_id])

X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"Feature shape: {X.shape}")
print(f"Classes: {le.classes_}")

# ================= SPLIT AND TRAIN =================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Build simple Dense model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MELS * MAX_LEN,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Evaluate
val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\nSimple Dense Model Validation Accuracy: {val_acc:.4f}")

# Save
model.save('models/simple_transfer_model.h5')
joblib.dump(le, 'models/simple_transfer_le.pkl')
print("✅ Simple model saved")