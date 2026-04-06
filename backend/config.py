import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "final_cnn_lstm_model.h5"  # Your original model
SVM_PATH = "models/ensemble_model.pkl"
SAMPLE_RATE = 22050  # Added this
N_MELS = 128
MAX_LEN = 160
DURATION = 3.0
CONF_THRESHOLD = 0.55
BATCH_SIZE=32
EPOCHS=60

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]

def extract_mel(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Ensure consistent length
    target_samples = sr * DURATION
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=2048, hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0,0),(0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel

def predict_emotion(path):
    mel = extract_mel(path)
    mel = mel[np.newaxis, ..., np.newaxis]

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(mel)[0]
    idx = np.argmax(probs)
    conf = probs[idx]

    if conf < CONF_THRESHOLD:
        return "neutral", conf

    return EMOTIONS[idx], conf