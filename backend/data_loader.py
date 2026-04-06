import os
import numpy as np
from config import DATA_PATH, EMOTIONS
from features.mel import extract_logmel

def load_data():
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

            path = os.path.join(actor_dir, file)

            # Original
            X.append(extract_logmel(path, augment=False))
            y.append(EMOTIONS[emotion_id])

            # Augmented copy
            X.append(extract_logmel(path, augment=True))
            y.append(EMOTIONS[emotion_id])

    return np.array(X), np.array(y)