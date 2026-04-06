import librosa
import numpy as np
from config import SAMPLE_RATE, N_MELS, MAX_LEN

# ================= SPEC AUGMENT =================
def spec_augment(mel, freq_mask=15, time_mask=20):
    mel = mel.copy()

    # Frequency mask
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, mel.shape[0] - f)
    mel[f0:f0+f, :] = 0

    # Time mask
    t = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, mel.shape[1] - t)
    mel[:, t0:t0+t] = 0

    return mel


def extract_logmel(path, augment=False):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=3.0)

    target = int(sr * 3.0)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel)

    # pad/trim
    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    # normalize
    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

    if augment:
        mel = spec_augment(mel)

    return mel