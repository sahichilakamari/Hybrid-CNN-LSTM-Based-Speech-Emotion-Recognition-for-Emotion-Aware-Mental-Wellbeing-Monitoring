# utils/audio.py
import librosa
import numpy as np
import soundfile as sf
from config import SAMPLE_RATE, MAX_LEN, DURATION
import noisereduce as nr
import warnings
warnings.filterwarnings('ignore')

def load_audio(path, sr=SAMPLE_RATE, duration=DURATION, mono=True):
    """
    Load audio file with consistent parameters
    """
    try:
        y, sr_loaded = librosa.load(
            path, 
            sr=sr, 
            duration=duration, 
            mono=mono
        )
        
        # Ensure correct length
        target_length = sr * duration
        if len(y) < target_length:
            # Pad with zeros
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        elif len(y) > target_length:
            # Trim to target length
            y = y[:target_length]
            
        return y
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # Return silent audio of correct length
        return np.zeros(sr * duration)

def preprocess_audio(y, sr=SAMPLE_RATE):
    """
    Preprocess audio: remove noise, normalize, etc.
    """
    # Remove background noise
    y_clean = nr.reduce_noise(y=y, sr=sr)
    
    # Normalize amplitude
    y_norm = librosa.util.normalize(y_clean)
    
    return y_norm

def audio_augmentation(y, sr=SAMPLE_RATE):
    """
    Apply data augmentation to audio
    Returns a list of augmented versions
    """
    augmented = [y]  # Start with original
    
    # Add noise
    noise = np.random.randn(len(y))
    y_noisy = y + NOISE_LEVEL * noise
    augmented.append(y_noisy)
    
    # Pitch shift
    n_steps = np.random.uniform(-PITCH_SHIFT_RANGE, PITCH_SHIFT_RANGE)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    augmented.append(y_pitch)
    
    # Time stretch
    rate = np.random.uniform(*TIME_STRETCH_RANGE)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    # Ensure same length
    if len(y_stretch) > len(y):
        y_stretch = y_stretch[:len(y)]
    else:
        y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)))
    augmented.append(y_stretch)
    
    # Random shift (time offset)
    shift = np.random.randint(0, len(y) // 4)
    y_shift = np.roll(y, shift)
    augmented.append(y_shift)
    
    return augmented

def extract_all_features(y, sr=SAMPLE_RATE):
    """
    Extract all possible audio features for comprehensive analysis
    Returns a dictionary of features
    """
    features = {}
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_EXTENDED)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features['mfcc'] = mfcc
    features['mfcc_delta'] = delta
    features['mfcc_delta2'] = delta2
    
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    features['mel'] = mel
    features['logmel'] = librosa.power_to_db(mel)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['chroma'] = chroma
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    features['spectral_centroid'] = spectral_centroid
    features['spectral_bandwidth'] = spectral_bandwidth
    features['spectral_rolloff'] = spectral_rolloff
    features['spectral_contrast'] = spectral_contrast
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features['tonnetz'] = tonnetz
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features['zcr'] = zcr
    
    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features['rms'] = rms
    
    # Tempo and beat
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['tempo'] = tempo
    
    return features

def get_statistical_features(feature_matrix):
    """
    Extract statistical features from feature matrix
    Useful for ML models
    """
    stats = []
    
    # For each feature dimension
    for i in range(feature_matrix.shape[0]):
        feature_vector = feature_matrix[i, :]
        stats.extend([
            np.mean(feature_vector),     # Mean
            np.std(feature_vector),      # Standard deviation
            np.median(feature_vector),   # Median
            np.min(feature_vector),      # Minimum
            np.max(feature_vector),      # Maximum
            np.percentile(feature_vector, 25),  # 25th percentile
            np.percentile(feature_vector, 75),  # 75th percentile
            np.max(feature_vector) - np.min(feature_vector),  # Range
            np.mean(np.abs(feature_vector - np.mean(feature_vector))),  # Mean absolute deviation
            np.std(feature_vector) / (np.mean(feature_vector) + 1e-10)  # Coefficient of variation
        ])
    
    return np.array(stats)