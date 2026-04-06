
# ensemble_inference.py
import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model

class SEREnsemble:
    def __init__(self, ensemble_info_path='models/ensemble_info.pkl'):
        # Load ensemble info
        info = joblib.load(ensemble_info_path)
        self.models = [load_model(path) for path in info['model_paths']]
        self.le = info['label_encoder']
        self.input_shape = info['input_shape']
        
    def extract_features(self, audio_path, sr=22050):
        """Extract mel spectrogram features"""
        y, sr = librosa.load(audio_path, sr=sr)
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128,
            n_fft=2048, hop_length=512
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure consistent shape
        if mel.shape[1] < 160:
            mel = np.pad(mel, ((0,0),(0, 160 - mel.shape[1])))
        else:
            mel = mel[:, :160]
        
        return mel[..., np.newaxis]  # Add channel dimension
    
    def predict(self, audio_path):
        """Predict emotion from audio file"""
        # Extract features
        features = self.extract_features(audio_path)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(features, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Get results
        emotion_id = np.argmax(avg_pred)
        confidence = np.max(avg_pred)
        emotion = self.le.inverse_transform([emotion_id])[0]
        
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': avg_pred[0].tolist(),
            'all_emotions': self.le.classes_.tolist()
        }

# Usage example
if __name__ == "__main__":
    ensemble = SEREnsemble()
    result = ensemble.predict("path_to_audio.wav")
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
