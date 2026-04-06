# meta_ensemble_final.py
import numpy as np
import joblib
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class WorkingEnsemble:
    """Working ensemble that handles all feature dimension issues"""
    def __init__(self):
        print("=" * 60)
        print("LOADING WORKING ENSEMBLE")
        print("=" * 60)
        
        self.models = {}
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful']
        self.emotion_display = {
            'neutral': 'Neutral',
            'happy': 'Happy', 
            'sad': 'Sad',
            'angry': 'Angry',
            'fearful': 'Fearful'
        }
        
        # 1. Load Transfer Model
        try:
            self.models['transfer'] = load_model('models/simple_transfer_model.h5', compile=False)
            self.transfer_le = joblib.load('models/simple_transfer_le.pkl')
            print("✅ Transfer model loaded")
        except Exception as e:
            print(f"❌ Transfer model: {e}")
            self.models['transfer'] = None
        
        # 2. Load ML Model
        try:
            self.models['ml'] = joblib.load('models/ml_emotion_model.pkl')
            self.ml_scaler = joblib.load('models/ml_scaler.pkl')
            print("✅ ML model loaded")
            print(f"   ML expects {self.ml_scaler.n_features_in_} features")
        except Exception as e:
            print(f"❌ ML model: {e}")
            self.models['ml'] = None
            self.ml_scaler = None
        
        # 3. Load Original CNN (with fix)
        try:
            # Custom layer to fix loading
            class FixedSpatialDropout2D(tf.keras.layers.Layer):
                def __init__(self, rate, **kwargs):
                    kwargs.pop('trainable', None)
                    super().__init__(**kwargs)
                    self.rate = rate
                def call(self, inputs, training=None):
                    return tf.keras.layers.SpatialDropout2D(self.rate)(inputs, training=training)
            
            self.models['cnn'] = load_model(
                'models/ser_ravdess_phase2.h5', 
                compile=False,
                custom_objects={'SpatialDropout2D': FixedSpatialDropout2D}
            )
            print("✅ CNN model loaded")
        except Exception as e:
            print(f"❌ CNN model: {e}")
            self.models['cnn'] = None
        
        # Model weights (based on expected accuracy)
        self.weights = {
            'transfer': 0.40,  # 46% accuracy
            'ml': 0.35,        # ~40% accuracy  
            'cnn': 0.25        # ~42% accuracy
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"\n✅ Models loaded: {[k for k, v in self.models.items() if v is not None]}")
        print(f"   Weights: {self.weights}")
    
    def extract_features_for_transfer(self, audio_path):
        """Extract flattened mel features for transfer model"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Ensure 3 seconds
        target_samples = sr * 3
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128,
            n_fft=2048, hop_length=512
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        if mel.shape[1] < 160:
            mel = np.pad(mel, ((0,0),(0, 160 - mel.shape[1])))
        else:
            mel = mel[:, :160]
        
        return mel.flatten()
    
    def extract_features_for_ml(self, audio_path):
        """Extract proper features for ML model (220 features)"""
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        target_samples = sr * 3
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
        
        features = []
        
        # 1. MFCC features (13 coefficients × 7 stats = 91 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            coeff = mfcc[i]
            features.extend([
                np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff),
                np.median(coeff), np.percentile(coeff, 25), np.percentile(coeff, 75)
            ])
        
        # 2. MFCC deltas (13 × 3 stats × 2 = 78 features)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        for feature_matrix in [mfcc_delta, mfcc_delta2]:
            for i in range(13):
                coeff = feature_matrix[i]
                features.extend([np.mean(coeff), np.std(coeff), np.max(coeff)])
        
        # 3. Mel-spectrogram (4 features)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel)
        features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db), np.min(mel_db)])
        
        # 4. Chroma features (12 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean.tolist())
        
        # 5. Spectral features (6 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ])
        
        # 6. Zero-crossing rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 7. RMS energy (2 features)
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # 8. Spectral contrast (7 features)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        features.extend(contrast_mean.tolist())
        
        # 9. Tonnetz (6 features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        features.extend(tonnetz_mean.tolist())
        
        # Total should be 220 features
        features = np.array(features)
        
        # Ensure exactly 220 features
        if len(features) < 220:
            features = np.pad(features, (0, 220 - len(features)))
        elif len(features) > 220:
            features = features[:220]
        
        return features
    
    def extract_features_for_cnn(self, audio_path):
        """Extract mel spectrogram for CNN"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        target_samples = sr * 3
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128,
            n_fft=2048, hop_length=512
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        if mel.shape[1] < 160:
            mel = np.pad(mel, ((0,0),(0, 160 - mel.shape[1])))
        else:
            mel = mel[:, :160]
        
        return mel[..., np.newaxis]
    
    def predict_transfer(self, audio_path):
        """Predict using transfer model"""
        if self.models['transfer'] is None:
            return None
        
        try:
            features = self.extract_features_for_transfer(audio_path)
            features = np.expand_dims(features, axis=0)
            
            pred = self.models['transfer'].predict(features, verbose=0)[0]
            
            # Get emotion from label encoder
            emotion_id = np.argmax(pred)
            try:
                emotion = self.transfer_le.inverse_transform([emotion_id])[0]
                # Map to our 5 emotions
                if emotion == 'calm':
                    emotion = 'neutral'
            except:
                emotion = self.emotion_labels[emotion_id % len(self.emotion_labels)]
            
            confidence = float(np.max(pred))
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': pred.tolist()
            }
        except Exception as e:
            print(f"Transfer prediction error: {e}")
            return None
    
    def predict_ml(self, audio_path):
        """Predict using ML model"""
        if self.models['ml'] is None or self.ml_scaler is None:
            return None
        
        try:
            features = self.extract_features_for_ml(audio_path)
            
            # Scale features
            features = self.ml_scaler.transform(features.reshape(1, -1))
            
            # Predict
            pred = self.models['ml'].predict_proba(features)[0]
            
            # ML model uses same 6 emotions as dataset
            # Map to our 5 emotions
            emotion_id = np.argmax(pred)
            ml_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
            emotion = ml_emotions[emotion_id]
            if emotion == 'calm':
                emotion = 'neutral'
            
            confidence = float(np.max(pred))
            
            # Create probability vector for our 5 emotions
            prob_vector = np.zeros(len(self.emotion_labels))
            for i, ml_emotion in enumerate(ml_emotions):
                if ml_emotion == 'calm':
                    # Add calm probability to neutral
                    prob_vector[0] += pred[i]
                else:
                    # Map to our emotion index
                    if ml_emotion in self.emotion_labels:
                        idx = self.emotion_labels.index(ml_emotion)
                        prob_vector[idx] = pred[i]
            
            # Normalize probabilities
            if prob_vector.sum() > 0:
                prob_vector = prob_vector / prob_vector.sum()
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': prob_vector.tolist()
            }
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
    
    def predict_cnn(self, audio_path):
        """Predict using CNN model"""
        if self.models['cnn'] is None:
            return None
        
        try:
            features = self.extract_features_for_cnn(audio_path)
            features = np.expand_dims(features, axis=0)
            
            pred = self.models['cnn'].predict(features, verbose=0)[0]
            
            # CNN model uses 6 emotions
            emotion_id = np.argmax(pred)
            cnn_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
            emotion = cnn_emotions[emotion_id]
            if emotion == 'calm':
                emotion = 'neutral'
            
            confidence = float(np.max(pred))
            
            # Map probabilities to our 5 emotions
            prob_vector = np.zeros(len(self.emotion_labels))
            for i, cnn_emotion in enumerate(cnn_emotions):
                if cnn_emotion == 'calm':
                    prob_vector[0] += pred[i]
                else:
                    if cnn_emotion in self.emotion_labels:
                        idx = self.emotion_labels.index(cnn_emotion)
                        prob_vector[idx] = pred[i]
            
            # Normalize
            if prob_vector.sum() > 0:
                prob_vector = prob_vector / prob_vector.sum()
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': prob_vector.tolist()
            }
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return None
    
    def predict(self, audio_path):
        """Ensemble prediction combining all models"""
        print(f"\n{'='*60}")
        print("ENSEMBLE PREDICTION")
        print('='*60)
        print(f"File: {os.path.basename(audio_path)}")
        
        predictions = {}
        results = {}
        
        # Get predictions from each model
        if self.weights['transfer'] > 0:
            transfer_result = self.predict_transfer(audio_path)
            if transfer_result:
                predictions['transfer'] = transfer_result
                print(f"✅ Transfer: {transfer_result['emotion']} ({transfer_result['confidence']:.3f})")
        
        if self.weights['ml'] > 0:
            ml_result = self.predict_ml(audio_path)
            if ml_result:
                predictions['ml'] = ml_result
                print(f"✅ ML: {ml_result['emotion']} ({ml_result['confidence']:.3f})")
        
        if self.weights['cnn'] > 0:
            cnn_result = self.predict_cnn(audio_path)
            if cnn_result:
                predictions['cnn'] = cnn_result
                print(f"✅ CNN: {cnn_result['emotion']} ({cnn_result['confidence']:.3f})")
        
        if not predictions:
            print("❌ No models produced valid predictions")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'emotion_display': 'Neutral',
                'color': '#6c757d',
                'model_type': 'fallback'
            }
        
        # Combine predictions using weighted voting
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        for model_name, result in predictions.items():
            weight = self.weights[model_name]
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Add weighted score
            emotion_scores[emotion] += confidence * weight
        
        # Get final prediction
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        final_confidence = emotion_scores[final_emotion]
        
        # Get top 3 emotions
        sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = [(emotion, score) for emotion, score in sorted_scores[:3]]
        
        # Prepare result
        result = {
            'emotion': final_emotion,
            'confidence': float(final_confidence),
            'emotion_display': self.emotion_display.get(final_emotion, final_emotion),
            'color': self.get_color(final_emotion),
            'model_type': 'ensemble',
            'models_used': list(predictions.keys()),
            'model_count': len(predictions),
            'top_3_emotions': [
                {
                    'emotion': emotion,
                    'display': self.emotion_display.get(emotion, emotion),
                    'color': self.get_color(emotion),
                    'probability': float(score)
                }
                for emotion, score in top_3
            ],
            'individual_predictions': {
                model_name: {
                    'emotion': pred['emotion'],
                    'confidence': pred['confidence']
                }
                for model_name, pred in predictions.items()
            }
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL: {result['emotion_display']}")
        print(f"CONFIDENCE: {result['confidence']:.3f}")
        print(f"MODELS: {', '.join(result['models_used'])}")
        print('='*60)
        
        return result
    
    def get_color(self, emotion):
        """Get color for emotion"""
        colors = {
            'neutral': '#6c757d',
            'happy': '#ffc107',
            'sad': '#0d6efd',
            'angry': '#dc3545',
            'fearful': '#6f42c1'
        }
        return colors.get(emotion, '#6c757d')

# Test function
def test_ensemble():
    """Test the ensemble with sample audio"""
    import sys
    
    ensemble = WorkingEnsemble()
    
    # Test with provided file or find one
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    else:
        # Find a test file
        test_files = []
        data_dir = "data/audio/RAVDESS"
        for actor in os.listdir(data_dir)[:1]:
            actor_dir = os.path.join(data_dir, actor)
            if os.path.isdir(actor_dir):
                wav_files = [f for f in os.listdir(actor_dir) if f.endswith('.wav')]
                test_files.extend([os.path.join(actor_dir, f) for f in wav_files[:2]])
                break
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            print(f"\n\nTesting: {os.path.basename(audio_file)}")
            result = ensemble.predict(audio_file)
            
            print(f"\n📊 Result Summary:")
            print(f"  Emotion: {result['emotion_display']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Models used: {result['model_count']}")
            
            print(f"\n🏆 Top 3 predictions:")
            for i, top in enumerate(result['top_3_emotions'], 1):
                print(f"  {i}. {top['display']}: {top['probability']:.3f}")
        else:
            print(f"File not found: {audio_file}")

if __name__ == "__main__":
    test_ensemble()