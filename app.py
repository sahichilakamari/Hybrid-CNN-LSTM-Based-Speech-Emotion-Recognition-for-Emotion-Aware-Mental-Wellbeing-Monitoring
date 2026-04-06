# app.py (Updated with proper Meta-Ensemble integration)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from features.mel import extract_logmel
from features.mfcc import extract_mfcc
from config import MODEL_PATH, SVM_PATH, N_MELS, MAX_LEN, SAMPLE_RATE
from utils.emotions import get_recommendation, EMOTION_LABELS, EMOTION_DISPLAY_NAMES, EMOTION_COLORS
import librosa
import os
import sys

# Add the current directory to path to import meta_ensemble
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load emotion labels from utils
print(f"Using emotion labels: {EMOTION_LABELS}")
print(f"Total emotions: {len(EMOTION_LABELS)}")

# ================= LOAD ALL MODELS =================
models_status = {}

# 1. Load original deep learning model
try:
    dl_model = load_model(MODEL_PATH, compile=False)
    models_status['original_dl'] = True
    print("✅ Original deep learning model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load original deep learning model: {e}")
    dl_model = None
    models_status['original_dl'] = False

# 2. Load K-Fold Ensemble models
try:
    ensemble_info = joblib.load('models/ensemble_info.pkl')
    ensemble_models = [load_model(path, compile=False) for path in ensemble_info['model_paths']]
    ensemble_le = ensemble_info['label_encoder']
    models_status['kfold_ensemble'] = True
    print(f"✅ K-Fold Ensemble loaded successfully ({len(ensemble_models)} models)")
except Exception as e:
    print(f"❌ Failed to load K-Fold Ensemble: {e}")
    ensemble_models = None
    ensemble_info = None
    models_status['kfold_ensemble'] = False

# 3. Load Transfer Learning model
try:
    transfer_model = load_model('models/simple_transfer_model.h5', compile=False)
    transfer_le = joblib.load('models/simple_transfer_le.pkl')
    models_status['transfer_learning'] = True
    print("✅ Transfer Learning model loaded successfully")
except Exception as e:
    print(f"⚠️  Transfer Learning model not found: {e}")
    transfer_model = None
    transfer_le = None
    models_status['transfer_learning'] = False

# 4. Load ML fallback models
try:
    svm = joblib.load(SVM_PATH)
    models_status['svm'] = True
    print("✅ SVM model loaded successfully")
except:
    print("❌ SVM model not found")
    svm = None
    models_status['svm'] = False

# 5. Load enhanced ML model
try:
    ml_model = joblib.load("models/ml_emotion_model.pkl")
    ml_scaler = joblib.load("models/ml_scaler.pkl")
    models_status['enhanced_ml'] = True
    print("✅ Enhanced ML model loaded successfully")
except:
    print("❌ Enhanced ML model not found")
    ml_model = None
    ml_scaler = None
    models_status['enhanced_ml'] = False

# 6. Initialize WorkingEnsemble for meta-ensemble
try:
    from meta_ensemble import WorkingEnsemble
    working_ensemble = WorkingEnsemble()
    models_status['meta_ensemble'] = True
    print("✅ WorkingEnsemble initialized successfully")
except Exception as e:
    print(f"⚠️  Failed to initialize WorkingEnsemble: {e}")
    working_ensemble = None
    models_status['meta_ensemble'] = False

# ================= HELPER FUNCTIONS =================
def extract_features_for_ml(audio_path):
    """
    Extract features for ML model (same as training)
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=3.0)
        
        # Ensure consistent length
        target_samples = sr * 3
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            y = y[:target_samples]
        
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            coeff = mfcc[i]
            features.extend([
                np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff),
                np.median(coeff), np.percentile(coeff, 25), np.percentile(coeff, 75)
            ])
        
        # MFCC deltas
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        for feature_matrix in [mfcc_delta, mfcc_delta2]:
            for i in range(13):
                coeff = feature_matrix[i]
                features.extend([np.mean(coeff), np.std(coeff), np.max(coeff)])
        
        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel)
        features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db), np.min(mel_db)])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean.tolist())
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ])
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        features.extend(contrast_mean.tolist())
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        features.extend(tonnetz_mean.tolist())
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting ML features: {e}")
        return None

def ensemble_predict_cnn(audio_path):
    """Make prediction using K-Fold Ensemble"""
    if ensemble_models is None:
        return None
    
    try:
        # Extract features
        mel = extract_logmel(audio_path, target_length=MAX_LEN)
        mel = mel[np.newaxis, ..., np.newaxis]
        
        # Get predictions from all models
        predictions = []
        for model in ensemble_models:
            pred = model.predict(mel, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Get results
        emotion_id = np.argmax(avg_pred)
        confidence = np.max(avg_pred)
        
        # Map to our emotion labels (assuming ensemble uses same mapping)
        # If not, we need to map from ensemble_le to EMOTION_LABELS
        if ensemble_le is not None:
            emotion_label = ensemble_le.inverse_transform([emotion_id])[0]
            # Convert to our label format if needed
            if emotion_label not in EMOTION_LABELS:
                # Simple mapping (adjust based on your actual labels)
                label_map = {
                    'neutral': 'neutral', 'calm': 'neutral', 'happy': 'happy',
                    'sad': 'sad', 'angry': 'angry', 'fearful': 'fearful'
                }
                emotion_label = label_map.get(emotion_label, 'neutral')
        else:
            # Fallback to direct index
            emotion_label = EMOTION_LABELS[emotion_id % len(EMOTION_LABELS)]
        
        return {
            'emotion_id': int(emotion_id),
            'emotion_label': emotion_label,
            'confidence': float(confidence),
            'ensemble_size': len(ensemble_models),
            'predictions': [p[0].tolist() for p in predictions]
        }
    except Exception as e:
        print(f"Ensemble prediction error: {e}")
        return None

def meta_ensemble_predict(audio_path):
    """Meta-ensemble combining multiple model types"""
    all_predictions = []
    weights = {'cnn_ensemble': 0.4, 'transfer': 0.3, 'ml': 0.3}
    
    # 1. CNN Ensemble prediction
    if models_status['kfold_ensemble']:
        cnn_result = ensemble_predict_cnn(audio_path)
        if cnn_result:
            # Create probability vector aligned with EMOTION_LABELS
            cnn_probs = np.zeros(len(EMOTION_LABELS))
            emotion_idx = EMOTION_LABELS.index(cnn_result['emotion_label']) if cnn_result['emotion_label'] in EMOTION_LABELS else 0
            cnn_probs[emotion_idx] = cnn_result['confidence']
            all_predictions.append(('cnn_ensemble', cnn_probs, weights['cnn_ensemble']))
    
    # 2. Transfer Learning prediction (if available)
    if models_status['transfer_learning'] and transfer_model is not None and transfer_le is not None:
        try:
            # Extract features for transfer model (flattened mel spectrogram)
            y, sr = librosa.load(audio_path, sr=16000)
            if len(y) < sr * 3:
                y = np.pad(y, (0, sr * 3 - len(y)))
            else:
                y = y[:sr * 3]
            
            # Extract mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128,
                n_fft=2048, hop_length=512
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            
            if mel.shape[1] < 160:
                mel = np.pad(mel, ((0,0),(0, 160 - mel.shape[1])))
            else:
                mel = mel[:, :160]
            
            # Flatten for transfer model
            features = mel.flatten()
            features = np.expand_dims(features, axis=0)
            
            # Predict
            pred = transfer_model.predict(features, verbose=0)[0]
            
            # Create probability vector for our 5 emotions
            prob_vector = np.zeros(len(EMOTION_LABELS))
            emotion_id = np.argmax(pred)
            
            # Get emotion from label encoder
            try:
                emotion = transfer_le.inverse_transform([emotion_id])[0]
                # Map to our emotion labels
                if emotion == 'calm':
                    emotion = 'neutral'
            except:
                emotion = EMOTION_LABELS[emotion_id % len(EMOTION_LABELS)]
            
            # Map probabilities
            for i in range(len(pred)):
                try:
                    pred_emotion = transfer_le.inverse_transform([i])[0]
                    if pred_emotion == 'calm':
                        pred_emotion = 'neutral'
                    if pred_emotion in EMOTION_LABELS:
                        idx = EMOTION_LABELS.index(pred_emotion)
                        prob_vector[idx] += pred[i]
                except:
                    continue
            
            # Normalize
            if prob_vector.sum() > 0:
                prob_vector = prob_vector / prob_vector.sum()
            
            all_predictions.append(('transfer', prob_vector, weights['transfer']))
            
        except Exception as e:
            print(f"Transfer learning error: {e}")
    
    # 3. ML model prediction
    if models_status['enhanced_ml'] and ml_model is not None and ml_scaler is not None:
        try:
            features = extract_features_for_ml(audio_path)
            if features is not None:
                features_scaled = ml_scaler.transform(features.reshape(1, -1))
                ml_pred = ml_model.predict_proba(features_scaled)[0]
                all_predictions.append(('ml', ml_pred, weights['ml']))
        except Exception as e:
            print(f"ML model error: {e}")
    
    # 4. Original DL model (fallback)
    if models_status['original_dl'] and dl_model is not None and not all_predictions:
        try:
            x = extract_logmel(audio_path, target_length=MAX_LEN)
            x = x[np.newaxis, ..., np.newaxis]
            pred = dl_model.predict(x, verbose=0)
            all_predictions.append(('original_dl', pred[0], 1.0))
        except Exception as e:
            print(f"Original DL model error: {e}")
    
    # Combine predictions
    if not all_predictions:
        return None
    
    # Weighted average
    total_weight = sum(w for _, _, w in all_predictions)
    final_pred = np.zeros(len(EMOTION_LABELS))
    
    for name, pred, weight in all_predictions:
        # Ensure same length
        if len(pred) != len(final_pred):
            if len(pred) > len(final_pred):
                pred = pred[:len(final_pred)]
            else:
                # Pad with zeros
                padded = np.zeros(len(final_pred))
                padded[:len(pred)] = pred
                pred = padded
        
        final_pred += pred * (weight / total_weight)
    
    # Get final result
    emotion_id = np.argmax(final_pred)
    confidence = np.max(final_pred)
    
    return {
        'emotion_id': int(emotion_id),
        'emotion_label': EMOTION_LABELS[emotion_id],
        'confidence': float(confidence),
        'probabilities': final_pred.tolist(),
        'models_used': [name for name, _, _ in all_predictions],
        'model_count': len(all_predictions)
    }

# ================= ROUTES =================
@app.route("/predict", methods=["POST"])
def predict():
    """Original prediction endpoint (backward compatible)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    path = "temp.wav"
    file.save(path)
    
    result = {}
    
    # Try deep learning model first
    if dl_model is not None:
        try:
            x = extract_logmel(path, target_length=MAX_LEN)
            x = x[np.newaxis, ..., np.newaxis]
            pred = dl_model.predict(x, verbose=0)
            emotion_id = int(np.argmax(pred))
            confidence = float(np.max(pred))
            
            # Ensure emotion_id is within bounds
            if emotion_id >= len(EMOTION_LABELS):
                emotion_id = min(emotion_id, len(EMOTION_LABELS) - 1)
                
            result["model_type"] = "deep_learning"
            result["confidence"] = confidence
            result["emotion_id"] = emotion_id
            result["emotion_label"] = EMOTION_LABELS[emotion_id]
            result["emotion_display"] = EMOTION_DISPLAY_NAMES.get(emotion_id, "Unknown")
            result["color"] = EMOTION_COLORS.get(emotion_id, '#6c757d')
            
        except Exception as e:
            print(f"DL model error: {e}")
    
    # If DL failed or not available, try enhanced ML model
    if 'model_type' not in result and ml_model is not None and ml_scaler is not None:
        try:
            features = extract_features_for_ml(path)
            if features is not None:
                features_scaled = ml_scaler.transform(features.reshape(1, -1))
                pred = ml_model.predict_proba(features_scaled)[0]
                emotion_id = int(np.argmax(pred))
                confidence = float(np.max(pred))
                
                # Ensure emotion_id is within bounds
                if emotion_id >= len(EMOTION_LABELS):
                    emotion_id = min(emotion_id, len(EMOTION_LABELS) - 1)
                
                result["model_type"] = "ml_enhanced"
                result["confidence"] = confidence
                result["emotion_id"] = emotion_id
                result["emotion_label"] = EMOTION_LABELS[emotion_id]
                result["emotion_display"] = EMOTION_DISPLAY_NAMES.get(emotion_id, "Unknown")
                result["color"] = EMOTION_COLORS.get(emotion_id, '#6c757d')
                result["fallback_info"] = "Used enhanced ML model"
        
        except Exception as e:
            print(f"Enhanced ML model error: {e}")
    
    # If still no result, try basic SVM
    if 'model_type' not in result and svm is not None:
        try:
            x = extract_mfcc(path).mean(axis=1).reshape(1, -1)
            pred = svm.predict_proba(x)[0]
            emotion_id = int(np.argmax(pred))
            confidence = float(np.max(pred))
            
            # Ensure emotion_id is within bounds
            if emotion_id >= len(EMOTION_LABELS):
                emotion_id = min(emotion_id, len(EMOTION_LABELS) - 1)
            
            result["model_type"] = "svm_basic"
            result["confidence"] = confidence
            result["emotion_id"] = emotion_id
            result["emotion_label"] = EMOTION_LABELS[emotion_id]
            result["emotion_display"] = EMOTION_DISPLAY_NAMES.get(emotion_id, "Unknown")
            result["color"] = EMOTION_COLORS.get(emotion_id, '#6c757d')
            result["fallback_info"] = "Used basic SVM fallback"
            
        except Exception as e:
            print(f"SVM model error: {e}")
    
    # If all models failed
    if 'model_type' not in result:
        # Return a default neutral response
        result = {
            "model_type": "fallback",
            "confidence": 0.5,
            "emotion_id": 0,
            "emotion_label": "neutral",
            "emotion_display": "Neutral",
            "color": "#6c757d",
            "message": "Using fallback neutral emotion"
        }
    
    # Get recommendations
    recommendations = get_recommendation(result["emotion_id"], result["confidence"])
    result.update(recommendations)
    
    # Clean up temp file
    if os.path.exists(path):
        os.remove(path)
    
    return jsonify(result)

@app.route("/predict_ensemble", methods=["POST"])
def predict_ensemble():
    """New endpoint for K-Fold Ensemble prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    path = "temp.wav"
    file.save(path)
    
    if ensemble_models is None:
        return jsonify({
            "error": "Ensemble models not loaded",
            "available_models": models_status
        }), 503
    
    result = ensemble_predict_cnn(path)
    
    if result is None:
        return jsonify({
            "error": "Ensemble prediction failed",
            "model_type": "ensemble_failed"
        }), 500
    
    # Add additional information
    result["model_type"] = "kfold_ensemble"
    result["emotion_display"] = EMOTION_DISPLAY_NAMES.get(
        EMOTION_LABELS.index(result["emotion_label"]) if result["emotion_label"] in EMOTION_LABELS else 0,
        result["emotion_label"]
    )
    result["color"] = EMOTION_COLORS.get(
        EMOTION_LABELS.index(result["emotion_label"]) if result["emotion_label"] in EMOTION_LABELS else 0,
        '#6c757d'
    )
    
    # Get recommendations
    emotion_id = EMOTION_LABELS.index(result["emotion_label"]) if result["emotion_label"] in EMOTION_LABELS else 0
    recommendations = get_recommendation(emotion_id, result["confidence"])
    result.update(recommendations)
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)
    
    return jsonify(result)

@app.route("/predict_meta", methods=["POST"])
def predict_meta():
    """Meta-ensemble endpoint using WorkingEnsemble class"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    path = "temp.wav"
    file.save(path)
    
    # Use the WorkingEnsemble if available
    if working_ensemble is not None:
        try:
            result = working_ensemble.predict(path)
            
            # Add additional information
            result["model_type"] = "meta_ensemble"
            
            # Ensure emotion is in our standard labels
            if result['emotion'] not in EMOTION_LABELS:
                # Map to our standard labels
                emotion_map = {
                    'neutral': 'neutral',
                    'happy': 'happy',
                    'sad': 'sad',
                    'angry': 'angry',
                    'fearful': 'fearful'
                }
                result['emotion'] = emotion_map.get(result['emotion'], 'neutral')
            
            # Get emotion ID
            if result['emotion'] in EMOTION_LABELS:
                emotion_id = EMOTION_LABELS.index(result['emotion'])
            else:
                emotion_id = 0
            
            # Add recommendations
            recommendations = get_recommendation(emotion_id, result["confidence"])
            result.update(recommendations)
            
            # Clean up
            if os.path.exists(path):
                os.remove(path)
            
            return jsonify(result)
            
        except Exception as e:
            print(f"WorkingEnsemble error: {e}")
    
    # Fallback to the manual meta-ensemble
    print("WorkingEnsemble failed, falling back to manual meta-ensemble")
    result = meta_ensemble_predict(path)
    
    if result is None:
        # Fallback to original predict
        print("Manual meta-ensemble failed, falling back to original predict")
        return predict()
    
    # Add display information
    result["model_type"] = "meta_ensemble"
    result["emotion_display"] = EMOTION_DISPLAY_NAMES.get(result["emotion_id"], "Unknown")
    result["color"] = EMOTION_COLORS.get(result["emotion_id"], '#6c757d')
    
    # Get recommendations
    recommendations = get_recommendation(result["emotion_id"], result["confidence"])
    result.update(recommendations)
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)
    
    return jsonify(result)

@app.route("/predict_with_fallback", methods=["POST"])
def predict_with_fallback():
    """Endpoint with fallback to mock data"""
    try:
        return predict()
    except Exception as e:
        print(f"Predict failed, returning mock data: {e}")
        # Return mock data
        return jsonify({
            "model_type": "mock",
            "confidence": 0.75,
            "emotion_id": 0,
            "emotion_label": "neutral",
            "emotion_display": "Neutral",
            "color": "#6c757d",
            "recommendation": "This is a mock response. Check backend connection.",
            "suggestions": ["Ensure backend is running", "Check audio file format"]
        })

@app.route("/emotions", methods=["GET"])
def get_emotions():
    """Return available emotions"""
    emotions_list = []
    for i, label in enumerate(EMOTION_LABELS):
        emotions_list.append({
            "id": i,
            "label": label,
            "display_name": EMOTION_DISPLAY_NAMES.get(i, label),
            "color": EMOTION_COLORS.get(i, '#6c757d'),
            "severity": 0.1 * (i + 1)  # Simple severity score
        })
    
    return jsonify({
        "emotions": emotions_list,
        "count": len(EMOTION_LABELS),
        "message": f"Available emotions: {', '.join(EMOTION_LABELS)}"
    })

@app.route("/models", methods=["GET"])
def get_models():
    """Return available models and their status"""
    return jsonify({
        "models": models_status,
        "recommended_endpoint": "/predict_meta for best accuracy",
        "endpoints": {
            "/predict": "Original model (backward compatible)",
            "/predict_ensemble": "K-Fold Ensemble (better accuracy)",
            "/predict_meta": "Meta-Ensemble (best accuracy)"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "models": models_status,
        "num_emotions": len(EMOTION_LABELS),
        "emotions": EMOTION_LABELS,
        "recommendation": "Use /predict_meta endpoint for best results"
    }
    return jsonify(status)

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    print("=" * 60)
    print("SER-Mind Backend Server (Enhanced)")
    print(f"Emotion Classes: {EMOTION_LABELS}")
    print(f"Available Models: {models_status}")
    print("\nAvailable Endpoints:")
    print("  /predict           - Original model")
    print("  /predict_ensemble  - K-Fold Ensemble (recommended)")
    print("  /predict_meta      - Meta-Ensemble (best accuracy)")
    print("  /models            - List available models")
    print("  /health            - Health check")
    print("\nServer running on: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)