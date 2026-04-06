# train_ml.py
import os
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import librosa
from utils.emotions import EMOTION_MAP, EMOTION_LABELS
import warnings
warnings.filterwarnings('ignore')

def extract_enhanced_features(audio_path):
    """
    Extract comprehensive audio features for ML models
    """
    try:
        # Load audio with more robust error handling
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0, mono=True)
        
        # Ensure consistent length with better padding
        target_length = sr * 3
        if len(y) < target_length:
            # Use reflect padding for better audio continuity
            padding = target_length - len(y)
            pad_before = padding // 2
            pad_after = padding - pad_before
            y = np.pad(y, (pad_before, pad_after), mode='reflect')
        else:
            y = y[:target_length]
        
        features = []
        
        # 1. MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            coeff = mfcc[i]
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.max(coeff),
                np.min(coeff),
                np.median(coeff),
                np.percentile(coeff, 25),
                np.percentile(coeff, 75)
            ])
        
        # 2. MFCC deltas and delta-deltas
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        for feature_matrix in [mfcc_delta, mfcc_delta2]:
            for i in range(13):
                coeff = feature_matrix[i]
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(coeff)
                ])
        
        # 3. Mel-spectrogram statistics
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel)
        features.extend([
            np.mean(mel_db),
            np.std(mel_db),
            np.max(mel_db),
            np.min(mel_db)
        ])
        
        # 4. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean.tolist())
        
        # 5. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        # 6. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # 7. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([
            np.mean(rms),
            np.std(rms)
        ])
        
        # 8. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        features.extend(contrast_mean.tolist())
        
        # 9. Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        features.extend(tonnetz_mean.tolist())
        
        # 10. Additional features for robustness
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.extend([
            np.mean(y_harmonic),
            np.std(y_harmonic),
            np.mean(y_percussive),
            np.std(y_percussive)
        ])
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.extend([
            np.mean(flatness),
            np.std(flatness)
        ])
        
        # Poly features
        poly_features = librosa.feature.poly_features(y=y, sr=sr, order=2)
        for i in range(3):
            features.extend([
                np.mean(poly_features[i]),
                np.std(poly_features[i])
            ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return zeros of expected length (slightly larger now)
        return np.zeros(320)  # Increased for additional features

def create_dataset(data_dir="data/audio"):
    """
    Create dataset from audio files with better error handling
    """
    audio_paths = []
    labels = []
    
    print("Scanning audio files...")
    
    # Count total files first
    total_files = 0
    for root, _, files in os.walk(data_dir):
        total_files += len([f for f in files if f.endswith(".wav")])
    
    print(f"Total .wav files found: {total_files}")
    
    processed_files = 0
    skipped_files = 0
    
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                processed_files += 1
                try:
                    # RAVDESS filename format: XX-XX-XX-XX-XX-XX-XX-XX.wav
                    # Various filename patterns support
                    parts = f.split("-")
                    
                    # Try different patterns
                    emotion_code = None
                    if len(parts) >= 3:
                        # RAVDESS format: 03-01-XX-XX-XX-XX-XX.wav
                        emotion_code = parts[2]
                    elif "_" in f:
                        # Alternative format
                        emotion_code = f.split("_")[1] if len(f.split("_")) > 1 else None
                    
                    if emotion_code and emotion_code in EMOTION_MAP:
                        audio_paths.append(os.path.join(root, f))
                        labels.append(EMOTION_MAP[emotion_code])
                    else:
                        print(f"  Skipping {f}: Could not parse emotion code")
                        skipped_files += 1
                        
                except Exception as e:
                    print(f"  Error processing file {f}: {e}")
                    skipped_files += 1
                    continue
                
                if processed_files % 32 == 0:
                    print(f"  Processed {processed_files}/{total_files} files...")
    
    print(f"\nSuccessfully loaded {len(audio_paths)} audio files")
    print(f"Skipped {skipped_files} files due to errors or invalid format")
    print(f"Total files attempted: {processed_files}")
    
    return audio_paths, np.array(labels)

def balance_dataset(X, y):
    """
    Balance dataset using SMOTE for better performance
    """
    from collections import Counter
    from imblearn.over_sampling import SMOTE
    
    class_counts = Counter(y)
    print(f"Original class distribution: {dict(class_counts)}")
    
    # Use SMOTE for oversampling minority classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    balanced_counts = Counter(y_balanced)
    print(f"Balanced class distribution: {dict(balanced_counts)}")
    
    return X_balanced, y_balanced

def train_models():
    """
    Train multiple ML models and select the best one
    """
    print("="*80)
    print("TRAINING ML MODELS FOR SPEECH EMOTION RECOGNITION")
    print("="*80)
    
    # Step 1: Create dataset
    audio_paths, labels = create_dataset()
    
    if len(audio_paths) == 0:
        print("Error: No audio files found!")
        return
    
    print(f"\nClass distribution: {np.bincount(labels)}")
    print(f"Classes: {EMOTION_LABELS}")
    
    # Step 2: Extract features with progress tracking
    print(f"\nExtracting features from {len(audio_paths)} audio files...")
    X = []
    valid_indices = []
    failed_files = []
    
    for i, path in enumerate(audio_paths):
        if i % 160 == 0:
            print(f"  Processed {i}/{len(audio_paths)} files...")
        
        try:
            features = extract_enhanced_features(path)
            # Check if features are valid (not all zeros)
            if not np.all(features == 0):
                X.append(features)
                valid_indices.append(i)
            else:
                print(f"  Warning: All-zero features for {path}")
                failed_files.append(path)
        except Exception as e:
            print(f"  Error extracting features from {path}: {e}")
            failed_files.append(path)
            continue
    
    X = np.array(X)
    y = labels[valid_indices]
    
    print(f"\nFeature extraction complete!")
    print(f"Successfully processed: {len(X)} files")
    print(f"Failed to process: {len(failed_files)} files")
    
    if len(failed_files) > 0 and len(failed_files) <= 10:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Average feature vector length: {X.shape[1]}")
    
    if len(X) == 0:
        print("Error: No valid features extracted!")
        return
    
    # Step 3: Balance dataset
    print("\nBalancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y)
    print(f"Balanced dataset shape: {X_balanced.shape}")
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    
    # Step 4: Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42,
        stratify=y_balanced
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Step 5: Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 6: Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print('='*50)
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS, zero_division=0))
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    # Step 7: Select best model
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    best_model_name = None
    best_accuracy = 0
    
    print("\nRank | Model                | Test Accuracy | CV Accuracy")
    print("-" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        print(f"{rank:4d} | {name:20s} | {result['accuracy']:13.4f} | {result['cv_accuracy']:11.4f}")
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_name = name
    
    print(f"\n✅ Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Step 8: Train final model on all data
    print(f"\nTraining final model ({best_model_name}) on all data...")
    X_all_scaled = scaler.fit_transform(X_balanced)
    final_model = results[best_model_name]['model']
    
    # Re-train on all data
    final_model.fit(X_all_scaled, y_balanced)
    
    # Final evaluation
    final_cv_scores = cross_val_score(final_model, X_all_scaled, y_balanced, cv=5, 
                                     scoring='accuracy', n_jobs=-1)
    print(f"Final CV Accuracy: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std() * 2:.4f})")
    
    # Step 9: Save everything
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/ml_emotion_model.pkl"
    joblib.dump(final_model, model_path)
    
    # Save scaler
    scaler_path = "models/ml_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Save all models for ensemble voting later
    all_models_path = "models/all_ml_models.pkl"
    joblib.dump(results, all_models_path)
    
    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'feature_count': X.shape[1],
        'accuracy': float(results[best_model_name]['accuracy']),
        'cv_accuracy': float(final_cv_scores.mean()),
        'num_samples': len(X_balanced),
        'class_distribution': np.bincount(y_balanced).tolist(),
        'emotion_labels': EMOTION_LABELS,
        'feature_vector_length': X.shape[1],
        'all_model_accuracies': {name: float(res['accuracy']) for name, res in results.items()}
    }
    
    with open("models/ml_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ All models saved to {all_models_path}")
    print(f"✅ Metadata saved to models/ml_model_metadata.json")
    
    # Step 10: Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    if hasattr(final_model, 'feature_importances_'):
        # For tree-based models
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[-20:][::-1]
        
        print("\nTop 20 Most Important Features:")
        print("Rank | Feature Index | Importance")
        print("-" * 40)
        for rank, idx in enumerate(indices[:20]):
            print(f"{rank+1:4d} | {idx:13d} | {importances[idx]:.6f}")
    
    elif hasattr(final_model, 'coef_'):
        # For linear models
        if len(final_model.coef_.shape) == 2:
            # Multi-class
            avg_coef = np.mean(np.abs(final_model.coef_), axis=0)
            indices = np.argsort(avg_coef)[-20:][::-1]
            
            print("\nTop 20 Most Important Features (Average Coefficient Magnitude):")
            print("Rank | Feature Index | Importance")
            print("-" * 40)
            for rank, idx in enumerate(indices[:20]):
                print(f"{rank+1:4d} | {idx:13d} | {avg_coef[idx]:.6f}")
    
    # Step 11: Ensemble model (optional)
    print("\n" + "="*80)
    print("ENSEMBLE MODEL CREATION")
    print("="*80)
    
    from sklearn.ensemble import VotingClassifier
    
    # Create ensemble of top 3 models
    top_models = [(name, results[name]['model']) for name, _ in sorted_results[:3]]
    
    ensemble = VotingClassifier(
        estimators=top_models,
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test_scaled))
    print(f"Ensemble (Top 3 models) Accuracy: {ensemble_accuracy:.4f}")
    
    # Save ensemble model
    ensemble_path = "models/ensemble_model.pkl"
    joblib.dump(ensemble, ensemble_path)
    print(f"✅ Ensemble model saved to {ensemble_path}")
    
    return final_model, scaler, metadata, results

if __name__ == "__main__":
    try:
        model, scaler, metadata, results = train_models()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best Model: {metadata['best_model']}")
        print(f"Accuracy: {metadata['accuracy']:.4f}")
        print(f"Number of features: {metadata['feature_count']}")
        print(f"Number of samples: {metadata['num_samples']}")
        print(f"Emotions: {metadata['emotion_labels']}")
        
        # Print all model accuracies
        print("\nAll Model Accuracies:")
        for name, acc in metadata['all_model_accuracies'].items():
            print(f"  {name:20s}: {acc:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()