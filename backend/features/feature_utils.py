# features/feature_utils.py
import os
import numpy as np
import joblib
from pathlib import Path
from config import FEATURES_DIR

class FeatureCache:
    """
    Cache features to disk to avoid recomputation
    """
    def __init__(self, cache_dir=FEATURES_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, audio_path, feature_type):
        """Generate cache file path"""
        audio_name = Path(audio_path).stem
        return self.cache_dir / f"{audio_name}_{feature_type}.npy"
    
    def save_features(self, audio_path, feature_type, features):
        """Save features to cache"""
        cache_path = self.get_cache_path(audio_path, feature_type)
        np.save(cache_path, features)
    
    def load_features(self, audio_path, feature_type):
        """Load features from cache"""
        cache_path = self.get_cache_path(audio_path, feature_type)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def clear_cache(self):
        """Clear all cached features"""
        for file in self.cache_dir.glob("*.npy"):
            file.unlink()

class FeatureSelector:
    """
    Select optimal features to avoid overfitting
    """
    def __init__(self):
        self.selected_features = None
        self.feature_importance = None
    
    def select_features_rf(self, X, y, n_features=50):
        """
        Select features using Random Forest importance
        """
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-n_features:][::-1]
        
        self.selected_features = indices
        self.feature_importance = importances[indices]
        
        return X[:, indices], indices
    
    def select_features_pca(self, X, n_components=50):
        """
        Reduce dimensionality using PCA
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        
        self.explained_variance = pca.explained_variance_ratio_
        
        return X_reduced, pca
    
    def select_features_forward(self, X, y, n_features=50):
        """
        Forward feature selection
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.ensemble import RandomForestClassifier
        
        sfs = SequentialFeatureSelector(
            RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features,
            direction='forward',
            cv=3,
            n_jobs=-1
        )
        
        X_selected = sfs.fit_transform(X, y)
        self.selected_features = sfs.get_support(indices=True)
        
        return X_selected, self.selected_features

def balance_dataset(X, y, method='undersample'):
    """
    Balance dataset to prevent overfitting to majority class
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=3)
    else:
        return X, y
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    print(f"Balanced dataset: {len(y)} -> {len(y_balanced)} samples")
    print(f"Class distribution: {np.bincount(y_balanced)}")
    
    return X_balanced, y_balanced