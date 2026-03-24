"""
ml_detector.py
--------------
Machine learning seizure detector using RandomForest.
Trains on extracted features, predicts seizure probability.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')


class MLSeizureDetector:
    """
    ML-based seizure detector with feature scaling.
    """
    
    def __init__(self, model_type='random_forest', n_estimators=200):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.is_trained = False
        self.feature_importance = None
        
    def train(self, X, y, validation_split=0.2):
        """
        Train the model on feature matrix.
        
        Parameters:
        -----------
        X : array (n_samples, n_features)
            Feature matrix from feature_extractor
        y : array (n_samples,)
            Labels: 0 = normal, 1 = seizure
        validation_split : float
            Fraction for validation
        """
        print(f"\nTraining {self.model_type} on {len(X)} samples...")
        print(f"  Features: {X.shape[1]}")
        print(f"  Seizure samples: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, 
            stratify=y, random_state=42
        )
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        # Calculate sensitivity/specificity on validation set
        y_val_pred = self.model.predict(X_val)
        tp = np.sum((y_val == 1) & (y_val_pred == 1))
        tn = np.sum((y_val == 0) & (y_val_pred == 0))
        fp = np.sum((y_val == 0) & (y_val_pred == 1))
        fn = np.sum((y_val == 1) & (y_val_pred == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Val accuracy:   {val_acc:.3f}")
        print(f"  Sensitivity:    {sensitivity:.3f}")
        print(f"  Specificity:    {specificity:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            print(f"\n  Top 3 features:")
            top_idx = np.argsort(self.feature_importance)[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                print(f"    {i}. Feature {idx}: {self.feature_importance[idx]:.3f}")
        
        self.is_trained = True
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def predict_proba(self, X):
        """Predict seizure probability for windows."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def detect(self, X, times, threshold=None):
        """
        Detect seizures from predictions.
        
        Returns:
        --------
        events : list of dicts with start_time, end_time, confidence
        probs : array of probabilities
        """
        if threshold is None:
            threshold = self.threshold
            
        probs = self.predict_proba(X)
        
        # Find contiguous regions above threshold
        events = []
        in_event = False
        event_start = None
        event_probs = []
        
        for i, (t, p) in enumerate(zip(times, probs)):
            if p >= threshold and not in_event:
                in_event = True
                event_start = t
                event_probs = [p]
            elif p >= threshold and in_event:
                event_probs.append(p)
            elif p < threshold and in_event:
                in_event = False
                events.append({
                    'start_time': float(event_start),
                    'end_time': float(t),
                    'peak_prob': float(np.max(event_probs)),
                    'mean_prob': float(np.mean(event_probs)),
                    'n_windows': len(event_probs)
                })
        
        # Handle case where event continues to end
        if in_event:
            events.append({
                'start_time': float(event_start),
                'end_time': float(times[-1]),
                'peak_prob': float(np.max(event_probs)),
                'mean_prob': float(np.mean(event_probs)),
                'n_windows': len(event_probs)
            })
        
        return events, probs
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'threshold': self.threshold,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance
            }, f)
        print(f"\nModel saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.threshold = data['threshold']
            self.model_type = data['model_type']
            self.feature_importance = data.get('feature_importance')
        self.is_trained = True
        print(f"Model loaded from {filepath}")