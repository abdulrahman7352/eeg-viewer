"""
compare_versions.py
-------------------
Compare v0.1.0 (12 features) vs v0.2.0 (15 features).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from eeg_viewer.loader import EDFLoader
from eeg_viewer.processor import run_preprocessing_pipeline
from eeg_viewer.feature_extractor import extract_features
from eeg_viewer.ml_detector import MLSeizureDetector

# Test file
TEST_FILE = 'data/chb01_03.edf'
SEIZURE_START, SEIZURE_END = 2996, 3036

print("="*60)
print("VERSION COMPARISON: v0.1.0 vs v0.2.0")
print("="*60)

# Load and preprocess
loader = EDFLoader(TEST_FILE)
loader.load()
data, times = loader.get_data_array()
clean = run_preprocessing_pipeline(data, loader.sfreq)

# Extract v0.2.0 features (15)
X, win_times = extract_features(clean, loader.sfreq)
print(f"\nv0.2.0 Features: {X.shape[1]}")
print(f"  - 12 basic (band power, Hjorth, etc.)")
print(f"  - 3 advanced (entropy, FOOOF slope, coherence)")

# Create labels
y = np.zeros(len(win_times))
for i, t in enumerate(win_times):
    if (t - 1.0) < SEIZURE_END and (t + 1.0) > SEIZURE_START:
        y[i] = 1

print(f"\nDataset: {len(X)} windows, {sum(y)} seizure ({100*sum(y)/len(y):.1f}%)")

# Train and evaluate
detector = MLSeizureDetector(n_estimators=200)
metrics = detector.train(X, y, validation_split=0.2)

print("\n" + "="*60)
print("v0.2.0 RESULTS:")
print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
print(f"  Specificity: {metrics['specificity']:.1%}")
print(f"  Val Accuracy: {metrics['val_acc']:.1%}")
print("="*60)
print("\nv0.1.0 baseline was: 80.1% sensitivity, 98.9% specificity")
print("Target for v0.2.0: 80%+ sensitivity")
