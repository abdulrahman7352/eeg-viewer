"""
train_model.py
--------------
Train ML model on CHB-MIT dataset with seizure annotations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from eeg_viewer.loader import EDFLoader
from eeg_viewer.processor import run_preprocessing_pipeline
from eeg_viewer.feature_extractor import extract_features
from eeg_viewer.ml_detector import MLSeizureDetector


# Known seizures from chb01-summary.txt
# Known seizures from chb01-summary.txt
# Format: (file_path, seizure_start_sec, seizure_end_sec)
TRAINING_FILES = [
    # chb01_03 - 1 seizure
    ('data/chb01_03.edf', 2996, 3036),
    
    # chb01_04 - 1 seizure  
    ('data/chb01_04.edf', 1467, 1494),
    
    # chb01_15 - 1 seizure
    ('data/chb01_15.edf', 1732, 1772),
    
    # chb01_16 - 1 seizure
    ('data/chb01_16.edf', 1015, 1066),
    
    # chb01_18 - 1 seizure
    ('data/chb01_18.edf', 1720, 1810),
    
    # chb01_21 - 1 seizure
    ('data/chb01_21.edf', 327, 420),
    
    # chb01_26 - 1 seizure
    ('data/chb01_26.edf', 1862, 1963),
]

def create_training_data(edf_path, seizure_start, seizure_end):
    """
    Create labeled dataset from one file.
    Seizure period = 1, rest = 0
    """
    print(f"\nProcessing {Path(edf_path).name}...")
    
    loader = EDFLoader(edf_path)
    loader.load()
    
    data, times = loader.get_data_array()
    clean = run_preprocessing_pipeline(data, loader.sfreq)
    
    # Extract features
    X, win_times = extract_features(clean, loader.sfreq, window_sec=2.0, step_sec=0.5)
    
    # Create labels: 1 if window overlaps with seizure
    y = np.zeros(len(win_times))
    for i, t in enumerate(win_times):
        window_start = t - 1.0  # 2-sec window centered at t
        window_end = t + 1.0
        if window_start < seizure_end and window_end > seizure_start:
            y[i] = 1
    
    n_seizure = int(sum(y))
    print(f"  Windows: {len(y)} | Seizure windows: {n_seizure} ({100*n_seizure/len(y):.1f}%)")
    
    return X, y


def main():
    """Train model on multiple files."""
    
    print("="*50)
    print("ML MODEL TRAINING")
    print("="*50)
    
    all_X = []
    all_y = []
    
    # Process each training file
    for edf_path, sz_start, sz_end in TRAINING_FILES:
        if not Path(edf_path).exists():
            print(f"\nSkipping {edf_path} - file not found")
            continue
            
        try:
            X, y = create_training_data(edf_path, sz_start, sz_end)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"  Error: {e}")
    
    if not all_X:
        print("\nNo training data found! Download files first.")
        return
    
    # Combine all data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\n{'='*50}")
    print(f"TOTAL DATASET:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Seizure ratio: {100*sum(y)/len(y):.2f}%")
    print(f"{'='*50}")
    
    # Train model
    detector = MLSeizureDetector(model_type='random_forest', n_estimators=200)
    metrics = detector.train(X, y, validation_split=0.2)
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'seizure_detector_v1.pkl'
    detector.save(str(model_path))
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"Model saved to: {model_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()