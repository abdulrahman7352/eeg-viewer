from eeg_viewer.loader import EDFLoader
from eeg_viewer.processor import run_preprocessing_pipeline
from eeg_viewer.feature_extractor import extract_features

loader = EDFLoader('data/chb01_03.edf')
loader.load()
data, times = loader.get_data_array()
clean = run_preprocessing_pipeline(data, loader.sfreq)

X, win_times = extract_features(clean, loader.sfreq)
print(f'Features: {X.shape[1]} (should be 15)')
print(f'Feature names: 15 features')
print(f'Sample entropy: {X[0,12]:.3f}')
print(f'FOOOF slope: {X[0,13]:.3f}')
print(f'Alpha coherence: {X[0,14]:.3f}')
