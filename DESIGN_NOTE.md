# Design Note - EEG Viewer v0.2.0

## Architecture
Modular Python package with clean separation of concerns:
- loader: EDF file I/O
- processor: Signal preprocessing
- feature_extractor: 15 clinical features
- detector: Rule-based detection
- ml_detector: RandomForest ML
- visualizer: Interactive plots
- config: YAML configuration

## Versions
- v0.1.0: 12 features, 80.1% sensitivity, 98.9% specificity
- v0.2.0: 15 features, 80.7% sensitivity, 99.1% specificity

## Assumptions
- EDF files from CHB-MIT dataset
- 256 Hz sampling rate
- Seizure annotations available for training
