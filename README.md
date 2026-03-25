# EEG Viewer - ML-Based Seizure Detection

A production-ready EEG analysis tool with rule-based and machine learning seizure detection. Built for the CHB-MIT Scalp EEG Database.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-v0.2.0-orange.svg)

## Features

- **Dual Detection Modes**: Rule-based threshold or ML RandomForest classifier
- **15 Clinical Features**: Band powers, Hjorth parameters, sample entropy, FOOOF 1/f slope, alpha coherence
- **Interactive Visualization**: Scrollable EEG viewer with seizure event marking
- **Config-Driven**: All parameters in YAML - no hardcoded values
- **Modular Architecture**: Clean separation of loader, processor, detector, visualizer

## Quick Start

bash
# Clone repository
git clone https://github.com/abdulrahman7352/eeg-viewer.git
cd eeg-viewer

# Install dependencies
pip install -e .

# Download data (CHB-MIT)
# Place EDF files in data/ folder

# Run with rule-based detection
python scripts/run_viewer.py

# Run with ML detection (recommended)
python scripts/run_viewer.py --use-ml

# Train new model
python scripts/train_model.py