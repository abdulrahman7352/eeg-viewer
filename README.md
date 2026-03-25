# EEG Viewer - ML-Based Seizure Detection

A production-ready EEG analysis tool with rule-based and machine learning seizure detection. Built for the CHB-MIT Scalp EEG Database.

## Features

- Dual Detection Modes: Rule-based threshold or ML RandomForest classifier
- 15 Clinical Features: Band powers, Hjorth parameters, sample entropy, FOOOF 1/f slope, alpha coherence
- Interactive Visualization: Scrollable EEG viewer with seizure event marking
- Config-Driven: All parameters in YAML - no hardcoded values
- Modular Architecture: Clean separation of loader, processor, detector, visualizer

## Quick Start

```bash
git clone https://github.com/abdulrahman7352/eeg-viewer.git
cd eeg-viewer
pip install -e .
python scripts/run_viewer.py --use-ml