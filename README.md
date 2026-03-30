# 🧠 EEG Seizure Detection Viewer

**Live Web App:** [Click here to view the live detector](https://eeg-viewer-6lhd658bjf4fekxhmqnnl9.streamlit.app/)

## Overview
A production-ready EEG analysis tool with rule-based and machine learning seizure detection. Built for the CHB-MIT Scalp EEG Database, this system processes raw `.edf` (European Data Format) files to automatically flag candidate seizure events, saving clinical reviewers from scrolling through hours of raw data.

## Features
* **Cloud Deployed:** Interactive Streamlit UI hosted on Streamlit Community Cloud.
* **Automated Detection:** Identifies abnormal power spikes (Ictal events) using dynamic baseline calculation.
* **Dual Detection Modes:** Rule-based thresholding and ML-based classifier (RandomForest).
* **Clinical Feature Extraction:** Calculates band powers, Hjorth parameters, sample entropy, FOOOF 1/f slope, and alpha coherence.
* **Modular Architecture:** Clean separation of loader, processor, detector, and visualizer. Config-driven with no hardcoded values.

## Quick Start (Local Development)

```bash
git clone [https://github.com/abdulrahman7352/eeg-viewer.git](https://github.com/abdulrahman7352/eeg-viewer.git)
cd eeg-viewer
pip install -r requirements.txt
streamlit run app.py