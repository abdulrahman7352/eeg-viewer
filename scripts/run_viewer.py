"""
Main entry point for EEG Viewer.
Usage: 
  python scripts/run_viewer.py                    # Use rule-based detector
  python scripts/run_viewer.py --use-ml           # Use ML detector
  python scripts/run_viewer.py --config custom.yaml --use-ml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_viewer.config import load_config
from eeg_viewer.loader import EDFLoader
from eeg_viewer.processor import run_preprocessing_pipeline
from eeg_viewer.feature_extractor import extract_features
from eeg_viewer.detector import PowerSpikeDetector, merge_nearby_events
from eeg_viewer.ml_detector import MLSeizureDetector
from eeg_viewer.visualizer import EEGViewer


def detect_rule_based(clean, sfreq, det_cfg):
    """Rule-based detection using band power."""
    from eeg_viewer.processor import compute_band_power
    
    powers, win_times = compute_band_power(
        clean, sfreq,
        band=det_cfg['band_hz'],
        window_sec=det_cfg['window_sec'],
        step_sec=det_cfg['step_sec']
    )
    
    detector = PowerSpikeDetector(threshold_factor=det_cfg['threshold_factor'])
    detector.fit_baseline(powers)
    raw_events = detector.detect(powers, win_times)
    events = merge_nearby_events(raw_events, gap_sec=det_cfg['gap_sec'])
    
    return events, powers, win_times, detector.threshold


def detect_ml(clean, sfreq, model_path='models/seizure_detector_v1.pkl'):
    """ML-based detection using trained model."""
    # Extract features
    X, win_times = extract_features(clean, sfreq, window_sec=2.0, step_sec=0.5)
    
    # Load model
    detector = MLSeizureDetector()
    detector.load(model_path)
    
    # Detect events
    events, probs = detector.detect(X, win_times, threshold=0.5)
    
    return events, probs, win_times, detector.threshold


def main(config_path="config.yaml", use_ml=False):
    cfg = load_config(config_path)
    
    data_cfg = cfg['data']
    proc_cfg = cfg['processing']
    det_cfg = cfg['detection']
    disp_cfg = cfg['display']
    
    # 1. Load
    print(f"Loading: {data_cfg['edf_path']}")
    loader = EDFLoader(data_cfg['edf_path'])
    loader.load()
    print(f"  {loader.n_channels} channels | {loader.sfreq} Hz | {loader.duration_sec:.1f}s")
    
    # 2. Preprocess
    print("Preprocessing...")
    data, times = loader.get_data_array()
    clean = run_preprocessing_pipeline(
        data, loader.sfreq,
        notch_hz=proc_cfg['notch_hz'],
        bandpass_low=proc_cfg['bandpass_hz'][0],
        bandpass_high=proc_cfg['bandpass_hz'][1]
    )
    
    # 3. Detect
    if use_ml:
        print("Detecting with ML model...")
        events, powers, win_times, threshold = detect_ml(clean, loader.sfreq)
    else:
        print("Detecting with rule-based method...")
        events, powers, win_times, threshold = detect_rule_based(clean, loader.sfreq, det_cfg)
    
    print(f"  {len(events)} events detected")
    
    # 4. Visualize
    print("Launching viewer - close window to exit...")
    viewer = EEGViewer(
        data_uv=clean,
        times=times,
        channel_names=loader.channel_names,
        sfreq=loader.sfreq,
        window_sec=disp_cfg['window_sec'],
        amplitude_scale=disp_cfg['amplitude_scale']
    )
    viewer.set_events(events)
    viewer.plot_power(powers, win_times, threshold)
    viewer.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-ml", action="store_true", 
                       help="Use ML model instead of rule-based detector")
    args = parser.parse_args()
    main(args.config, use_ml=args.use_ml)