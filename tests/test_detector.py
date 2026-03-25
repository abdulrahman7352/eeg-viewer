import numpy as np
from eeg_viewer.detector import PowerSpikeDetector

def test_detector_fit_and_detect():
    powers = np.concatenate([np.ones(50) * 100, np.ones(20) * 200])
    times = np.arange(70) * 0.5
    det = PowerSpikeDetector(threshold_factor=1.5)
    det.fit_baseline(powers)
    events = det.detect(powers, times)
    assert len(events) > 0
