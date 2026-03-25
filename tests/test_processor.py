import numpy as np
from eeg_viewer.processor import apply_notch_filter

def test_notch_filter_preserves_shape():
    data = np.random.randn(4, 1000)
    filtered = apply_notch_filter(data, 256.0, 60.0)
    assert filtered.shape == data.shape
