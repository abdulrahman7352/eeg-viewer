"""
loader.py
Responsibility: Load an EDF file and return EEG data.
Does ONE thing only - no filtering, no plotting here.
"""

import mne
import numpy as np
from pathlib import Path


class EDFLoader:

    def __init__(self, edf_path):
        self.edf_path = Path(edf_path)
        if not self.edf_path.exists():
            raise FileNotFoundError(
                f"EDF file not found: {self.edf_path}"
            )
        self._raw = None

    def load(self):
        self._raw = mne.io.read_raw_edf(
            str(self.edf_path),
            preload=True,
            verbose=False
        )
        return self._raw

    def _check_loaded(self):
        if self._raw is None:
            raise RuntimeError(
                "Data not loaded. Call load() first."
            )

    @property
    def channel_names(self):
        self._check_loaded()
        return self._raw.ch_names

    @property
    def sfreq(self):
        self._check_loaded()
        return self._raw.info['sfreq']

    @property
    def duration_sec(self):
        self._check_loaded()
        return self._raw.times[-1]

    @property
    def n_channels(self):
        self._check_loaded()
        return len(self._raw.ch_names)

    def get_data_array(self):
        self._check_loaded()
        data, times = self._raw[:]
        return data, times

    def get_window(self, t_start, t_end):
        self._check_loaded()
        start_idx = int(t_start * self.sfreq)
        end_idx = int(t_end * self.sfreq)
        data, times = self._raw[:, start_idx:end_idx]
        return data, times
