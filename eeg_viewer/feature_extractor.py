"""
feature_extractor.py
--------------------
Extract EEG features for ML - v0.2.0 with advanced features.
Features: 15 total (12 basic + sample entropy + FOOOF slope + coherence)
"""

import numpy as np
from scipy import stats, signal
from fooof import FOOOF


class FeatureExtractor:
    """
    Extract features from EEG windows.
    15 features for improved seizure detection.
    """
    
    def __init__(self, sfreq=256.0):
        self.sfreq = sfreq
        self.feature_names = [
            # Basic 12
            'delta_power', 'theta_power', 'alpha_power', 
            'beta_power', 'gamma_power',
            'theta_delta_ratio', 'alpha_theta_ratio', 'beta_alpha_ratio',
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            'line_length',
            # Advanced 3 (v0.2.0)
            'sample_entropy',
            'fooof_slope',
            'alpha_coherence'
        ]
        
    def extract_window(self, data_window):
        """Extract all features from single window."""
        # Spatial average
        avg_signal = np.mean(data_window, axis=0)
        
        # Fallback if average is zero
        if np.std(avg_signal) < 1e-10:
            avg_signal = data_window[0, :]
        
        features = {}
        
        # === 1. BAND POWERS (FFT) ===
        n = len(avg_signal)
        yf = np.fft.rfft(avg_signal)
        power_spectrum = np.abs(yf) ** 2
        freqs = np.fft.rfftfreq(n, 1/self.sfreq)
        
        bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 70)
        }
        
        total_power = 0
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(power_spectrum[mask])
            features[f'{band_name}_power'] = band_power
            total_power += band_power
        
        if total_power > 0:
            for band_name in bands.keys():
                features[f'{band_name}_power'] /= (total_power / 100)
        
        # === 2. BAND RATIOS ===
        features['theta_delta_ratio'] = features['theta_power'] / (features['delta_power'] + 1e-10)
        features['alpha_theta_ratio'] = features['alpha_power'] / (features['theta_power'] + 1e-10)
        features['beta_alpha_ratio'] = features['beta_power'] / (features['alpha_power'] + 1e-10)
        
        # === 3. HJORTH ===
        std_x = np.std(avg_signal)
        features['hjorth_activity'] = std_x ** 2
        dx = np.diff(avg_signal)
        std_dx = np.std(dx)
        
        if std_x > 1e-10:
            features['hjorth_mobility'] = std_dx / std_x
            ddx = np.diff(dx)
            std_ddx = np.std(ddx)
            if std_dx > 1e-10:
                mob_dx = std_ddx / std_dx
                features['hjorth_complexity'] = features['hjorth_mobility'] / mob_dx
            else:
                features['hjorth_complexity'] = 0
        else:
            features['hjorth_mobility'] = 0
            features['hjorth_complexity'] = 0
        
        # === 4. LINE LENGTH ===
        features['line_length'] = np.sum(np.abs(dx))
        if features['line_length'] < 1e-10:
            features['line_length'] = 1e-10
        
        # === 5. SAMPLE ENTROPY (v0.2.0) ===
        features['sample_entropy'] = self._sample_entropy(avg_signal)
        
        # === 6. FOOOF SLOPE (v0.2.0) ===
        features['fooof_slope'] = self._fooof_slope(freqs, power_spectrum)
        
        # === 7. ALPHA COHERENCE (v0.2.0) ===
        features['alpha_coherence'] = self._alpha_coherence(data_window, freqs)
        
        return np.array([features[name] for name in self.feature_names])
    
    def _sample_entropy(self, signal, m=2, r=0.2):
        """Sample entropy - lower during seizures (more regular)."""
        N = len(signal)
        if N < m + 1:
            return 0
        
        r = r * np.std(signal)
        
        def _count_matches(template, m_val):
            count = 0
            for i in range(N - m_val):
                if i == template:
                    continue
                match = True
                for k in range(m_val):
                    if abs(signal[template + k] - signal[i + k]) > r:
                        match = False
                        break
                if match:
                    count += 1
            return count
        
        B, A = 0, 0
        for i in range(N - m):
            B += _count_matches(i, m)
        for i in range(N - m - 1):
            A += _count_matches(i, m + 1)
        
        if B == 0 or A == 0:
            return 0
        return -np.log(A / B)
    
    def _fooof_slope(self, freqs, power_spectrum):
        """FOOOF aperiodic slope - shallower in epileptic brain."""
        try:
            # Fit FOOOF: 1-40 Hz, no peaks
            fm = FOOOF(peak_width_limits=[2, 8], min_peak_height=0.05, verbose=False)
            mask = (freqs >= 1) & (freqs <= 40)
            fm.fit(freqs[mask], power_spectrum[mask])
            return fm.aperiodic_params_[1]  # Slope is 2nd parameter
        except:
            return -1.0  # Default slope if fit fails
    
    def _alpha_coherence(self, data_window, freqs):
        """Alpha band (8-13 Hz) coherence across channel pairs."""
        n_channels = data_window.shape[0]
        if n_channels < 2:
            return 0
        
        # Compute coherence between first 2 channels in alpha band
        fmin, fmax = 8, 13
        f, Cxy = signal.coherence(data_window[0], data_window[1], 
                                   fs=self.sfreq, nperseg=256)
        alpha_mask = (f >= fmin) & (f <= fmax)
        return np.mean(Cxy[alpha_mask]) if np.any(alpha_mask) else 0
    
    def extract_all(self, data, window_sec=2.0, step_sec=0.5):
        """Extract features from all windows."""
        win_samples = int(window_sec * self.sfreq)
        step_samples = int(step_sec * self.sfreq)
        n_samples = data.shape[1]
        
        features_list, times = [], []
        
        for start in range(0, n_samples - win_samples, step_samples):
            end = start + win_samples
            window = data[:, start:end]
            feats = self.extract_window(window)
            features_list.append(feats)
            times.append((start + win_samples / 2) / self.sfreq)
        
        return np.array(features_list), np.array(times)


def extract_features(data, sfreq, window_sec=2.0, step_sec=0.5):
    """Convenience function."""
    extractor = FeatureExtractor(sfreq)
    return extractor.extract_all(data, window_sec, step_sec)