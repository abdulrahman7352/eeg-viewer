"""
feature_extractor.py
--------------------
Extract clinically-proven EEG features for ML.
"""

import numpy as np
from scipy import stats, signal


class FeatureExtractor:
    """
    Extract features from EEG windows.
    12 features proven effective for seizure detection.
    """
    
    def __init__(self, sfreq=256.0):
        self.sfreq = sfreq
        self.feature_names = [
            'delta_power', 'theta_power', 'alpha_power', 
            'beta_power', 'gamma_power',
            'theta_delta_ratio', 'alpha_theta_ratio', 'beta_alpha_ratio',
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            'line_length'
        ]
        
    def extract_window(self, data_window):
        """
        Extract all features from single window.
        
        Parameters:
        -----------
        data_window : array (n_channels, n_samples)
        
        Returns:
        --------
        features : array (12,)
        """
        avg_signal = np.mean(data_window, axis=0)
        
        # If signal is all zeros (average reference artifact), 
        # use single channel instead
        if np.std(avg_signal) < 1e-10:
            # Use first channel instead of average
            avg_signal = data_window[0, :]
        
        features = {}
        
        # === 1. BAND POWERS (FFT-based for accuracy) ===
        n = len(avg_signal)
        yf = np.fft.rfft(avg_signal)
        power_spectrum = np.abs(yf) ** 2
        freqs = np.fft.rfftfreq(n, 1/self.sfreq)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 70)
        }
        
        total_power = 0
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            # Sum power in band (more reliable than integration)
            band_power = np.sum(power_spectrum[mask])
            features[f'{band_name}_power'] = band_power
            total_power += band_power
        
        # Normalize to get relative powers (avoid numerical issues)
        if total_power > 0:
            for band_name in bands.keys():
                features[f'{band_name}_power'] /= (total_power / 100)  # Scale up
        
        # === 2. BAND RATIOS ===
        features['theta_delta_ratio'] = (
            features['theta_power'] / (features['delta_power'] + 1e-10)
        )
        features['alpha_theta_ratio'] = (
            features['alpha_power'] / (features['theta_power'] + 1e-10)
        )
        features['beta_alpha_ratio'] = (
            features['beta_power'] / (features['alpha_power'] + 1e-10)
        )
        
        # === 3. HJORTH PARAMETERS ===
        # Use standard deviation instead of variance for numerical stability
        std_x = np.std(avg_signal)
        features['hjorth_activity'] = std_x ** 2  # variance = std^2
        
        # Mobility and Complexity
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
        # Scale by window length to get meaningful values
        features['line_length'] = np.sum(np.abs(np.diff(avg_signal)))
        # Add small constant to avoid exact zeros in log features
        if features['line_length'] < 1e-10:
            features['line_length'] = 1e-10
        
        # Return in consistent order
        return np.array([features[name] for name in self.feature_names])
    
    def extract_all(self, data, window_sec=2.0, step_sec=0.5):
        """
        Extract features from all windows in recording.
        
        Returns:
        --------
        X : array (n_windows, 12)
        times : array (n_windows,)
        """
        win_samples = int(window_sec * self.sfreq)
        step_samples = int(step_sec * self.sfreq)
        n_samples = data.shape[1]
        
        features_list = []
        times = []
        
        for start in range(0, n_samples - win_samples, step_samples):
            end = start + win_samples
            window = data[:, start:end]
            
            feats = self.extract_window(window)
            features_list.append(feats)
            times.append((start + win_samples / 2) / self.sfreq)
        
        return np.array(features_list), np.array(times)


def extract_features(data, sfreq, window_sec=2.0, step_sec=0.5):
    """
    Convenience function.
    """
    extractor = FeatureExtractor(sfreq)
    return extractor.extract_all(data, window_sec, step_sec)