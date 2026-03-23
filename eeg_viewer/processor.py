"""
processor.py
------------
Responsibility: Clean and process raw EEG data.
All functions take numpy arrays and return numpy arrays.
No plotting, no loading, no detection here.
"""

import numpy as np
from scipy import signal as sp_signal
import mne


# ── 1. NOTCH FILTER ─────────────────────────────────────────
def apply_notch_filter(data, sfreq, notch_hz=60.0):
    """
    Remove powerline noise using a notch filter.

    Why: Electrical power in the US runs at 60Hz.
    This creates a strong artifact in EEG at 60Hz.
    We remove it without touching other frequencies.

    Args:
    data : numpy array (n_channels, n_samples) in volts
    sfreq : sampling frequency in Hz
    notch_hz : powerline frequency to remove (60 US, 50 Europe)

    Returns:
    filtered data, same shape as input
    """
    b, a = sp_signal.iirnotch(notch_hz, Q=30, fs=sfreq)
    return sp_signal.filtfilt(b, a, data, axis=1)


# ── 2. BANDPASS FILTER ──────────────────────────────────────
def apply_bandpass_filter(data, sfreq,
low_hz=0.5, high_hz=70.0):
    """
    Keep only frequencies between low_hz and high_hz.

    Why: Brain signals live in 0.5-70Hz.
    Below 0.5Hz = slow drift (not brain activity)
    Above 70Hz = muscle noise and high freq artifacts

    Uses zero-phase Butterworth filter (filtfilt):
    filtfilt runs the filter FORWARD then BACKWARD.
    This cancels out any phase shift.
    Phase shift = time delay = wrong seizure timing.
    We cannot have that in clinical software.

    Args:
    data : numpy array (n_channels, n_samples)
    sfreq : sampling frequency in Hz
    low_hz : lower cutoff frequency
    high_hz : upper cutoff frequency

    Returns:
    filtered data, same shape as input
    """
    nyq = sfreq / 2.0
    low = low_hz / nyq
    high = high_hz / nyq

    # Clip to valid range just in case
    low = max(low, 0.001)
    high = min(high, 0.999)

    b, a = sp_signal.butter(4, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, data, axis=1)


# ── 3. BAD CHANNEL DETECTION ────────────────────────────────
def detect_bad_channels(data, sfreq,
flat_thresh=0.1,
noise_thresh=10.0):
    """
    Find channels that are flat or extremely noisy.

    Why: If an electrode falls off or makes poor contact,
    that channel has wrong data. Using it will corrupt
    all analyses. Better to flag and interpolate it.

    Flat channel = std < flat_thresh microvolts
    (electrode disconnected or broken)

    Noisy channel = std > noise_thresh * median std
    (too much noise compared to others)

    Args:
    data : numpy array (n_channels, n_samples)
    should be in microvolts for thresholds
    sfreq : sampling frequency
    flat_thresh : std below this = flat channel (uV)
    noise_thresh : multiplier above median std = noisy

    Returns:
    bad_indices : list of channel indices that are bad
    reason : dict {index: 'flat' or 'noisy'}
    """
    stds = np.std(data, axis=1)
    median_std = np.median(stds)
    bad_indices = []
    reason = {}

    for i, std in enumerate(stds):
        if std < flat_thresh:
            bad_indices.append(i)
            reason[i] = 'flat'
        elif std > noise_thresh * median_std:
            bad_indices.append(i)
            reason[i] = 'noisy'

    return bad_indices, reason


# ── 4. BAD CHANNEL INTERPOLATION ────────────────────────────
def interpolate_bad_channels(raw, bad_channel_names):
    """
    Replace bad channels using spherical spline interpolation.

    Why: Simply dropping bad channels changes the channel
    layout and breaks downstream analysis.
    Interpolation estimates what the bad channel
    should have looked like based on its neighbours.
    This is the clinical standard approach.

    Args:
    raw : MNE Raw object (from loader)
    bad_channel_names : list of channel name strings

    Returns:
    raw object with bad channels interpolated
    """
    if not bad_channel_names:
        return raw

    raw.info['bads'] = bad_channel_names

    # Set montage so MNE knows electrode positions
    # (needed for neighbour-based interpolation)
    try:
        montage = mne.channels.make_standard_montage(
        'standard_1020'
        )
        raw.set_montage(montage, match_case=False,
        on_missing='ignore')
        raw.interpolate_bads(reset_bads=True)
    except Exception as e:
        print(f" Warning: interpolation skipped — {e}")

    return raw


# ── 5. RE-REFERENCING ────────────────────────────────────────
def apply_average_reference(data):
    """
    Subtract the average of all channels from each channel.

    Why: Raw EEG is measured relative to a reference
    electrode (usually on the ear or scalp).
    The choice of reference affects all channels.
    Average reference is more neutral — it uses the
    mean of ALL channels as the reference point.
    This is standard in clinical EEG analysis.

    Args:
    data : numpy array (n_channels, n_samples)

    Returns:
    re-referenced data, same shape
    """
    mean_signal = np.mean(data, axis=0, keepdims=True)
    return data - mean_signal


# ── 6. BAND POWER (WELCH METHOD) ────────────────────────────
def compute_band_power(data, sfreq,
band,
window_sec=2.0,
step_sec=0.5):
    """
    Compute power in a frequency band using sliding windows.

    Why Welch method:
    Welch divides the signal into overlapping windows,
    computes FFT on each, then averages the results.
    This gives a smoother, more stable estimate of
    power than a single FFT on the whole signal.
    It is the standard method for clinical EEG power.

    Args:
    data : numpy array (n_channels, n_samples)
    sfreq : sampling frequency in Hz
    band : [low_hz, high_hz] frequency band
    window_sec : length of each sliding window
    step_sec : step between windows

    Returns:
    powers : numpy array (n_windows,)
    mean band power across channels
    times : numpy array (n_windows,)
    centre time of each window in seconds
    """
    win_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    n_samples = data.shape[1]

    powers = []
    times = []

    for start in range(0, n_samples - win_samples,
    step_samples):
        end = start + win_samples
        window = data[:, start:end]

        freqs, psd = sp_signal.welch(
        window,
        fs=sfreq,
        nperseg=win_samples,
        axis=1
        )

        mask = (freqs >= band[0]) & (freqs <= band[1])
        band_power = psd[:, mask].mean()

        powers.append(band_power)
        times.append((start + win_samples / 2) / sfreq)

    return np.array(powers), np.array(times)


# ── 7. CONVERT TO MICROVOLTS ────────────────────────────────
def convert_to_microvolts(data):
    """
    Convert from volts (MNE default) to microvolts.

    Why: MNE stores EEG in volts by default.
    EEG amplitudes are typically 10-100 microvolts.
    In volts that is 0.00001 to 0.0001.
    Microvolts are easier to work with and
    match clinical conventions.

    Args:
    data : numpy array in volts

    Returns:
    data in microvolts (multiplied by 1,000,000)
    """
    return data * 1e6


# ── 8. FULL PIPELINE ────────────────────────────────────────
def run_preprocessing_pipeline(data, sfreq,
notch_hz=60.0,
bandpass_low=0.5,
bandpass_high=70.0):
    """
    Run the complete preprocessing pipeline in order.

    Order matters:
    1. Notch first — remove powerline before bandpass
    2. Bandpass — keep brain frequencies only
    3. Convert uV — switch to microvolts
    4. Rereference — average reference last

    Args:
    data : raw numpy array (n_channels, n_samples)
    sfreq : sampling frequency
    notch_hz : powerline frequency
    bandpass_low : lower bandpass cutoff
    bandpass_high : upper bandpass cutoff

    Returns:
    clean data (n_channels, n_samples) in microvolts
    """
    print(" Applying notch filter...")
    data = apply_notch_filter(data, sfreq, notch_hz)

    print(" Applying bandpass filter...")
    data = apply_bandpass_filter(
    data, sfreq, bandpass_low, bandpass_high
    )

    print(" Converting to microvolts...")
    data = convert_to_microvolts(data)

    print(" Applying average reference...")
    data = apply_average_reference(data)

    return data
