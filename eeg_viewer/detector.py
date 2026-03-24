"""
detector.py
-----------
Responsibility: Detect candidate seizure events
from processed EEG band power timeseries.

Detection method: sliding window power spike detection.
A window is flagged when its band power exceeds a
multiple of the baseline power.

This is a rule-based detector — no ML model yet.
An ML model would replace or augment the threshold
rule in a production system.
"""

import numpy as np


class PowerSpikeDetector:
    """
    Detects windows where band power rises significantly
    above baseline — a simple proxy for seizure onset.

    Clinical reasoning:
    Seizures produce hypersynchronous neuronal firing
    which causes a large increase in EEG power across
    multiple frequency bands. Detecting this power
    increase is one of the simplest and most robust
    seizure detection approaches.

    Limitations (important for Design Note):
    - Cannot distinguish seizure from muscle artifact
    (both cause power increases)
    - Threshold is sensitive to baseline quality
    - Does not use spatial information across channels
    - A false positive rate has not been measured
    """

    def __init__(self, threshold_factor=2.5,
    n_baseline_windows=120):
        """
        Args:
        threshold_factor : power must exceed baseline
        by this multiple to be flagged
        2.5 = 250% of baseline power
        n_baseline_windows : number of windows at the
        start used to estimate baseline
        120 windows × 0.5s step = 60s
        """
        self.threshold_factor = threshold_factor
        self.n_baseline_windows = n_baseline_windows
        self.baseline_power = None
        self.threshold = None

    def fit_baseline(self, powers):
        """
        Estimate baseline power from the start
        of the recording (before any seizure activity).

        Uses MEDIAN not MEAN because:
        median is resistant to outliers
        if there is an early artifact it will not
        inflate the baseline estimate

        Args:
        powers : numpy array of band power values
        """
        n = min(self.n_baseline_windows, len(powers))
        self.baseline_power = np.median(powers[:n])
        self.threshold = (self.baseline_power *
        self.threshold_factor)

        print(f" Baseline power : {self.baseline_power:.2f}")
        print(f" Threshold : {self.threshold:.2f} "
              f"({self.threshold_factor}x baseline)")

    def detect(self, powers, times):
        """
        Find windows where power exceeds threshold.

        Args:
        powers : numpy array (n_windows,) of band power
        times : numpy array (n_windows,) of window times

        Returns:
        events : list of dicts, one per detected window
        each dict has:
        time_sec : centre time of window
        power : power value
        threshold : threshold used
        fold_change : how many times above baseline
        """
        if self.threshold is None:
            raise RuntimeError(
            "Call fit_baseline() before detect()."
            )

        events = []
        for power, time in zip(powers, times):
            if power > self.threshold:
                events.append({
                'time_sec' : float(time),
                'power' : float(power),
                'threshold' : float(self.threshold),
                'fold_change': float(
                power / self.baseline_power
                )
                })

        return events


def merge_nearby_events(events, gap_sec=5.0):
    """
    Merge events that are close together in time
    into single events.

    Why:
    A seizure lasting 40 seconds will trigger
    many consecutive windows (one every 0.5s).
    Without merging we get 80 separate events
    for one seizure.
    After merging we get 1 event with a start
    and end time — much more clinically useful.

    Args:
    events : list of event dicts from detect()
    gap_sec : events within this many seconds
    are merged into one

    Returns:
    merged : list of merged event dicts with
    start_time, end_time, peak_power,
    mean_fold_change, n_windows
    """
    if not events:
        return []

    merged = []
    current = {
    'start_time' : events[0]['time_sec'],
    'end_time' : events[0]['time_sec'],
    'peak_power' : events[0]['power'],
    'fold_changes' : [events[0]['fold_change']],
    'n_windows' : 1
    }

    for ev in events[1:]:
        if ev['time_sec'] - current['end_time'] <= gap_sec:
            # Extend current event
            current['end_time'] = ev['time_sec']
            current['peak_power'] = max(
            current['peak_power'], ev['power']
            )
            current['fold_changes'].append(ev['fold_change'])
            current['n_windows'] += 1
        else:
        # Save current, start new
            current['mean_fold_change'] = float(
            np.mean(current['fold_changes'])
            )
            del current['fold_changes']
            merged.append(current)

            current = {
            'start_time' : ev['time_sec'],
            'end_time' : ev['time_sec'],
            'peak_power' : ev['power'],
            'fold_changes': [ev['fold_change']],
            'n_windows' : 1
            }

    # Save the last event
    current['mean_fold_change'] = float(
    np.mean(current['fold_changes'])
    )
    del current['fold_changes']
    merged.append(current)

    return merged


def evaluate_against_annotation(merged_events,
seizure_start,
seizure_end,
tolerance_sec=30.0):
    """
    Check if our detector found the known seizure.

    Why this matters:
    In regulated software you must validate your
    detector against known ground truth.
    This function does a simple check:
    did any detected event overlap with the
    annotated seizure window?

    Args:
    merged_events : output of merge_nearby_events()
    seizure_start : known seizure start in seconds
    seizure_end : known seizure end in seconds
    tolerance_sec : how close counts as a detection

    Returns:
    result dict with detected, delay, summary string
    """
    if not merged_events:
        return {
        'detected' : False,
        'delay_sec' : None,
        'summary' : 'No events detected at all'
        }

    for ev in merged_events:
        overlap_start = max(ev['start_time'],
        seizure_start - tolerance_sec)
        overlap_end = min(ev['end_time'],
        seizure_end + tolerance_sec)

        if overlap_start <= overlap_end:
            delay = ev['start_time'] - seizure_start
            return {
            'detected' : True,
            'delay_sec' : delay,
            'event_start' : ev['start_time'],
            'event_end' : ev['end_time'],
            'peak_power' : ev['peak_power'],
            'mean_fold_change' : ev['mean_fold_change'],
            'summary' : (
            f"DETECTED at {ev['start_time']:.1f}s "
            f"(seizure at {seizure_start}s, "
            f"delay={delay:.1f}s)"
            )
            }

    return {
    'detected' : False,
    'delay_sec': None,
    'summary' : (
    f"Missed — no event near "
    f"{seizure_start}-{seizure_end}s"
    )
    }
