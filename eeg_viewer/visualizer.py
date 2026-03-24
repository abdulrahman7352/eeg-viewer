import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class EEGViewer:
    def __init__(self, data_uv, times, channel_names, sfreq,
                 window_sec=10.0, amplitude_scale=150.0):
        self.data = data_uv
        self.times = times
        self.ch_names = channel_names
        self.sfreq = sfreq
        self.window_sec = window_sec
        self.amp_scale = amplitude_scale
        self.n_channels = data_uv.shape[0]
        self.events = []
        self.t_start = 0.0
        self.duration = times[-1]
        self.power_data = None
        self.power_times = None
        self.power_threshold = None

    def set_events(self, events):
        self.events = events

    def plot_power(self, powers, times, threshold):
        self.power_data = powers
        self.power_times = times
        self.power_threshold = threshold

    def plot(self):
        fig, (ax, ax_pw) = plt.subplots(2, 1, figsize=(16, 10),
                                         gridspec_kw={'height_ratios': [4, 1]})
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        self.fig = fig
        self.ax = ax
        self.ax_pw = ax_pw
        
        if self.power_data is not None:
            self._draw_power()
        
        self._draw_window()
        ax_slider = plt.axes([0.15, 0.04, 0.70, 0.03])
        self.slider = Slider(ax_slider, 'Time (s)', 0.0,
                             max(0.1, self.duration - self.window_sec),
                             valinit=0.0, valstep=1.0)
        self.slider.on_changed(self._on_slider)
        plt.show()

    def _draw_power(self):
        self.ax_pw.cla()
        self.ax_pw.plot(self.power_times, self.power_data, 
                       color='darkorange', linewidth=1.0)
        self.ax_pw.axhline(self.power_threshold, color='red', linestyle='--',
                          linewidth=0.8, 
                          label=f'Threshold ({self.power_threshold:.2e})')
        self.ax_pw.set_xlabel("Time (s)")
        self.ax_pw.set_ylabel("Band Power")
        self.ax_pw.set_title("Sliding Window Band Power")
        self.ax_pw.legend(fontsize=8)

    def _draw_window(self):
        t_end = self.t_start + self.window_sec
        start_idx = int(self.t_start * self.sfreq)
        end_idx = int(t_end * self.sfreq)
        end_idx = min(end_idx, self.data.shape[1])
        self.ax.cla()
        times_win = self.times[start_idx:end_idx]
        
        # Plot each channel
        for i, name in enumerate(self.ch_names):
            offset = (self.n_channels - 1 - i) * self.amp_scale
            channel_data = self.data[i, start_idx:end_idx]
            self.ax.plot(times_win, channel_data + offset,
                        linewidth=0.5, color='steelblue')
            self.ax.text(self.t_start, offset, name,
                        fontsize=6, va='center', color='gray')
        
        # Mark detected events - use start_time and end_time
        for ev in self.events:
            start_t = ev['start_time']
            end_t = ev['end_time']
            # Check if event overlaps with current window
            if not (end_t < self.t_start or start_t > t_end):
                # Draw vertical line at start
                self.ax.axvline(start_t, color='red', alpha=0.6, 
                               linewidth=1.5, linestyle='--')
                # Draw shaded region for event duration
                self.ax.axvspan(start_t, end_t, color='red', alpha=0.1)
        
        self.ax.set_xlim(self.t_start, t_end)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Channels")
        self.ax.set_title(f"EEG Viewer - {len(self.events)} events detected")
        self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.t_start = val
        self._draw_window()
