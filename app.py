import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

st.set_page_config(page_title="EEG Seizure Viewer", layout="wide")

st.title("🧠 EEG Seizure Detection Viewer")
st.markdown("Rule-based seizure detection (Power Threshold Method)")

uploaded_file = st.file_uploader("📁 Upload EDF file", type=['edf'])

if uploaded_file is not None:
    temp_path = Path("temp_upload.edf")
    temp_path.write_bytes(uploaded_file.getvalue())
    
    try:
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
        total_duration = int(raw.times[-1])
        sfreq = raw.info['sfreq']
        
        st.sidebar.header("⚙️ Detection Settings")
        
        # Smart baseline calculation based on file length
        if total_duration < 60:
            baseline_sec = max(10, total_duration // 3)
            st.sidebar.warning(f"Short file! Using first {baseline_sec}s for baseline")
        elif total_duration < 300:
            baseline_sec = 30
        else:
            baseline_sec = 120
        
        threshold_factor = st.sidebar.slider("Detection Sensitivity (x baseline)", 1.0, 10.0, 2.5)
        min_duration_sec = st.sidebar.slider("Min Seizure Duration (sec)", 2, 60, 5)
        
        channel_names = raw.ch_names
        selected_ch = st.sidebar.selectbox("Channel for detection", channel_names, index=0)
        
        full_data = raw.get_data(picks=selected_ch).flatten()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{total_duration} sec")
        col2.metric("Minutes", f"{total_duration/60:.1f} min")
        col3.metric("Channels", len(raw.ch_names))
        col4.metric("Sample Rate", f"{sfreq:.0f} Hz")
        
        st.info(f"Analyzing: {total_duration} seconds ({total_duration/60:.1f} minutes) on '{selected_ch}'")
        
        window_size = int(2 * sfreq)
        step_size = int(0.5 * sfreq)
        
        powers = []
        times = []
        
        for i in range(0, len(full_data) - window_size, step_size):
            window = full_data[i:i+window_size]
            power = np.mean(window**2)
            powers.append(power)
            times.append(i / sfreq)
        
        powers = np.array(powers)
        times = np.array(times)
        
        baseline_windows = int(baseline_sec / 0.5)
        baseline_segment = powers[:baseline_windows]
        
        if np.max(baseline_segment) < 1e-10:
            st.error("⚠️ First 2 minutes appear flat! Using middle section.")
            mid = len(powers) // 2
            baseline_segment = powers[mid:mid+baseline_windows]
            baseline_power = np.median(baseline_segment)
        else:
            baseline_power = np.median(baseline_segment)
        
        threshold = baseline_power * threshold_factor if baseline_power > 0 else 0.001
        
        st.write(f"Baseline: {baseline_power:.6e} (from first {baseline_sec}s)")
        st.write(f"Threshold ({threshold_factor}x): {threshold:.6e}")
        
        if baseline_power < 1e-10:
            st.error("Baseline is zero - detection may not work!")
        
        above_thresh = powers > threshold
        events = []
        in_event = False
        event_start = 0
        
        for i, (is_above, t) in enumerate(zip(above_thresh, times)):
            if is_above and not in_event:
                in_event = True
                event_start = t
            elif not is_above and in_event:
                duration = t - event_start
                if duration >= min_duration_sec:
                    start_idx = max(0, np.searchsorted(times, event_start) - 1)
                    end_idx = min(len(powers)-1, i)
                    peak_power = np.max(powers[start_idx:end_idx]) if end_idx > start_idx else powers[start_idx]
                    events.append({
                        'start': event_start,
                        'end': t,
                        'duration': duration,
                        'peak_power': peak_power,
                        'fold_change': peak_power / baseline_power if baseline_power > 0 else 0
                    })
                in_event = False
        
        if in_event:
            duration = times[-1] - event_start
            if duration >= min_duration_sec:
                start_idx = max(0, np.searchsorted(times, event_start) - 1)
                peak_power = np.max(powers[start_idx:])
                events.append({
                    'start': event_start,
                    'end': times[-1],
                    'duration': duration,
                    'peak_power': peak_power,
                    'fold_change': peak_power / baseline_power if baseline_power > 0 else 0
                })
        
        st.subheader(f"⚡ Detection Results: {len(events)} events found")
        
        if events:
            st.success(f"Detected {len(events)} events in {total_duration/60:.1f}-minute recording")
            
            fig, ax = plt.subplots(figsize=(15, 5))
            
            if total_duration > 3600:
                time_plot = times / 3600
                xlabel = "Time (hours)"
            elif total_duration > 300:
                time_plot = times / 60
                xlabel = "Time (minutes)"
            else:
                time_plot = times
                xlabel = "Time (seconds)"
            
            ax.plot(time_plot, powers, label='Power', color='blue', alpha=0.7)
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold')
            if baseline_power > 0:
                ax.axhline(y=baseline_power, color='g', linestyle=':', label='Baseline')
            
            for ev in events:
                ev_start_plot = ev['start'] / (3600 if total_duration > 3600 else 60 if total_duration > 300 else 1)
                ev_end_plot = ev['end'] / (3600 if total_duration > 3600 else 60 if total_duration > 300 else 1)
                ax.axvspan(ev_start_plot, ev_end_plot, alpha=0.3, color='red')
                ax.axvline(x=ev_start_plot, color='darkred', linestyle='-', linewidth=2)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Power")
            ax.set_title(f"Full Recording Analysis ({total_duration/60:.1f} minutes)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.write("**Detected Events:**")
            for i, ev in enumerate(events, 1):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"Event {i}", f"{ev['start']:.0f}s")
                col2.metric("End", f"{ev['end']:.0f}s")
                col3.metric("Duration", f"{ev['duration']:.0f}s")
                col4.metric("Above Baseline", f"{ev['fold_change']:.1f}x")
            
            st.markdown("---")
            st.subheader("🔍 View Specific Time Window")
            
            event_options = [f"Event {i+1}: {ev['start']:.0f}s - {ev['end']:.0f}s ({ev['duration']:.0f}s)" for i, ev in enumerate(events)]
            selected_event = st.selectbox("Jump to event:", event_options)
            event_num = int(selected_event.split(':')[0].replace('Event ', '')) - 1
            view_start = max(0, int(events[event_num]['start'] - 10))
            view_duration = st.slider("View Duration (sec)", 5, min(120, total_duration), 30)
        else:
            st.info("No sustained events detected")
            view_start = st.number_input("Start Time (sec)", 0, max(0, total_duration-30), 0)
            view_duration = st.slider("View Duration (sec)", 5, min(120, total_duration), 30)
        
        st.subheader(f"📊 EEG View: {view_start}s to {min(view_start+view_duration, total_duration)}s")
        
        start_sample = int(view_start * sfreq)
        end_sample = min(int((view_start + view_duration) * sfreq), len(full_data))
        
        picks = raw.ch_names[:min(8, len(raw.ch_names))]
        raw_view = raw.pick_channels(picks)
        data_view = raw_view.get_data(start=start_sample, stop=end_sample)
        times_view = np.arange(data_view.shape[1]) / sfreq + view_start
        
        fig2, axes = plt.subplots(len(picks), 1, figsize=(12, 8), sharex=True)
        if len(picks) == 1:
            axes = [axes]
        
        for i, (ax, ch_name) in enumerate(zip(axes, picks)):
            ax.plot(times_view, data_view[i], color='black', linewidth=0.5)
            ax.set_ylabel(ch_name, rotation=0, ha='right', fontsize=8)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for ev in events:
                if ev['start'] <= view_start + view_duration and ev['end'] >= view_start:
                    ax.axvspan(max(ev['start'], view_start), min(ev['end'], view_start+view_duration), 
                              alpha=0.2, color='red')
        
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        st.pyplot(fig2)
        
        temp_path.unlink()
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("👆 Upload chb01_03.edf, chb01_15.edf, etc.")
    st.markdown("""
    **Features:**
    - Full file analysis (10 min to 60+ min)
    - Smart baseline detection
    - Seizure detection at 1732s, 2996s, etc.
    """)