import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from src.dataset import get_raw_for_subject

def start_simulator(subject_id=1):
    print(f"Loading data for Subject {subject_id}...")
    raw = get_raw_for_subject(subject_id)
    
    # Pick only EEG channels
    raw_eeg = raw.copy().pick_types(eeg=True)
    data = raw_eeg.get_data() # (channels, samples)
    srate = raw_eeg.info['sfreq']
    ch_names = raw_eeg.ch_names
    n_channels = len(ch_names)
    
    print(f"Dataset info: {n_channels} channels at {srate} Hz")

    # 1. Create LSL Stream Info
    # Name: BCI-Simulator, Type: EEG, Channels: 22, Rate: 250, Format: float32, ID: bci-sim-001
    info = StreamInfo('BCI-Simulator', 'EEG', n_channels, srate, 'float32', f'sub-{subject_id}')
    
    # Add channel labels to metadata
    desc = info.desc()
    channels = desc.append_child("channels")
    for name in ch_names:
        ch = channels.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("type", "EEG")
        ch.append_child_value("unit", "microvolts")

    # 2. Create Outlet
    outlet = StreamOutlet(info)
    
    print("Streaming started. Press Ctrl+C to stop.")
    print(f"LSL Stream Name: BCI-Simulator")
    
    # 3. Stream loop
    sample_idx = 0
    total_samples = data.shape[1]
    
    # To maintain precise timing
    start_time = time.time()
    
    try:
        while True:
            # Get current sample
            sample = data[:, sample_idx % total_samples].tolist()
            
            # Push to LSL
            outlet.push_sample(sample)
            
            sample_idx += 1
            
            # Calculate next sleep time to maintain sampling rate
            next_sample_time = start_time + (sample_idx / srate)
            sleep_time = next_sample_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if sample_idx % (int(srate) * 5) == 0:
                print(f"Streamed {sample_idx // int(srate)} seconds of data...")
                
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")

if __name__ == "__main__":
    # Ensure pylsl is installed: pip install pylsl
    import sys
    try:
        start_simulator(1)
    except ImportError:
        print("Error: pylsl not found. Please run 'pip install pylsl' first.")
        sys.exit(1)
