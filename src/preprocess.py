import mne
from mne.preprocessing import ICA
import numpy as np

def preprocess_raw(raw, l_freq=8.0, h_freq=30.0, use_ica=True):
    """
    Apply standard preprocessing to raw EEG data:
    1. Band-pass filtering
    2. ICA for artifact removal (EOG)
    """
    print(f"Applying band-pass filter ({l_freq}-{h_freq} Hz)...")
    # Filter EEG channels
    raw.filter(l_freq, h_freq, picks='eeg', fir_design='firwin')
    
    if use_ica:
        print("Running ICA for artifact removal...")
        # ICA needs a higher high-pass for better convergence, but we already filtered.
        # Usually 1Hz is better for ICA, but let's stick to our MI range for now 
        # or apply a temporary 1Hz filter if needed.
        ica = ICA(n_components=20, random_state=42, max_iter='auto')
        ica.fit(raw, picks='eeg')
        
        # Find EOG artifacts
        # BCI IV 2a has 3 EOG channels. MOABB labels them correctly?
        # Let's check ch_names for EOG.
        eog_chs = [ch for ch in raw.ch_names if 'EOG' in ch]
        if eog_chs:
            print(f"Found EOG channels: {eog_chs}. Finding artifacts...")
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs[0])
            ica.exclude = eog_indices
            print(f"Excluding ICA components: {eog_indices}")
        
        raw = ica.apply(raw)
        
    return raw

def create_epochs(raw, tmin=0.0, tmax=4.0):
    """
    Create epochs from preprocessed raw data.
    """
    print(f"Creating epochs (tmin={tmin}, tmax={tmax})...")
    events, event_id = mne.events_from_annotations(raw)
    
    # We only care about the 4 motor imagery classes
    mi_event_id = {k: v for k, v in event_id.items() if k in ['left_hand', 'right_hand', 'feet', 'tongue']}
    
    epochs = mne.Epochs(raw, events, event_id=mi_event_id, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, proj=False)
    
    return epochs

if __name__ == "__main__":
    from src.dataset import get_raw_for_subject
    
    # Process subject 1
    raw = get_raw_for_subject(1)
    # Subset to first few minutes for speed in testing if needed, 
    # but let's try the whole thing.
    raw_preprocessed = preprocess_raw(raw)
    epochs = create_epochs(raw_preprocessed)
    
    print(f"Created {len(epochs)} epochs.")
    print(f"Epoch data shape: {epochs.get_data().shape}") # (n_epochs, n_channels, n_times)
