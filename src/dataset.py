import mne
from moabb.datasets import BNCI2014_001
import os

def fetch_bci_iv_2a(subject_ids=None):
    """
    Fetch BCI Competition IV 2a dataset using MOABB.
    Returns a dictionary of MNE raw objects.
    """
    if subject_ids is None:
        subject_ids = [1]
    
    dataset = BNCI2014_001()
    print(f"Fetching MOABB data for subjects: {subject_ids}...")
    
    # get_data returns {subject: {session: {run: raw}}}
    data = dataset.get_data(subjects=subject_ids)
    return data

def get_raw_for_subject(subject_id):
    """
    Convenience function to get concatenated raw data for a subject.
    """
    data = fetch_bci_iv_2a([subject_id])
    subject_data = data[subject_id]
    
    # Combine all sessions and runs
    raw_list = []
    for session in subject_data:
        for run in subject_data[session]:
            raw_list.append(subject_data[session][run])
    
    # Concatenate all runs for simplicity in initial EDA
    combined_raw = mne.concatenate_raws(raw_list)
    return combined_raw

if __name__ == "__main__":
    # Test fetch for subject 1
    try:
        raw = get_raw_for_subject(1)
        print("Success!")
        print(raw.info)
    except Exception as e:
        print(f"Error fetching data: {e}")
