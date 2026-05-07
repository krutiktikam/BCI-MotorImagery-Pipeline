import requests
import numpy as np
import torch
from src.dataset import get_raw_for_subject
from src.preprocess import preprocess_raw, create_epochs
import time

def test_api():
    print("Fetching sample data for testing...")
    # Get subject 1 data
    raw = get_raw_for_subject(1)
    # Preprocess (excluding ICA for simplicity in this test window if we want, 
    # but let's do it to match training)
    raw_pre = preprocess_raw(raw, use_ica=False) 
    epochs = create_epochs(raw_pre)
    
    # Get one sample
    epochs_eeg = epochs.copy().pick_types(eeg=True)
    sample_data = epochs_eeg.get_data()[0] # (22, 1001)
    sample_label = epochs_eeg.events[0, -1]
    
    classes = {1: 'left_hand', 2: 'right_hand', 3: 'feet', 4: 'tongue'}
    print(f"Sample ground truth: {classes.get(sample_label, 'unknown')}")

    # API endpoint
    url = "http://127.0.0.1:8000/predict"
    
    payload = {
        "data": sample_data.tolist()
    }
    
    print("Sending request to API...")
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            print(f"Inference took: {end_time - start_time:.4f} seconds")
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
