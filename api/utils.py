import torch
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=8.0, highcut=30.0, fs=250.0):
    """
    Apply bandpass filter to a numpy array (channels, time).
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    # Apply to each channel
    filtered_data = lfilter(b, a, data, axis=1)
    return filtered_data

def prepare_input(data):
    """
    Prepare raw window for model input.
    Input data: (channels, time)
    Output data: (1, 1, channels, time) as torch tensor
    """
    # Assuming data is already filtered or we filter here
    # For real-time, we might use a simple butter filter
    filtered = apply_bandpass(data)
    
    # Add batch and channel dimensions
    input_tensor = torch.from_numpy(filtered).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return input_tensor
