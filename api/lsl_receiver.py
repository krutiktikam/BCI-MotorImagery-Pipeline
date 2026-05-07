import numpy as np
import requests
import time
from pylsl import StreamInlet, resolve_byprop

def start_receiver(stream_name='BCI-Simulator'):
    print(f"Looking for LSL stream: {stream_name}...")
    # Resolve the stream
    streams = resolve_byprop('name', stream_name, timeout=5)
    if not streams:
        print(f"Error: Could not find stream {stream_name}. Is the simulator running?")
        return

    inlet = StreamInlet(streams[0])
    print(f"Connected to {stream_name}. Starting real-time prediction...")

    # Parameters to match model
    N_CHANNELS = 22
    N_TIMES = 1001
    
    # Buffer to hold data
    # (channels, time)
    buffer = np.zeros((N_CHANNELS, N_TIMES))
    
    # API endpoint
    api_url = "http://127.0.0.1:8000/predict"

    print("Buffering initial window...")
    
    count = 0
    try:
        while True:
            # Pull a single sample
            # sample is a list of channel values
            sample, timestamp = inlet.pull_sample()
            
            # Shift buffer and add new sample
            buffer[:, :-1] = buffer[:, 1:]
            buffer[:, -1] = sample
            
            count += 1
            
            # Every 1 second (250 samples), send a prediction request
            if count >= 250:
                print(f"--- Sending window for prediction ---")
                
                payload = {"data": buffer.tolist()}
                
                try:
                    response = requests.post(api_url, json=payload, timeout=0.5)
                    if response.status_code == 200:
                        res = response.json()
                        pred = res['prediction']
                        conf = res['confidence']
                        print(f"Result: {pred.upper()} ({conf*100:.1f}%)")
                    else:
                        print(f"API Error: {response.status_code}")
                except Exception as e:
                    print(f"Request failed: {e}")
                
                count = 0 # Reset counter for next sliding window
                
    except KeyboardInterrupt:
        print("\nReceiver stopped.")

if __name__ == "__main__":
    start_receiver()
