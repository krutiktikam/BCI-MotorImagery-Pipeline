import streamlit as st
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
import time
import requests
from collections import deque

# Page config
st.set_page_config(page_title="NeuroRehab-BCI Dashboard", layout="wide")

st.title("🧠 NeuroRehab-BCI Real-Time Dashboard")

# --- Sidebar: Connection Settings ---
st.sidebar.header("Connection Settings")
stream_name = st.sidebar.text_input("LSL Stream Name", "BCI-Simulator")
api_url = st.sidebar.text_input("Inference API URL", "http://127.0.0.1:8000/predict")

# --- Initialize LSL Connection ---
@st.cache_resource
def get_lsl_inlet(name):
    try:
        streams = resolve_byprop('name', name, timeout=2)
        if streams:
            return StreamInlet(streams[0])
    except Exception as e:
        st.error(f"Error connecting to LSL: {e}")
    return None

inlet = get_lsl_inlet(stream_name)

if not inlet:
    st.warning(f"Waiting for LSL stream '{stream_name}'... Make sure the simulator is running.")
    if st.button("Retry Connection"):
        st.rerun()
    st.stop()

# --- State Management ---
if "eeg_buffer" not in st.session_state:
    # Buffer for visualization (last 2 seconds = 500 samples)
    st.session_state.eeg_buffer = deque(maxlen=500)
    st.session_state.prediction = {"class": "Idle", "confidence": 0.0, "probs": {}}

# --- Layout: Two Columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live EEG Signals")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Model Prediction")
    pred_placeholder = st.empty()
    prob_placeholder = st.empty()

# --- Update Loop ---
# We use a fragment to update only the charts/stats
@st.fragment(run_every=0.1)
def update_dashboard():
    # 1. Pull data from LSL
    # We pull multiple samples to keep up with the 250Hz rate
    samples = []
    while True:
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if not sample:
            break
        samples.append(sample)
    
    if samples:
        # Update buffer
        for s in samples:
            st.session_state.eeg_buffer.append(s)
            
        # 2. Prepare plot data
        data = np.array(st.session_state.eeg_buffer)
        # BCI IV 2a has 22 channels. Let's show a few main ones: C3, Cz, C4 (approx indices 7, 9, 11)
        # For simplicity, let's just show the first 3 channels
        channels_to_show = [0, 1, 2] 
        df = pd.DataFrame(data[:, channels_to_show], columns=[f"CH {i+1}" for i in channels_to_show])
        
        with col1:
            chart_placeholder.line_chart(df, height=400)

    # 3. Poll for latest prediction
    try:
        response = requests.get(f"{api_url.replace('/predict', '/latest')}", timeout=0.1)
        if response.status_code == 200:
            latest = response.json()
            pred = latest['prediction']
            conf = latest['confidence']
            probs = latest['all_probabilities']
            
            with col2:
                # Big prediction text
                pred_placeholder.metric("Prediction", pred.upper(), f"{conf*100:.1f}% confidence")
                
                # Probability bar chart
                if probs:
                    prob_df = pd.DataFrame({
                        'Task': list(probs.keys()),
                        'Probability': list(probs.values())
                    })
                    prob_placeholder.bar_chart(prob_df.set_index('Task'))
        else:
            with col2:
                pred_placeholder.error("Inference API: Error")
    except Exception as e:
        with col2:
            pred_placeholder.error(f"Inference API: Unreachable")

# Start updates
update_dashboard()
