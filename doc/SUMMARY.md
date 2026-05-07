# Technical Summary: NeuroRehab-BCI

## Overview
This project implements a real-time motor imagery classification system designed for neuro-rehabilitation applications. It bridges the gap between offline research and live application by providing a robust LSL-based streaming architecture.

## Technical Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Signal Processing**: MNE-Python, SciPy
- **Backend**: FastAPI, Uvicorn
- **Data Streaming**: PyLSL

## Key Components
### 1. Signal Processing (`src/preprocess.py`)
- **Filtering**: 5th order Butterworth bandpass (8-30 Hz) targeting Mu and Beta rhythms.
- **Artifact Removal**: Independent Component Analysis (ICA) to isolate EOG/EMG noise.
- **Windowing**: 4-second epochs (1001 samples at 250 Hz).

### 2. Deep Learning Model (`models/arch/cnn_ts.py`)
- **Architecture**: EEGNet.
- **Temporal Conv**: Extracts frequency features.
- **Spatial Conv**: Learns topographical patterns across 22 EEG channels.
- **Separable Conv**: Reduces parameters while maintaining feature depth.

### 3. Simulation Suite (`api/`)
- **LSL Simulator**: Converts static MOABB dataset runs into real-time streams.
- **LSL Receiver**: Implements a sliding window buffer to provide continuous inference.

## Performance Metrics (Initial)
- **Subject 1 (BCI IV-2a)**: ~56% accuracy in 10 epochs (Baseline 25%).
- **Inference Latency**: <200ms per window.
