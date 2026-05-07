# NeuroRehab-BCI: Real-Time Motor Imagery Classification

An end-to-end Brain-Computer Interface (BCI) pipeline for classifying Motor Imagery (MI) EEG signals using Deep Learning (EEGNet) and FastAPI.

## 🚀 Features
- **Deep Learning Pipeline**: Uses EEGNet (Temporal-Spatial Convolution) for high-accuracy MI classification.
- **FastAPI Inference**: Production-ready API for real-time signal processing and prediction.
- **Hardware Simulation**: Full LSL (Lab Streaming Layer) simulation suite to test the pipeline without an actual EEG device.
- **Standardized Data**: Integration with MOABB to fetch and process BCI Competition IV-2a datasets.

## 🛠️ Architecture
1. **Data**: BCI Competition IV-2a (4-class: Left, Right, Feet, Tongue).
2. **Preprocessing**: 8-30Hz Bandpass filter + ICA (Artifact removal).
3. **Inference**: FastAPI server hosting a PyTorch model.
4. **Communication**: LSL for real-time data streaming.

## 📦 Installation
```powershell
# Clone the repository
git clone <your-repo-url>
cd BCI_MI_pjt

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage (Real-Time Simulation)
To test the system without hardware, open three terminals:

1. **Terminal 1: Start Virtual EEG**
   ```powershell
   $env:PYTHONPATH = "."; python api/lsl_simulator.py
   ```
2. **Terminal 2: Start Inference API**
   ```powershell
   $env:PYTHONPATH = "."; python -m api.main
   ```
3. **Terminal 3: Start Real-Time Bridge**
   ```powershell
   $env:PYTHONPATH = "."; python api/lsl_receiver.py
   ```

## 📂 Project Structure
- `api/`: FastAPI server, LSL utilities, and simulators.
- `models/`: Model architectures (EEGNet) and saved weights.
- `src/`: Core logic for data fetching, preprocessing, and training.
- `doc/`: Detailed project plans and logs.

## 📝 License
MIT
