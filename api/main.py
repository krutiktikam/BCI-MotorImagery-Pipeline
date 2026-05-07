from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from models.arch.cnn_ts import EEGNet
from api.utils import apply_bandpass, prepare_input

app = FastAPI(title="NeuroRehab-BCI Inference API")

# Global state for latest prediction
latest_prediction = {
    "prediction": "idle",
    "confidence": 0.0,
    "all_probabilities": {}
}

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# We need the same params as used in training
# Subject 1 had 22 EEG channels and 1001 time points
N_CHANNELS = 22
N_TIMES = 1001
N_CLASSES = 4

model = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES)
try:
    model.load_state_dict(torch.load('models/saved/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Warning: best_model.pth not found. API will fail on prediction.")

class SignalWindow(BaseModel):
    data: list[list[float]] # Matrix of (22, 1001)

@app.get("/")
def read_root():
    return {"message": "BCI Inference API is running."}

@app.get("/latest")
async def get_latest():
    return latest_prediction

@app.post("/predict")
async def predict(window: SignalWindow):
    global latest_prediction
    try:
        # Convert to numpy
        arr = np.array(window.data)
        if arr.shape != (N_CHANNELS, N_TIMES):
            raise HTTPException(status_code=400, detail=f"Expected shape ({N_CHANNELS}, {N_TIMES}), got {arr.shape}")
        
        # Preprocess and To Tensor
        input_tensor = prepare_input(arr).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
        
        classes = ['left_hand', 'right_hand', 'feet', 'tongue']
        
        res = {
            "prediction": classes[predicted.item()],
            "confidence": probabilities[0][predicted.item()].item(),
            "all_probabilities": {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
        }
        
        latest_prediction = res
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
