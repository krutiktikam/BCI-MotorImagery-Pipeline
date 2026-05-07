import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from src.dataset import get_raw_for_subject
from src.preprocess import preprocess_raw, create_epochs
from models.arch.cnn_ts import EEGNet
from sklearn.preprocessing import LabelEncoder

class EEGDataset(Dataset):
    def __init__(self, epochs):
        # Pick only EEG channels
        epochs_eeg = epochs.copy().pick_types(eeg=True)
        self.data = epochs_eeg.get_data() # (n_epochs, n_channels, n_times)
        # Add channel dimension for CNN: (n_epochs, 1, n_channels, n_times)
        self.data = np.expand_dims(self.data, axis=1).astype(np.float32)
        
        # Labels
        self.labels = epochs.events[:, -1]
        # Map labels to 0-3
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/saved/best_model.pth')
            print("Saved best model weights.")

if __name__ == "__main__":
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and Preprocess for a single subject (Subject 1)
    raw = get_raw_for_subject(1)
    raw_pre = preprocess_raw(raw)
    epochs_data = create_epochs(raw_pre)
    
    # Dataset
    dataset = EEGDataset(epochs_data)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Model
    n_channels = dataset.data.shape[2]
    n_times = dataset.data.shape[3]
    model = EEGNet(n_channels=n_channels, n_times=n_times).to(device)
    
    # Train
    train_model(model, train_loader, val_loader, epochs=10, device=device)
