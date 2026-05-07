import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, n_times=1001, F1=8, D=2, F2=16, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal & Spatial Conv
        self.block1 = nn.Sequential(
            # Temporal Conv
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            # Spatial Conv (Depthwise)
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 2: Separable Conv
        self.block2 = nn.Sequential(
            # Depthwise Separable Conv
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        # Calculate flatten size
        # n_times -> Block 1 Pool (1/4) -> Block 2 Pool (1/8) = n_times // 32
        self.flatten_size = F2 * (n_times // 32)
        
        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        # Input shape: (Batch, 1, Channels, Time)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Test model with dummy input
    model = EEGNet(n_channels=22, n_times=1001)
    dummy_input = torch.randn(1, 1, 22, 1001)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
