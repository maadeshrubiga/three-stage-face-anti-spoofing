import torch
import torch.nn as nn
import torchvision.models as models

# --- UTILITY: Frequency Analysis ---
def compute_fft(x):
    """
    Converts RGB images to Frequency Domain to spot Moiré patterns (Screen Pixels).
    """
    # Convert to grayscale: [Batch, 1, H, W]
    x_gray = torch.mean(x, dim=1, keepdim=True)
    # FFT
    fft = torch.fft.fft2(x_gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = 20 * torch.log(torch.abs(fft_shift) + 1e-6)
    return magnitude

# --- STAGE 1: Fast Screening (MobileNetV3) ---
class FastScreenNet(nn.Module):
    def __init__(self):
        super(FastScreenNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        # Binary Classification: 0=Spoof, 1=Live
        self.backbone.classifier[3] = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))

# --- STAGE 2: Forensic Expert (EfficientNet-V2-M + FFT) ---
class ForensicNet(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB Branch (EfficientNet-V2-M)
        self.rgb_backbone = models.efficientnet_v2_m(weights="DEFAULT")
        self.rgb_backbone.classifier = nn.Identity()
        # Output: 1280 features

        # FFT Branch (Frequency)
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # FFT output = 128

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(1280 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)   # NO sigmoid here (uses BCEWithLogitsLoss)
        )

    def forward(self, x):
        f_rgb = self.rgb_backbone(x)
        x_fft = compute_fft(x)  # (B,1,224,224)
        f_fft = self.fft_branch(x_fft)
        fused = torch.cat([f_rgb, f_fft], dim=1)
        return self.head(fused)