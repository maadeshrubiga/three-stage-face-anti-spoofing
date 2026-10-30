
import sys
import os
import torch
import cv2
import numpy as np
from torchvision import transforms

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from models import ForensicNet, FastScreenNet
except ImportError:
    print("Could not import models")
    sys.exit(1)

# Transformations (COPIED from analysis_engine.py)
trans_s2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def test_inference():
    print(f"Device: {DEVICE}")
    
    # 1. Load Model
    model = ForensicNet().to(DEVICE)
    # Update to V3 weights as per user
    path = os.path.join(PARENT_DIR, "weights_1/stage2_v3_best.pth")
    if os.path.exists(path):
        print(f"Loading weights from {path}")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        
        # Check bias
        try:
            bias = model.head[4].bias.item()
            print(f"Final Layer Bias: {bias:.4f}")
            weight_mean = model.head[4].weight.mean().item()
            print(f"Final Layer Weight Mean: {weight_mean:.4f}")
        except:
            print("Could not inspect final layer.")
    else:
        print(f"Weights not found at {path}")
        return

    # 2. Create Dummy Image (Gray/Face-like) or Load one
    # Let's create a random noise image and a solid color image
    print("\nTest 1: Random Noise (BGR)")
    img_bgr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Run Inference (BGR vs RGB)
    with torch.no_grad():
        # Case A: Standard
        t_bgr = trans_s2(img_bgr).unsqueeze(0).to(DEVICE)
        print(f"Standard Input Stats: Min={t_bgr.min():.2f}, Max={t_bgr.max():.2f}, Mean={t_bgr.mean():.2f}")
        
        logit_bgr = model(t_bgr)
        prob_bgr = torch.sigmoid(logit_bgr).item()
        print(f"Standard Input -> Raw Logit: {logit_bgr.item():.4f}, Prob: {prob_bgr:.6f}")

        # Case C: 128x128 Resolution (Possible Training Mismatch?)
        trans_128 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t_128 = trans_128(img_bgr).unsqueeze(0).to(DEVICE)
        
        logit_128 = model(t_128)
        prob_128 = torch.sigmoid(logit_128).item()
        print(f"128x128 Input -> Raw Logit: {logit_128.item():.4f}, Prob: {prob_128:.6f}")


if __name__ == "__main__":
    test_inference()
