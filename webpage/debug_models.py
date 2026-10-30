
import sys
import os
import torch
from torchvision import models

# Setup paths like app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

try:
    from models import FastScreenNet, ForensicNet
    print("Successfully imported models.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Stage 1
try:
    print("Initializing FastScreenNet...")
    model_s1 = FastScreenNet().to(DEVICE)
    print("FastScreenNet initialized and moved to device.")
    
    s1_path = os.path.join(PARENT_DIR, "weights/stage1_best.pth")
    if os.path.exists(s1_path):
        print(f"Loading weights from {s1_path}")
        model_s1.load_state_dict(torch.load(s1_path, map_location=DEVICE))
        print("Stage 1 weights loaded.")
    else:
        print(f"Warning: Stage 1 weights not found at {s1_path}")
except Exception as e:
    print(f"Error loading Stage 1: {e}")
    import traceback
    traceback.print_exc()

# Stage 2
try:
    print("Initializing ForensicNet...")
    model_s2 = ForensicNet().to(DEVICE)
    print("ForensicNet initialized and moved to device.")
    
    s2_path = os.path.join(PARENT_DIR, "weights_1/stage2_best.pth")
    if os.path.exists(s2_path):
        print(f"Loading weights from {s2_path}")
        model_s2.load_state_dict(torch.load(s2_path, map_location=DEVICE))
        print("Stage 2 weights loaded.")
    else:
        print(f"Warning: Stage 2 weights not found at {s2_path}")
except Exception as e:
    print(f"Error loading Stage 2: {e}")
    import traceback
    traceback.print_exc()
