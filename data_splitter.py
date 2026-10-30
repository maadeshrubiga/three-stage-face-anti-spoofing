import os
import json
import glob
import numpy as np
import random

# CONFIGURATION
DATASET_ROOT = 'dataset'
VAL_SPLIT = 0.2
SEED = 42

def get_image_files(base_path):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(base_path, '**', ext), recursive=True))
    return sorted(files)

def main():
    print(f"Scanning {DATASET_ROOT}...")
    
    # 1. Collect all files
    live_path = os.path.join(DATASET_ROOT, 'live')
    spoof_path = os.path.join(DATASET_ROOT, 'spoof')
    
    if not os.path.exists(live_path) or not os.path.exists(spoof_path):
        print("Error: 'dataset/live' or 'dataset/spoof' folders not found.")
        return

    live_files = get_image_files(live_path)
    spoof_files = get_image_files(spoof_path)
    
    print(f"Found {len(live_files)} Live samples.")
    print(f"Found {len(spoof_files)} Spoof samples.")
    
    # 2. Assign Labels (Live=1, Spoof=0)
    # Store as relative paths to keep JSON cleaner and portable
    all_data = []
    for f in live_files:
        all_data.append({'path': f, 'label': 1})
        
    for f in spoof_files:
        all_data.append({'path': f, 'label': 0})
        
    # 3. Shuffle and Split
    random.seed(SEED)
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * (1 - VAL_SPLIT))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # 4. Save to JSON
    train_json_path = os.path.join(DATASET_ROOT, 'train_split.json')
    val_json_path = os.path.join(DATASET_ROOT, 'val_split.json')
    
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=4)
        
    print(f"Successfully saved splits:")
    print(f"  Train: {len(train_data)} samples -> {train_json_path}")
    print(f"  Val:   {len(val_data)} samples -> {val_json_path}")

if __name__ == "__main__":
    main()
