import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import random

import json

class AntiSpoofDataset(Dataset):
    def __init__(self, root_dir, split_file=None, img_size=224, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.data = []
        
        if split_file and os.path.exists(split_file):
            print(f"Loading dataset from {split_file}...")
            with open(split_file, 'r') as f:
                data_list = json.load(f)
            
            # Convert to list of tuples and ensure proper path
            # The JSON stores relative paths from dataset root possibly?
            # data_splitter stored e.g. "dataset/live/xx.jpg" relative to CWD
            # So we can just use them directly if running from CWD, or verify.
            # In data_splitter we stored 'dataset/live/file.jpg'.
            # root_dir is usually passed as './dataset' or 'dataset'.
            # Let's assume paths in JSON are usable as is or relative to CWD.
            for item in data_list:
                self.data.append((item['path'], item['label']))
                
            print(f"Loaded {len(self.data)} samples from split file.")
            
        else:
            # Fallback to scanning logic
            print("Split file not found or not provided. Scanning directories...")
            
            # Custom glob function to support multiple extensions recursively
            def get_image_files(base_path):
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(base_path, '**', ext), recursive=True))
                return sorted(files)

            # LOAD LIVE IMAGES (Label: 1)
            live_files = get_image_files(os.path.join(root_dir, 'live'))
            for f in live_files:
                self.data.append((f, 1))
                
            # LOAD SPOOF IMAGES (Label: 0)
            spoof_files = get_image_files(os.path.join(root_dir, 'spoof'))
            for f in spoof_files:
                self.data.append((f, 0))
                
            print(f"Loaded {len(live_files)} Live and {len(spoof_files)} Spoof images.")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_synthetic_print(self, img_bgr):
        """ Creates a fake 'Print Attack' from a real face """
        # Add Noise
        noise = np.random.normal(0, 25, img_bgr.shape).astype(np.uint8)
        img_noisy = cv2.add(img_bgr, noise)
        # Gamma wash-out
        gamma = random.uniform(0.5, 0.9)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img_noisy, table)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = cv2.imread(path)
            if img is None: raise Exception()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return self.__getitem__(random.randint(0, len(self.data)-1))

        # SYNTHETIC AUGMENTATION
        # If image is LIVE, 30% chance to turn it into a SYNTHETIC SPOOF
        if self.mode == 'train' and label == 1 and random.random() < 0.3:
            img = self.generate_synthetic_print(img)
            label = 0 # Flip label to Spoof

        return self.transform(img), torch.tensor(label, dtype=torch.float32)