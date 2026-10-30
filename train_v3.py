import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import glob
import random
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# PART 1: DATALOADER (Updated for Multi-Dataset & Balancing)
# ==========================================

class AntiSpoofDataset(Dataset):
    def __init__(self, root_dirs, split_file=None, img_size=224, mode='train'):
        """
        root_dirs: List of paths, e.g. ['/kaggle/input/ds1', '/kaggle/input/ds2']
        """
        # Allow single string or list
        if isinstance(root_dirs, str):
            self.root_dirs = [root_dirs]
        else:
            self.root_dirs = root_dirs
            
        self.mode = mode
        self.img_size = img_size
        self.data = []
        
        # 1. Load from Split File (Optional - mostly for fixed recurrence)
        if split_file and os.path.exists(split_file):
            print(f"Loading dataset from {split_file}...")
            with open(split_file, 'r') as f:
                data_list = json.load(f)
            for item in data_list:
                self.data.append((item['path'], item['label']))
            print(f"Loaded {len(self.data)} samples from split file.")
            
        else:
            # 2. Scanning Mode (Multi-Dataset Support with Deep Search)
            print(f"Scanning directories: {self.root_dirs}")
            
            for root in self.root_dirs:
                if not os.path.exists(root):
                    print(f"  [WARN] Path not found: {root}")
                    continue
                
                print(f"  Walking tree: {root}...")
                count_live_local = 0
                count_spoof_local = 0
                
                # Recursive Walk to find ANY folder named 'live' or 'spoof'
                for dirpath, dirnames, filenames in os.walk(root):
                    folder_name = os.path.basename(dirpath).lower()
                    
                    # Check folder name
                    label = None
                    if folder_name == 'live' or folder_name == 'real':
                        label = 1
                    elif 'spoof' in folder_name or folder_name == 'attack' or folder_name == 'fake':
                        label = 0
                        
                    # If valid label folder found, add all images inside
                    if label is not None:
                        for f in filenames:
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                                full_path = os.path.join(dirpath, f)
                                self.data.append((full_path, label))
                                if label == 1: count_live_local += 1
                                else: count_spoof_local += 1
                                
                print(f"    -> Found {count_live_local} Live, {count_spoof_local} Spoof in {root}")

            print(f"Total Loaded: {len(self.data)} images.")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_synthetic_print(self, img_bgr):
        """ Creates a fake 'Print Attack' from a real face """
        noise = np.random.normal(0, 25, img_bgr.shape).astype(np.uint8)
        img_noisy = cv2.add(img_bgr, noise)
        gamma = random.uniform(0.5, 0.9)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img_noisy, table)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # Windows/Linux Path fixes
        path = path.replace("\\", "/")
        if path.startswith("dataset/"): # Fix relative path issues if any
            # Note: In Multi-Dataset mode, paths are usually absolute from glob, so this might be skipped
            pass 

        try:
            img = cv2.imread(path)
            if img is None: raise ValueError(f"Unreadable: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return self.__getitem__(random.randint(0, len(self.data)-1))

        # Synthetic Augmentation (Optional backup)
        if self.mode == 'train' and label == 1 and random.random() < 0.1:
            img = self.generate_synthetic_print(img)
            label = 0 

        return self.transform(img), torch.tensor(label, dtype=torch.float32)

# ==========================================
# PART 2: MODELS (Original from train_v2.py)
# ==========================================

def compute_fft(x):
    x_gray = torch.mean(x, dim=1, keepdim=True)
    fft = torch.fft.fft2(x_gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = 20 * torch.log(torch.abs(fft_shift) + 1e-6)
    return magnitude

class FastScreenNet(nn.Module):
    def __init__(self):
        super(FastScreenNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        self.backbone.classifier[3] = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))

class ForensicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = models.efficientnet_v2_m(weights="DEFAULT")
        self.rgb_backbone.classifier = nn.Identity()
        
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1280 + 128, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        f_rgb = self.rgb_backbone(x)
        x_fft = compute_fft(x)
        f_fft = self.fft_branch(x_fft)
        fused = torch.cat([f_rgb, f_fft], dim=1)
        return self.head(fused)

# ==========================================
# PART 3: TRAINING LOOP (Updated)
# ==========================================

# CONFIG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001
MODEL_TYPE = "STAGE2" 
VAL_SPLIT = 0.1
RESULTS_DIR = "results_v3"

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{MODEL_TYPE} Loss Curve')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png')); plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title(f'{MODEL_TYPE} Accuracy Curve')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_curve.png')); plt.close()

def save_confusion_matrix(cm, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Live'])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png')); plt.close()

def train():
    print(f"Training {MODEL_TYPE} on {DEVICE}...")
    ensure_dir(os.path.join(RESULTS_DIR, 'plots'))
    ensure_dir("weights")
    
    img_size = 224 if MODEL_TYPE == "STAGE2" else 128
    
    # --- DATASET SETUP (MULTI-ROOT) ---
    # Define your dataset roots here
    dataset_roots = [
        r'c:/Users/manos/Videos/Final Year/dataset'
    ]
    
    # 1. Load Everything
    full_dataset = AntiSpoofDataset(root_dirs=dataset_roots, img_size=img_size, mode='train')
    
    # 2. Split (Train/Val)
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Total Samples: {len(full_dataset)}")
    print(f"Splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # --- CLASS IMBALANCE HANDLING (WeightedRandomSampler) ---
    print("Calculating Class Weights for Balanced Sampling...")
    # Get labels for the training subset
    train_indices = train_dataset.indices
    train_labels = [full_dataset.data[i][1] for i in train_indices]
    
    # Count classes
    class_counts = np.bincount(train_labels)
    # Avoid zero division
    reciprocal_weights = []
    for c in class_counts:
        reciprocal_weights.append(1.0 / c if c > 0 else 0)
    
    # Assign weight to each sample
    sample_weights = [reciprocal_weights[label] for label in train_labels]
    
    if len(class_counts) > 1:
        print(f"  Class Counts (Train): Spoof={class_counts[0]}, Live={class_counts[1]}")
    
    # Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Setup Model
    if MODEL_TYPE == "STAGE1":
        model = FastScreenNet().to(DEVICE)
    else:
        model = ForensicNet().to(DEVICE)
        
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0; val_correct = 0; val_total = 0
        all_preds = []; all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * imgs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={epoch_loss:.4f} Acc={epoch_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"weights/{MODEL_TYPE.lower()}_v3_best.pth")
            
        torch.save(model.state_dict(), f"weights/{MODEL_TYPE.lower()}_v3_last.pth")

    # Final Evaluation
    print("Generating Reports...")
    plot_history(history, os.path.join(RESULTS_DIR, 'plots'))
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(conf_matrix, os.path.join(RESULTS_DIR, 'plots'))
    
    with open(os.path.join(RESULTS_DIR, 'performance_summary.txt'), 'w') as f:
        f.write(f"--- {MODEL_TYPE} V3 Training Summary ---\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")
        
    print(f"Training Complete. Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    train()
