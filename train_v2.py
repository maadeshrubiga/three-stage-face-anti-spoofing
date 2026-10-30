import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import glob
import random
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# PART 1: DATASET LOADER (from dataset_loader.py)
# ==========================================

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
            
            # Convert to list of tuples 
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
    
        # --- FIX 1: Windows → Linux slashes ---
        path = path.replace("\\", "/")
    
        # --- FIX 2: Remove duplicated 'dataset/' prefix ---
        if path.startswith("dataset/"):
            path = path[len("dataset/"):]
    
        # --- FIX 3: Join with Kaggle root ---
        if not path.startswith("/kaggle"):
            path = os.path.join(self.root_dir, path)
    
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Unreadable image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Skip bad samples safely
            return self.__getitem__(random.randint(0, len(self.data) - 1))
    
        # Synthetic spoof augmentation
        if self.mode == 'train' and label == 1 and random.random() < 0.3:
            img = self.generate_synthetic_print(img)
            label = 0
    
        return self.transform(img), torch.tensor(label, dtype=torch.float32)



# ==========================================
# PART 2: MODELS (from models.py)
# ==========================================

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

# --- STAGE 2: Forensic Expert (EfficientNet + FFT) ---


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
            nn.Linear(512, 1)   # NO sigmoid here
        )

    def forward(self, x):
        f_rgb = self.rgb_backbone(x)
        x_fft = compute_fft(x)  # (B,1,224,224)
        f_fft = self.fft_branch(x_fft)
        fused = torch.cat([f_rgb, f_fft], dim=1)
        return self.head(fused)


# ==========================================
# PART 3: TRAINING LOOP (from train.py)
# ==========================================

# CONFIGURATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS = 1
LR = 0.0001
MODEL_TYPE = "STAGE2" # Options: "STAGE1" or "STAGE2"
VAL_SPLIT = 0.1
RESULTS_DIR = "results"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{MODEL_TYPE} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()
    
    # 2. Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title(f'{MODEL_TYPE} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_curve.png'))
    plt.close()

def save_confusion_matrix(cm, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Live'])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def train():
    print(f"Training {MODEL_TYPE} on {DEVICE}...")
    ensure_dir(os.path.join(RESULTS_DIR, 'plots'))
    ensure_dir("weights")
    
    img_size = 224 if MODEL_TYPE == "STAGE2" else 128
    
    # 1. Setup Dataset (Train Mode and Val Mode)
    # Use the local project dataset path instead of Kaggle-specific hardcoded paths
    base = r"C:\Users\avmaa\Downloads\Three-Stage-Face-Anti-Spoofing--main\Three-Stage-Face-Anti-Spoofing--main"

    train_dataset = AntiSpoofDataset(
        root_dir=base + r"\dataset",
        split_file=base + r"\dataset\train_split.json",
        img_size=img_size,
        mode='train'
    )
    val_dataset = AntiSpoofDataset(
        root_dir=base + r"\dataset",
        split_file=base + r"\dataset\val_split.json",
        img_size=img_size,
        mode='val'
    )

    print(f"Dataset Loaded: {len(train_dataset)} Train, {len(val_dataset)} Validation")
    
    # --- Class Imbalance Handling: WeightedRandomSampler ---
    # Extract labels for the training set (dataset is already just the training split)
    train_labels = [label for _, label in train_dataset.data]
    
    # Count classes
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create Sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Use sampler in DataLoader (shuffle must be False when using sampler)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    if MODEL_TYPE == "STAGE1":
        model = FastScreenNet().to(DEVICE)
    else:
        model = ForensicNet().to(DEVICE)
        
    optimizer = optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # History tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # --- VALIDATE ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * imgs.size(0)
                predicted = (outputs > 0.5).float()
                
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={epoch_loss:.4f} Acc={epoch_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"weights/{MODEL_TYPE.lower()}_best.pth")
            
        # Save Last Model
        torch.save(model.state_dict(), f"weights/{MODEL_TYPE.lower()}_last.pth")

    # 4. Final Evaluation & Reporting
    print("Generating Reports...")
    plot_history(history, os.path.join(RESULTS_DIR, 'plots'))
    
    # Calculate detailed metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    # Check if recall_score, f1_score work; 
    # Note: sklearn metrics might handle 1-class cases carefully, but here we expect binary.
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    save_confusion_matrix(conf_matrix, os.path.join(RESULTS_DIR, 'plots'))
    
    # Save Metrics
    metrics = {
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': best_val_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    with open(os.path.join(RESULTS_DIR, 'performance_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(RESULTS_DIR, 'performance_summary.txt'), 'w') as f:
        f.write(f"--- {MODEL_TYPE} Training Summary ---\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\n(See plots in results/plots/)\n")

    print(f"Training Complete. Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    train()
