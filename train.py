import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset_loader import AntiSpoofDataset
from models import FastScreenNet, ForensicNet
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import json

# CONFIGURATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001
MODEL_TYPE = "STAGE1" # Options: "STAGE1" or "STAGE2"
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
    # We instantiate twice to get correct augmentation behavior, relies on deterministic file sorting
    # 1. Setup Dataset (Train Mode and Val Mode)
    # Load from offline splits
    train_dataset = AntiSpoofDataset(root_dir='./dataset', split_file='dataset/train_split.json', img_size=img_size, mode='train')
    val_dataset = AntiSpoofDataset(root_dir='./dataset', split_file='dataset/val_split.json', img_size=img_size, mode='val')
    
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
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
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
    
    # Calculate detailed metrics on Validation set (using last epoch's result or re-running? 
    # We use the accumulated preds from the last epoch which is 'all_preds')
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
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