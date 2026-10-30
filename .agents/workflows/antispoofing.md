---
description: AntiSpoofing System Project Workflow
---

# AntiSpoofing System Workflow

This workflow documents the project structure and the multi-stage inference logic for the AntiSpoofing system.

## Project Structure

The project follows this directory and file structure:

```text
AntiSpoofing_System/
│
├── dataset/                     # [MUST BE CREATED MANUALLY]
│   ├── live/                    # UTKFace, MeGlass, FEI images
│   └── spoof/                   # Roboflow, HF Monitors images
│
├── weights/                     # Folder to save trained models
│
├── models.py                    # Neural Network Definitions
├── dataset_loader.py            # Data Processing & Synthetic Generation
├── train.py                     # Training Script
├── inference.py                 # Real-time Webcam Script
└── requirements.txt             # Dependencies
```

Additional Resource: [Kaggle Dataset](https://www.kaggle.com/datasets/trainingdatapro/antispoofing-replay-dataset)

## Inference Logic (Multi-Stage)

The system processes input streams through three main stages:

### 1. Input & Pre-processing
- **Source**: Webcam / IP Camera (RGB Stream, 30 FPS).
- **Processing**:
    - Resize to 224×224.
    - Face Detection using Haar Cascade.
    - ROI (Region of Interest) Extraction.
    - Buffer management: Store last 16 frames.

### 2. Stage 1: Fast Screening (The Gatekeeper)
- **Script**: `models.py` (`FastScreenNet`)
- **Action**: Single frame check (Resized to 128×128).
- **Logic**:
    - Rejects Motion Blur: Laplacian Variance < 50 → FAIL.
    - `MobileNetV3` check.
- **Pass Threshold**: Score > 95% → Instant Allow.
- **Ambiguous**: If score ≤ 95%, proceed to Stage 2.

### 3. Stage 2: Forensic Expert
- **Script**: `models.py` (`ForensicNet`)
- **Action**: FFT-based spatial and frequency analysis (224×224 + FFT).
- **Architecture**:
    - **Branch A**: `EfficientNet` for spatial texture analysis.
    - **Branch B**: `FFT Conv2D` for frequency Moiré check.
- **Decision**: Is Live? Score > 50% → Proceed to Stage 3. Else → FAIL (Attack).

### 4. Stage 3: Temporal Check
- **Logic**: Motion check over the 16-frame buffer.
- **Requirement**: Motion > 0 to reject static photos (e.g., printed attacks).

### 5. Final Decision
- Results are displayed with a colored bounding box:
    - **Green**: Live
    - **Red**: Spoof / Attack
