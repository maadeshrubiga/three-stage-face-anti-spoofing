# Attack Coverage & Threat Model

This document outlines the **Presentation Attack Detection (PAD)** capabilities of the system. The model is designed to detect and mitigate various forms of biometric spoofing attacks (Presentation Attacks) as defined in **ISO/IEC 30107**.

## 1. Attack Vectors Covered

The system implements a multi-stage defense strategy to cover the following attack types:

### A. Print Attacks (Level 1 Threat)
*   **Description**: Attacker presents a printed photograph of the authorized user.
    *   *Sub-types*: Matte Paper, Glossy Photos, Laser Cuts (Eye holes).
*   **Coverage Status**: **HIGH**
*   **Defense Mechanism**:
    *   **Texture Analysis (Stage 2)**: The `EfficientNet` backbone analyzes micro-texture patterns. Printed paper lacks the subsurface scattering and skin pores of real skin, identifying it as a flat surface.
    *   **Synthetic Augmentation**: The training pipeline (`dataset_loader.py`) specifically simulates this by adding noise and gamma washout to real faces during training.

### B. Replay / Video Attacks (Level 2 Threat)
*   **Description**: Attacker plays a video or displays a static photo on a digital screen (Smartphone, Tablet, Laptop, High-Res Monitor).
*   **Coverage Status**: **HIGH**
*   **Defense Mechanism**:
    *   **Frequency Analysis (Stage 2)**: The **FFT (Fast Fourier Transform)** branch specifically targets **Moiré patterns**—aliasing artifacts that occur when a camera captures a pixel grid of a screen.
    *   **Motion Blur Screening (Stage 1)**: Replay attacks often involve holding a device; low-quality motion or unnatural stillness is flagged.

### C. 3D Mask & Static Attacks
*   **Description**: Attacker wears a mask or holds a 3D prop.
*   **Coverage Status**: **MEDIUM** (Temporal Dependent)
*   **Defense Mechanism**:
    *   **Temporal Logic (Stage 3)**: While not a deep learning model, the system requires natural motion (blinking/micro-expressions). A strictly static mask or photo will be rejected by the temporal buffer check if it lacks "liveness" variance over frames.

## 2. Defense Matrix

| Attack Type | Primary Defense | Secondary Defense | Verification |
| :--- | :--- | :--- | :--- |
| **Printed Photo** | **Texture Analysis** (RGB Branch) | Gaussian Noise Analysis | Train Metric (Val) |
| **Digital Screen** | **Moiré Detection** (FFT Branch) | Color Distortion | Test on `HF Monitors` |
| **Video Replay** | **Moiré Detection** (FFT Branch) | Temporal Consistency | Test on `Replay` Dataset |
| **Blurred/Low Q** | **Laplacian Variance** (Stage 1) | MobileNet Assessment | Stage 1 Threshold |

## 3. Evaluation Metrics by Attack Type

To ensure robust coverage, the system tracks specific metrics:

*   **APCER (Attack Presentation Classification Error Rate)**: % of attacks incorrectly classified as Live.
*   **BPCER (Bona Fide Presentation Classification Error Rate)**: % of real users incorrectly classified as Spoof.
*   **HTER (Half Total Error Rate)**: Average of APCER and BPCER.

## 4. Synthetic Stress Testing
The system includes a **Synthetic Attack Generator** in the training loop (`dataset_loader.py`) that randomly converts live samples into "Print Attacks" by applying:
*   **Gaussian Noise**: Simulating print grain.
*   **Gamma Washout**: Simulating loss of dynamic range in printing.

This ensures the model effectively learns the boundary between **Real Skin** and **Artificial Surfaces**.
