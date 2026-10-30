import cv2
import torch
import numpy as np
from torchvision import transforms
from models import FastScreenNet, ForensicNet
from collections import deque
from datetime import datetime

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD_S1 = 0.95  # High confidence needed to pass Stage 1
THRESHOLD_S2 = 0.50  # Stage 2 threshold

# Enhanced thresholds for combination techniques
LAPLACIAN_THRESHOLD = 50
EDGE_STRENGTH_THRESHOLD = 30
SPECTRAL_POWER_THRESHOLD = 0.15

# --- VIDEO SOURCE ---
VIDEO_SOURCE = "replay_video.mp4"    # Video file
# VIDEO_SOURCE = 0  # Webcam

# --- OUTPUT FILES ---
OUTPUT_VIDEO = "output_liveness_v2.mp4"
OUTPUT_LOG = "liveness_results_v2.txt"

# --- LOAD MODELS ---
print("Loading models...")
# Stage 1
model_s1 = FastScreenNet().to(DEVICE)
try:
    model_s1.load_state_dict(torch.load("weights/stage1_best.pth", map_location=DEVICE))
    print("Stage 1 model loaded successfully.")
except Exception as e:
    print(f"WARNING: Stage 1 weights not found. Using random weights. Error: {e}")
model_s1.eval()

# Stage 2
model_s2 = ForensicNet().to(DEVICE)
try:
    model_s2.load_state_dict(torch.load("weights_1/stage2_best.pth", map_location=DEVICE))
    print("Stage 2 model loaded successfully.")
except Exception as e:
    print(f"WARNING: Stage 2 weights not found. Using random weights. Error: {e}")
model_s2.eval()

# --- PREPROCESSING ---
trans_s1 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_s2 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ENHANCED FEATURE EXTRACTION FUNCTIONS ---

def compute_multiscale_blur(frame):
    """
    ENHANCED: Multi-scale Laplacian blur detection
    Combines blur variance across multiple scales for more robust detection
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale blur detection
    scales = [1.0, 0.75, 0.5]
    blur_scores = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            resized = gray
            
        laplacian = cv2.Laplacian(resized, cv2.CV_64F)
        variance = laplacian.var()
        blur_scores.append(variance)
    
    # Combine scores: weighted average (favor original scale)
    weights = [0.5, 0.3, 0.2]
    combined_blur = sum(s * w for s, w in zip(blur_scores, weights))
    
    return combined_blur, blur_scores

def compute_edge_strength(frame):
    """
    ENHANCED: Edge strength analysis using Sobel operator
    Combination with Laplacian for better texture analysis
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_strength = np.mean(edge_magnitude)
    
    return edge_strength

def compute_fft_analysis(frame):
    """
    ENHANCED: FFT-based frequency domain analysis
    Multiple combination techniques:
    1. Spectral power distribution
    2. High-frequency component analysis
    3. Moiré pattern detection
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Normalize
    magnitude_spectrum = magnitude_spectrum / (magnitude_spectrum.max() + 1e-8)
    
    # 1. Spectral Power Distribution
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Define frequency bands
    low_freq_radius = min(h, w) // 8
    high_freq_radius = min(h, w) // 3
    
    # Create masks
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    low_freq_mask = distance <= low_freq_radius
    high_freq_mask = (distance > low_freq_radius) & (distance <= high_freq_radius)
    
    # Compute power in each band
    low_freq_power = np.sum(magnitude_spectrum[low_freq_mask])
    high_freq_power = np.sum(magnitude_spectrum[high_freq_mask])
    total_power = np.sum(magnitude_spectrum)
    
    # Spectral power ratio (high/total)
    spectral_ratio = high_freq_power / (total_power + 1e-8)
    
    # 2. Moiré pattern detection (periodic peaks in high frequency)
    high_freq_region = magnitude_spectrum[high_freq_mask]
    moire_score = np.std(high_freq_region) if len(high_freq_region) > 0 else 0
    
    # 3. Overall frequency complexity
    freq_entropy = -np.sum(magnitude_spectrum * np.log(magnitude_spectrum + 1e-8))
    
    return {
        'spectral_ratio': spectral_ratio,
        'moire_score': moire_score,
        'freq_entropy': freq_entropy,
        'low_freq_power': low_freq_power,
        'high_freq_power': high_freq_power
    }

def compute_combined_quality_score(frame):
    """
    COMBINATION TECHNIQUE: Weighted ensemble of multiple quality metrics
    Returns a comprehensive quality score and individual components
    """
    # 1. Multi-scale blur
    blur_combined, blur_scales = compute_multiscale_blur(frame)
    
    # 2. Edge strength
    edge_strength = compute_edge_strength(frame)
    
    # 3. FFT analysis
    fft_features = compute_fft_analysis(frame)
    
    # Normalize scores to [0, 1]
    blur_score = min(blur_combined / 200.0, 1.0)  # Higher is better
    edge_score = min(edge_strength / 100.0, 1.0)  # Higher is better
    spectral_score = min(fft_features['spectral_ratio'] / 0.3, 1.0)  # Higher is better
    
    # Combined quality score (weighted average)
    # Weights: blur=0.4, edge=0.3, spectral=0.3
    quality_score = (0.4 * blur_score + 0.3 * edge_score + 0.3 * spectral_score)
    
    return {
        'quality_score': quality_score,
        'blur_combined': blur_combined,
        'blur_scales': blur_scales,
        'edge_strength': edge_strength,
        'fft_features': fft_features
    }

def check_liveness_enhanced(frame):
    """
    ENHANCED VERSION: Returns: (status, color, text, s1_score, s2_score, detail, quality_metrics)
    Incorporates combination techniques for improved accuracy
    """
    # --- ENHANCED QUALITY CHECKS ---
    quality_metrics = compute_combined_quality_score(frame)
    
    # 1. Multi-scale Laplacian Blur Check with Combination
    if quality_metrics['blur_combined'] < LAPLACIAN_THRESHOLD:
        detail = f"Blur={quality_metrics['blur_combined']:.1f}, Edge={quality_metrics['edge_strength']:.1f}"
        return "FAIL", (0, 0, 255), "Too Blurry", 0.0, 0.0, detail, quality_metrics
    
    # 2. Edge Strength Check (additional validation)
    if quality_metrics['edge_strength'] < EDGE_STRENGTH_THRESHOLD:
        detail = f"WeakEdge={quality_metrics['edge_strength']:.1f}"
        return "FAIL", (0, 0, 255), "Weak Texture", 0.0, 0.0, detail, quality_metrics
    
    # 3. FFT Spectral Analysis Check
    spectral_ratio = quality_metrics['fft_features']['spectral_ratio']
    moire_score = quality_metrics['fft_features']['moire_score']
    
    # Check for screen/print artifacts (abnormal spectral patterns)
    if spectral_ratio > 0.4 or moire_score > 0.5:
        detail = f"FFT_Anomaly: SR={spectral_ratio:.3f}, Moire={moire_score:.3f}"
        return "FAIL", (0, 0, 255), "Screen Detected", 0.0, 0.0, detail, quality_metrics
    
    # --- STAGE 1: Fast Check with Quality-Aware Threshold ---
    img_s1 = trans_s1(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score_s1 = model_s1(img_s1).item()
    
    # Adjust threshold based on quality score
    adjusted_threshold_s1 = THRESHOLD_S1 * (0.8 + 0.2 * quality_metrics['quality_score'])
    
    if score_s1 > adjusted_threshold_s1:
        detail = f"S1={score_s1:.2f}, Q={quality_metrics['quality_score']:.2f}"
        return "PASS", (0, 255, 0), f"S1 Clear ({score_s1:.2f})", score_s1, 0.0, detail, quality_metrics
    
    if score_s1 < 0.10:
        detail = f"S1={score_s1:.2f}, Q={quality_metrics['quality_score']:.2f}"
        return "FAIL", (0, 0, 255), "S1 Spoof", score_s1, 0.0, detail, quality_metrics

    # --- STAGE 2: Deep Forensic Check with FFT Enhancement ---
    img_s2 = trans_s2(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit_s2 = model_s2(img_s2)
        score_s2 = torch.sigmoid(logit_s2).item()
    
    # Combine S2 score with FFT features for ensemble decision
    # Penalize if spectral anomalies detected
    spectral_penalty = max(0, (spectral_ratio - SPECTRAL_POWER_THRESHOLD) * 2)
    adjusted_score_s2 = score_s2 * (1 - spectral_penalty)
    
    if adjusted_score_s2 > THRESHOLD_S2:
        detail = f"S2={score_s2:.2f}, Adj={adjusted_score_s2:.2f}, Q={quality_metrics['quality_score']:.2f}"
        return "PASS", (0, 255, 0), f"S2 Verified ({adjusted_score_s2:.2f})", score_s1, adjusted_score_s2, detail, quality_metrics
    else:
        detail = f"S2={score_s2:.2f}, Adj={adjusted_score_s2:.2f}, FFT_Flag"
        return "FAIL", (0, 0, 255), f"S2 Spoof ({adjusted_score_s2:.2f})", score_s1, adjusted_score_s2, detail, quality_metrics

# --- MAIN LOOP ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"ERROR: Could not open video source: {VIDEO_SOURCE}")
    exit(1)
print(f"Video source opened: {'Webcam' if VIDEO_SOURCE == 0 else VIDEO_SOURCE}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {frame_width}x{frame_height} @ {fps}fps, Total frames: {total_frames}")

# --- OUTPUT VIDEO SETUP ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
print(f"Saving output video to: {OUTPUT_VIDEO}")
print(f"Saving results log to: {OUTPUT_LOG}")

# Display window size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Load face cascades (try multiple for better detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# --- RESULTS TRACKING ---
frame_results = []
stats = {
    "total_frames": 0, 
    "faces_detected": 0, 
    "pass_count": 0, 
    "fail_count": 0, 
    "no_face_count": 0,
    "blur_fails": 0,
    "edge_fails": 0,
    "fft_fails": 0,
    "s1_fails": 0,
    "s2_fails": 0
}

print("\nProcessing video with ENHANCED combination techniques...")
print("Features: Multi-scale Blur + Edge Strength + FFT Analysis + Ensemble Voting")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame_count += 1
    stats["total_frames"] = frame_count

    # Detect faces with multiple cascades for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for detection
    
    # Try default cascade first
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    
    # If no faces found, try alt cascade with different params
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
    
    frame_result = {"frame": frame_count, "faces": [], "status": "NO_FACE"}

    # Show face count on frame
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "ENHANCED v2", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if len(faces) == 0:
        stats["no_face_count"] += 1
        cv2.putText(frame, "No Face Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        for i, (x, y, w, h) in enumerate(faces):
            stats["faces_detected"] += 1
            
            # ALWAYS draw Haar cascade detection box first (yellow/cyan)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)  # Yellow thin box for detection
            cv2.putText(frame, f"Face {i+1}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Expand ROI slightly for better detection
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_width, x + w + pad)
            y2 = min(frame_height, y + h + pad)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # Run ENHANCED Pipeline
            status, color, text, s1_score, s2_score, detail, quality_metrics = check_liveness_enhanced(face_roi)
            
            # Track failure reasons
            if "Blur" in detail:
                stats["blur_fails"] += 1
            if "Edge" in detail:
                stats["edge_fails"] += 1
            if "FFT" in detail or "Screen" in text:
                stats["fft_fails"] += 1
            if "S1" in text:
                stats["s1_fails"] += 1
            if "S2" in text:
                stats["s2_fails"] += 1
            
            if status == "PASS":
                stats["pass_count"] += 1
                frame_result["status"] = "PASS"
            else:
                stats["fail_count"] += 1
                frame_result["status"] = "FAIL"
            
            frame_result["faces"].append({
                "face_id": i+1,
                "bbox": (x, y, w, h),
                "status": status,
                "s1_score": s1_score,
                "s2_score": s2_score,
                "detail": detail,
                "quality_score": quality_metrics['quality_score'],
                "blur_combined": quality_metrics['blur_combined'],
                "edge_strength": quality_metrics['edge_strength'],
                "spectral_ratio": quality_metrics['fft_features']['spectral_ratio']
            })
            
            # Draw Liveness Results (thicker box on top)
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, 3)  # Thicker colored box for result
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show quality score
            quality_text = f"Q:{quality_metrics['quality_score']:.2f}"
            cv2.putText(frame, quality_text, (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    frame_results.append(frame_result)

    # Add frame info
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write to output video
    out.write(frame)

    # Resize for display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow('Enhanced Liveness Detection System v2', display_frame)
    
    # Progress update every 50 frames
    if frame_count % 50 == 0:
        print(f"  Processed {frame_count}/{total_frames} frames...")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- CALCULATE FINAL VERDICT ---
if stats["faces_detected"] == 0:
    final_verdict = "UNKNOWN - No faces detected"
    confidence = 0.0
else:
    fail_rate = stats["fail_count"] / stats["faces_detected"] * 100
    pass_rate = stats["pass_count"] / stats["faces_detected"] * 100
    
    if fail_rate >= 70:
        final_verdict = "SPOOF DETECTED"
        confidence = fail_rate
    elif pass_rate >= 70:
        final_verdict = "LIVE PERSON"
        confidence = pass_rate
    else:
        final_verdict = "UNCERTAIN"
        confidence = max(fail_rate, pass_rate)

# --- WRITE ENHANCED RESULTS TO FILE ---
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("         ENHANCED LIVENESS DETECTION RESULTS (v2)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Video Source: {VIDEO_SOURCE}\n")
    f.write(f"Video Resolution: {frame_width}x{frame_height} @ {fps}fps\n")
    f.write(f"Version: v2 - Enhanced with Combination Techniques\n")
    f.write("-" * 70 + "\n\n")
    
    f.write("ENHANCEMENT FEATURES:\n")
    f.write("-" * 30 + "\n")
    f.write("✓ Multi-scale Laplacian Blur Detection\n")
    f.write("✓ Edge Strength Analysis (Sobel)\n")
    f.write("✓ FFT Spectral Power Distribution\n")
    f.write("✓ Moiré Pattern Detection\n")
    f.write("✓ Quality-Aware Adaptive Thresholds\n")
    f.write("✓ Ensemble Voting (Blur + Edge + FFT)\n\n")
    
    f.write("SUMMARY STATISTICS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total Frames Processed: {stats['total_frames']}\n")
    f.write(f"Frames with Faces: {stats['total_frames'] - stats['no_face_count']}\n")
    f.write(f"Frames without Faces: {stats['no_face_count']}\n")
    f.write(f"Total Faces Detected: {stats['faces_detected']}\n")
    f.write(f"PASS Count: {stats['pass_count']}\n")
    f.write(f"FAIL Count: {stats['fail_count']}\n")
    
    if stats['faces_detected'] > 0:
        f.write(f"Pass Rate: {stats['pass_count']/stats['faces_detected']*100:.1f}%\n")
        f.write(f"Fail Rate: {stats['fail_count']/stats['faces_detected']*100:.1f}%\n")
    
    f.write("\nFAILURE BREAKDOWN:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Blur-based Failures: {stats['blur_fails']}\n")
    f.write(f"Edge-based Failures: {stats['edge_fails']}\n")
    f.write(f"FFT-based Failures: {stats['fft_fails']}\n")
    f.write(f"Stage 1 Model Failures: {stats['s1_fails']}\n")
    f.write(f"Stage 2 Model Failures: {stats['s2_fails']}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"FINAL VERDICT: {final_verdict}\n")
    f.write(f"CONFIDENCE: {confidence:.1f}%\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("FRAME-BY-FRAME RESULTS:\n")
    f.write("-" * 70 + "\n")
    
    for result in frame_results:
        if result["status"] == "NO_FACE":
            f.write(f"Frame {result['frame']:4d}: No face detected\n")
        else:
            for face in result["faces"]:
                f.write(f"Frame {result['frame']:4d}: Face {face['face_id']} - "
                       f"{face['status']} | S1={face['s1_score']:.3f} S2={face['s2_score']:.3f} | "
                       f"Q={face['quality_score']:.3f} | Blur={face['blur_combined']:.1f} "
                       f"Edge={face['edge_strength']:.1f} FFT={face['spectral_ratio']:.3f} | "
                       f"{face['detail']}\n")

print("\n" + "=" * 60)
print("ENHANCED PROCESSING COMPLETE (v2)")
print("=" * 60)
print(f"Total frames: {stats['total_frames']}")
print(f"Faces detected: {stats['faces_detected']}")
print(f"PASS: {stats['pass_count']} | FAIL: {stats['fail_count']}")
print(f"\nFailure Breakdown:")
print(f"  Blur: {stats['blur_fails']} | Edge: {stats['edge_fails']} | FFT: {stats['fft_fails']}")
print(f"  Stage1: {stats['s1_fails']} | Stage2: {stats['s2_fails']}")
print(f"\nFINAL VERDICT: {final_verdict} ({confidence:.1f}%)")
print(f"\nOutput video: {OUTPUT_VIDEO}")
print(f"Results log: {OUTPUT_LOG}")
