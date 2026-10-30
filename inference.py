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

# --- VIDEO SOURCE ---
VIDEO_SOURCE = "replay_video.mp4"    # Video file
# VIDEO_SOURCE = 0  # Webcam

# --- OUTPUT FILES ---
OUTPUT_VIDEO = "output_liveness.mp4"
OUTPUT_LOG = "liveness_results.txt"

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

def check_liveness(frame):
    """Returns: (status, color, text, s1_score, s2_score, detail)"""
    # --- STAGE 1: Fast Check ---
    # 1. Laplacian Blur Check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return "FAIL", (0, 0, 255), "Too Blurry", 0.0, 0.0, f"Blur={laplacian_var:.1f}"

    # 2. MobileNet Check
    img_s1 = trans_s1(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score_s1 = model_s1(img_s1).item()

    if score_s1 > THRESHOLD_S1:
        return "PASS", (0, 255, 0), f"S1 Clear ({score_s1:.2f})", score_s1, 0.0, "Stage1 PASS"
    
    if score_s1 < 0.10:
        return "FAIL", (0, 0, 255), "S1 Spoof", score_s1, 0.0, "Stage1 Spoof"

    # --- STAGE 2: Deep Forensic Check ---
    img_s2 = trans_s2(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit_s2 = model_s2(img_s2)
        score_s2 = torch.sigmoid(logit_s2).item()
        
    if score_s2 > THRESHOLD_S2:
        return "PASS", (0, 255, 0), f"S2 Verified ({score_s2:.2f})", score_s1, score_s2, "Stage2 PASS"
    else:
        return "FAIL", (0, 0, 255), f"S2 Spoof ({score_s2:.2f})", score_s1, score_s2, "Stage2 Spoof"

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
stats = {"total_frames": 0, "faces_detected": 0, "pass_count": 0, "fail_count": 0, "no_face_count": 0}

print("\nProcessing video...")
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

    if len(faces) == 0:
        stats["no_face_count"] += 1
        cv2.putText(frame, "No Face Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
            
            # Run Pipeline
            status, color, text, s1_score, s2_score, detail = check_liveness(face_roi)
            
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
                "detail": detail
            })
            
            # Draw Liveness Results (thicker box on top)
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, 3)  # Thicker colored box for result
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame_results.append(frame_result)

    # Add frame info
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write to output video
    out.write(frame)

    # Resize for display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow('Liveness Detection System', display_frame)
    
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

# --- WRITE RESULTS TO FILE ---
with open(OUTPUT_LOG, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("         LIVENESS DETECTION RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Video Source: {VIDEO_SOURCE}\n")
    f.write(f"Video Resolution: {frame_width}x{frame_height} @ {fps}fps\n")
    f.write("-" * 60 + "\n\n")
    
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
    
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"FINAL VERDICT: {final_verdict}\n")
    f.write(f"CONFIDENCE: {confidence:.1f}%\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("FRAME-BY-FRAME RESULTS:\n")
    f.write("-" * 60 + "\n")
    
    for result in frame_results:
        if result["status"] == "NO_FACE":
            f.write(f"Frame {result['frame']:4d}: No face detected\n")
        else:
            for face in result["faces"]:
                f.write(f"Frame {result['frame']:4d}: Face {face['face_id']} - "
                       f"{face['status']} | S1={face['s1_score']:.3f} S2={face['s2_score']:.3f} | {face['detail']}\n")

print("\n" + "=" * 50)
print("PROCESSING COMPLETE")
print("=" * 50)
print(f"Total frames: {stats['total_frames']}")
print(f"Faces detected: {stats['faces_detected']}")
print(f"PASS: {stats['pass_count']} | FAIL: {stats['fail_count']}")
print(f"\nFINAL VERDICT: {final_verdict} ({confidence:.1f}%)")
print(f"\nOutput video: {OUTPUT_VIDEO}")
print(f"Results log: {OUTPUT_LOG}")