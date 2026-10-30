import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from models import FastScreenNet, ForensicNet
from datetime import datetime
import gc
import os

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD_S1 = 0.95  # High confidence needed to pass Stage 1
THRESHOLD_S2 = 0.50  # Stage 2 threshold

# Enhanced thresholds for combination techniques
LAPLACIAN_THRESHOLD = 50
EDGE_STRENGTH_THRESHOLD = 30
SPECTRAL_POWER_THRESHOLD = 0.15

# GradCAM Settings
GRADCAM_ALPHA = 0.4  # Overlay transparency
SAVE_GRADCAM_FRAMES = False  # Disabled to save memory

# --- VIDEO SOURCE ---
VIDEO_SOURCE = "replay_video.mp4"    # Video file
# VIDEO_SOURCE = 0  # Webcam

# --- OUTPUT FILES ---
OUTPUT_VIDEO = "output_liveness_v3.mp4"
OUTPUT_LOG = "liveness_results_v3.txt"
GRADCAM_DIR = "gradcam_frames"


def compute_gradcam_s1(model, input_tensor, target_layer):
    """
    Compute GradCAM for Stage 1 model (memory efficient version).
    """
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        # Backward pass
        model.zero_grad()
        output.backward()
        
        if not activations or not gradients:
            return None
        
        # Compute CAM
        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        result = cam.cpu().numpy()
        
    finally:
        # Remove hooks
        handle_fwd.remove()
        handle_bwd.remove()
        # Clear memory
        activations.clear()
        gradients.clear()
        del activations, gradients
    
    return result


def compute_gradcam_s2(model, input_tensor):
    """
    Compute GradCAM for Stage 2 ForensicNet (memory efficient version).
    Returns combined CAM from RGB and FFT branches.
    """
    rgb_activations = []
    rgb_gradients = []
    fft_activations = []
    fft_gradients = []
    
    # Get target layers
    rgb_target = model.rgb_backbone.features[-1]
    fft_target = model.fft_branch[6]  # Conv2d(64, 128, ...)
    
    def rgb_fwd(module, input, output):
        rgb_activations.append(output.detach())
    
    def rgb_bwd(module, grad_input, grad_output):
        rgb_gradients.append(grad_output[0].detach())
    
    def fft_fwd(module, input, output):
        fft_activations.append(output.detach())
    
    def fft_bwd(module, grad_input, grad_output):
        fft_gradients.append(grad_output[0].detach())
    
    # Register hooks
    h1 = rgb_target.register_forward_hook(rgb_fwd)
    h2 = rgb_target.register_full_backward_hook(rgb_bwd)
    h3 = fft_target.register_forward_hook(fft_fwd)
    h4 = fft_target.register_full_backward_hook(fft_bwd)
    
    try:
        model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        # Backward pass
        model.zero_grad()
        output.backward()
        
        # Compute RGB CAM
        rgb_cam = None
        if rgb_activations and rgb_gradients:
            weights = torch.mean(rgb_gradients[0], dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * rgb_activations[0], dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze()
            if cam.max() > 0:
                cam = cam / cam.max()
            rgb_cam = cam.cpu().numpy()
        
        # Compute FFT CAM
        fft_cam = None
        if fft_activations and fft_gradients:
            weights = torch.mean(fft_gradients[0], dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * fft_activations[0], dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze()
            if cam.max() > 0:
                cam = cam / cam.max()
            fft_cam = cam.cpu().numpy()
        
        # Combined CAM
        combined_cam = None
        if rgb_cam is not None:
            combined_cam = rgb_cam.copy()
            if fft_cam is not None:
                # Resize FFT CAM to RGB CAM size
                fft_resized = cv2.resize(fft_cam, (rgb_cam.shape[1], rgb_cam.shape[0]))
                combined_cam = 0.7 * rgb_cam + 0.3 * fft_resized
                combined_cam = combined_cam / (combined_cam.max() + 1e-8)
        
    finally:
        # Remove all hooks
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        # Clear memory
        rgb_activations.clear()
        rgb_gradients.clear()
        fft_activations.clear()
        fft_gradients.clear()
    
    return combined_cam


def overlay_cam_on_image(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay GradCAM heatmap on the original image.
    """
    if cam is None:
        return image
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Convert to uint8 and apply colormap
    cam_uint8 = np.uint8(255 * cam_resized)
    cam_colored = cv2.applyColorMap(cam_uint8, colormap)
    
    # Overlay on original image
    overlayed = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
    
    return overlayed


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

# Get target layer for S1 GradCAM
s1_target_layer = model_s1.backbone.features[-1]
print("GradCAM ready for both stages.")

# --- PREPROCESSING ---
trans_s1 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_s2 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- FEATURE EXTRACTION FUNCTIONS ---

def compute_multiscale_blur(frame):
    """Multi-scale Laplacian blur detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scales = [1.0, 0.75, 0.5]
    blur_scores = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            resized = gray
        laplacian = cv2.Laplacian(resized, cv2.CV_64F)
        blur_scores.append(laplacian.var())
    
    weights = [0.5, 0.3, 0.2]
    combined_blur = sum(s * w for s, w in zip(blur_scores, weights))
    return combined_blur, blur_scores


def compute_edge_strength(frame):
    """Edge strength analysis using Sobel operator"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(edge_magnitude)


def compute_fft_analysis(frame):
    """FFT-based frequency domain analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum = magnitude_spectrum / (magnitude_spectrum.max() + 1e-8)
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    low_freq_radius = min(h, w) // 8
    high_freq_radius = min(h, w) // 3
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    low_freq_mask = distance <= low_freq_radius
    high_freq_mask = (distance > low_freq_radius) & (distance <= high_freq_radius)
    
    low_freq_power = np.sum(magnitude_spectrum[low_freq_mask])
    high_freq_power = np.sum(magnitude_spectrum[high_freq_mask])
    total_power = np.sum(magnitude_spectrum)
    
    spectral_ratio = high_freq_power / (total_power + 1e-8)
    high_freq_region = magnitude_spectrum[high_freq_mask]
    moire_score = np.std(high_freq_region) if len(high_freq_region) > 0 else 0
    
    return {
        'spectral_ratio': spectral_ratio,
        'moire_score': moire_score
    }


def compute_combined_quality_score(frame):
    """Weighted ensemble of multiple quality metrics"""
    blur_combined, blur_scales = compute_multiscale_blur(frame)
    edge_strength = compute_edge_strength(frame)
    fft_features = compute_fft_analysis(frame)
    
    blur_score = min(blur_combined / 200.0, 1.0)
    edge_score = min(edge_strength / 100.0, 1.0)
    spectral_score = min(fft_features['spectral_ratio'] / 0.3, 1.0)
    
    quality_score = (0.4 * blur_score + 0.3 * edge_score + 0.3 * spectral_score)
    
    return {
        'quality_score': quality_score,
        'blur_combined': blur_combined,
        'edge_strength': edge_strength,
        'fft_features': fft_features
    }


def check_liveness_with_gradcam(frame, frame_idx):
    """Enhanced liveness check with GradCAM (memory efficient)"""
    gradcam = None
    
    quality_metrics = compute_combined_quality_score(frame)
    
    # Quality checks
    if quality_metrics['blur_combined'] < LAPLACIAN_THRESHOLD:
        return "FAIL", (0, 0, 255), "Too Blurry", 0.0, 0.0, quality_metrics, gradcam
    
    if quality_metrics['edge_strength'] < EDGE_STRENGTH_THRESHOLD:
        return "FAIL", (0, 0, 255), "Weak Texture", 0.0, 0.0, quality_metrics, gradcam
    
    spectral_ratio = quality_metrics['fft_features']['spectral_ratio']
    moire_score = quality_metrics['fft_features']['moire_score']
    
    if spectral_ratio > 0.4 or moire_score > 0.5:
        return "FAIL", (0, 0, 255), "Screen Detected", 0.0, 0.0, quality_metrics, gradcam
    
    # Stage 1
    img_s1 = trans_s1(frame).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        score_s1 = model_s1(img_s1).item()
    
    adjusted_threshold_s1 = THRESHOLD_S1 * (0.8 + 0.2 * quality_metrics['quality_score'])
    
    if score_s1 > adjusted_threshold_s1:
        # Compute GradCAM for S1 (only every 5 frames to save memory)
        if frame_idx % 5 == 0:
            gradcam = compute_gradcam_s1(model_s1, img_s1, s1_target_layer)
        # Clean up
        del img_s1
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
        return "PASS", (0, 255, 0), f"S1 Clear ({score_s1:.2f})", score_s1, 0.0, quality_metrics, gradcam
    
    if score_s1 < 0.10:
        del img_s1
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
        return "FAIL", (0, 0, 255), "S1 Spoof", score_s1, 0.0, quality_metrics, gradcam

    # Stage 2
    del img_s1
    img_s2 = trans_s2(frame).unsqueeze(0).to(DEVICE)
    
    # Compute GradCAM for S2 (only every 5 frames)
    if frame_idx % 5 == 0:
        gradcam = compute_gradcam_s2(model_s2, img_s2)
    
    with torch.no_grad():
        logit_s2 = model_s2(img_s2)
        score_s2 = torch.sigmoid(logit_s2).item()
    
    # Clean up
    del img_s2
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    
    spectral_penalty = max(0, (spectral_ratio - SPECTRAL_POWER_THRESHOLD) * 2)
    adjusted_score_s2 = score_s2 * (1 - spectral_penalty)
    
    if adjusted_score_s2 > THRESHOLD_S2:
        return "PASS", (0, 255, 0), f"S2 Verified ({adjusted_score_s2:.2f})", score_s1, adjusted_score_s2, quality_metrics, gradcam
    else:
        return "FAIL", (0, 0, 255), f"S2 Spoof ({adjusted_score_s2:.2f})", score_s1, adjusted_score_s2, quality_metrics, gradcam


# --- CREATE OUTPUT DIRECTORY ---
if SAVE_GRADCAM_FRAMES:
    os.makedirs(GRADCAM_DIR, exist_ok=True)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"ERROR: Could not open video source: {VIDEO_SOURCE}")
    exit(1)
print(f"Video source opened: {'Webcam' if VIDEO_SOURCE == 0 else VIDEO_SOURCE}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {frame_width}x{frame_height} @ {fps}fps, Total frames: {total_frames}")

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
print(f"Saving output video to: {OUTPUT_VIDEO}")

# Display window size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Face cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Results tracking
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
    "s2_fails": 0,
    "gradcam_count": 0
}

# Store last valid GradCAM for persistence
last_gradcam = None
last_face_bbox = None

print("\nProcessing video with GradCAM visualization...")
print("Features: Multi-scale Blur + Edge + FFT + GradCAM Heatmap Overlay")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame_count += 1
    stats["total_frames"] = frame_count

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
    
    frame_result = {"frame": frame_count, "faces": [], "status": "NO_FACE"}

    # Header info
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "v3 + GradCAM", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if len(faces) == 0:
        stats["no_face_count"] += 1
        cv2.putText(frame, "No Face Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Use last GradCAM heatmap if available (persistence)
        if last_gradcam is not None and last_face_bbox is not None:
            x, y, w, h = last_face_bbox
            gradcam_overlay = overlay_cam_on_image(frame[y:y+h, x:x+w], last_gradcam, alpha=GRADCAM_ALPHA * 0.5)
            frame[y:y+h, x:x+w] = gradcam_overlay
    else:
        for i, (x, y, w, h) in enumerate(faces):
            stats["faces_detected"] += 1
            
            # Detection box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
            
            # Expand ROI
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_width, x + w + pad)
            y2 = min(frame_height, y + h + pad)
            
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            # Liveness check with GradCAM
            status, color, text, s1_score, s2_score, quality_metrics, gradcam = check_liveness_with_gradcam(face_roi, frame_count)
            
            # Update last GradCAM if new one computed
            if gradcam is not None:
                last_gradcam = gradcam
                last_face_bbox = (x1, y1, x2-x1, y2-y1)
                stats["gradcam_count"] += 1
            
            # Apply GradCAM overlay to face region in the frame
            if last_gradcam is not None:
                gradcam_overlay = overlay_cam_on_image(face_roi, last_gradcam, alpha=GRADCAM_ALPHA)
                frame[y1:y2, x1:x2] = gradcam_overlay
            
            # Track failures
            if "Blur" in text:
                stats["blur_fails"] += 1
            if "Texture" in text:
                stats["edge_fails"] += 1
            if "Screen" in text:
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
                "quality_score": quality_metrics['quality_score']
            })
            
            # Draw result box and text
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, 3)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Q:{quality_metrics['quality_score']:.2f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # GradCAM indicator
            if last_gradcam is not None:
                cv2.putText(frame, "CAM", (x+w-35, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    frame_results.append(frame_result)

    # Write frame
    out.write(frame)

    # Display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow('Liveness Detection v3 + GradCAM', display_frame)
    
    # Progress
    if frame_count % 50 == 0:
        print(f"  Processed {frame_count}/{total_frames} frames... (GradCAM: {stats['gradcam_count']})")
        gc.collect()  # Force garbage collection
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate verdict
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

# Write results
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("     LIVENESS DETECTION v3 + GradCAM RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Video: {VIDEO_SOURCE} ({frame_width}x{frame_height} @ {fps}fps)\n")
    f.write("-" * 70 + "\n\n")
    
    f.write("FEATURES:\n")
    f.write("  - Multi-scale Laplacian Blur Detection\n")
    f.write("  - Edge Strength Analysis (Sobel)\n")
    f.write("  - FFT Spectral Analysis\n")
    f.write("  - GradCAM Heatmap Overlay\n\n")
    
    f.write("STATISTICS:\n")
    f.write(f"  Total Frames: {stats['total_frames']}\n")
    f.write(f"  Faces Detected: {stats['faces_detected']}\n")
    f.write(f"  PASS: {stats['pass_count']} | FAIL: {stats['fail_count']}\n")
    f.write(f"  GradCAM Frames: {stats['gradcam_count']}\n")
    
    if stats['faces_detected'] > 0:
        f.write(f"  Pass Rate: {stats['pass_count']/stats['faces_detected']*100:.1f}%\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"FINAL VERDICT: {final_verdict}\n")
    f.write(f"CONFIDENCE: {confidence:.1f}%\n")
    f.write("=" * 70 + "\n")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE (v3 + GradCAM)")
print("=" * 60)
print(f"Total frames: {stats['total_frames']}")
print(f"Faces detected: {stats['faces_detected']}")
print(f"PASS: {stats['pass_count']} | FAIL: {stats['fail_count']}")
print(f"GradCAM frames generated: {stats['gradcam_count']}")
print(f"\nFINAL VERDICT: {final_verdict} ({confidence:.1f}%)")
print(f"\nOutput: {OUTPUT_VIDEO}")
print(f"Log: {OUTPUT_LOG}")
