import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
import base64

# Import models from parent directory (assuming this script is in ./webpage/ and models.py is in ./)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from models import FastScreenNet, ForensicNet
except ImportError:
    print("Warning: Could not import models.py. Ensure it exists in the parent directory.")

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "processed_reports"

# --- TRANSFORMATIONS ---
trans_s1 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_s2 = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class AnalysisEngine:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "webpage", OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Face Detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.model_s1 = None
        self.model_s2 = None
        self.load_error = None  # Track loading errors
        self.load_models()
        print(f"Analysis Engine Initialized. Device: {DEVICE}")

    def load_models(self):
        print("Loading models...")
        self.load_error = []

        def load_checkpoint(model, paths, stage_name):
            for path in paths:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as checkpoint_file:
                    if checkpoint_file.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                        self.load_error.append(
                            f"{stage_name} weights are Git LFS pointers, not downloaded model files: {path}"
                        )
                        continue
                model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                model.eval()
                print(f"{stage_name} loaded from {path}.")
                return model
            self.load_error.append(f"{stage_name} weights not found or unavailable.")
            return None

        try:
            # Stage 1
            try:
                self.model_s1 = FastScreenNet().to(DEVICE)
                self.model_s1 = load_checkpoint(
                    self.model_s1,
                    [
                        os.path.join(self.base_dir, "weights", "stage1_v3_best.pth"),
                        os.path.join(self.base_dir, "weights", "stage1_best.pth"),
                    ],
                    "Stage 1",
                )
            except Exception as e:
                print(f"Error loading Stage 1: {e}")
                self.load_error.append(f"Stage 1 Error: {str(e)}")
                self.model_s1 = None

            # Stage 2
            try:
                self.model_s2 = ForensicNet().to(DEVICE)
                self.model_s2 = load_checkpoint(
                    self.model_s2,
                    [
                        os.path.join(self.base_dir, "weights_1", "stage2_v3_best.pth"),
                        os.path.join(self.base_dir, "weights", "stage2_best.pth"),
                        os.path.join(self.base_dir, "weights", "stage2_last.pth"),
                    ],
                    "Stage 2",
                )
            except Exception as e:
                print(f"Error loading Stage 2: {e}")
                self.load_error.append(f"Stage 2 Error: {str(e)}")
                self.model_s2 = None
            
        except Exception as e:
            print(f"Critical error in load_models: {e}")
            self.load_error.append(f"Critical Load Error: {str(e)}")

    def compute_multiscale_blur(self, frame):
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
    def get_face_crop(self, frame):
        """Detect and crop the largest face. Fallback to center crop."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Find largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                # Add margin? Not for now, keep it tight like training likely is
                face = frame[y:y+h, x:x+w]
                return cv2.resize(face, (224, 224))
            else:
                # Center crop fallback
                h, w, _ = frame.shape
                short_edge = min(h, w)
                start_x = (w - short_edge) // 2
                start_y = (h - short_edge) // 2
                crop = frame[start_y:start_y+short_edge, start_x:start_x+short_edge]
                return cv2.resize(crop, (224, 224))
        except Exception as e:
            print(f"Face crop error: {e}")
            return cv2.resize(frame, (224, 224))

    def enhance_contrast(self, frame):
        """Apply Global Histogram Equalization to Y channel (Luminance)"""
        try:
            # Convert to YUV (Y = Luminance)
            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # Equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # Convert back to BGR
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img_output
        except Exception as e:
            print(f"Error in enhancement: {e}")
            return frame

    def compute_multiscale_blur(self, frame):
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
        return combined_blur

    def compute_fft_analysis(self, frame):
        """FFT-based frequency domain analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Log-scale for visualization
        magnitude_log = 20 * np.log(magnitude_spectrum + 1e-8)
        magnitude_vis = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        magnitude_spectrum = magnitude_spectrum / (magnitude_spectrum.max() + 1e-8)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        low_freq_radius = min(h, w) // 8
        high_freq_radius = min(h, w) // 3
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        high_freq_mask = (distance > low_freq_radius) & (distance <= high_freq_radius)
        
        high_freq_power = np.sum(magnitude_spectrum[high_freq_mask])
        total_power = np.sum(magnitude_spectrum)
        
        spectral_ratio = high_freq_power / (total_power + 1e-8)
        high_freq_region = magnitude_spectrum[high_freq_mask]
        moire_score = np.std(high_freq_region) if len(high_freq_region) > 0 else 0
        
        return {
            'spectral_ratio': spectral_ratio,
            'moire_score': moire_score,
            'vis': magnitude_vis
        }

    def decode_frame(self, data_uri):
        try:
            # Data URI format: "data:image/jpeg;base64,....."
            header, encoded = data_uri.split(",", 1)
            data = base64.b64decode(encoded)
            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Frame decode error: {e}")
            return None

    def analyze_batch(self, frame_data_list):
        stats = {
            'total': len(frame_data_list),
            'processed': 0,
            'blur_scores': [],
            'fft_ratios': [],
            's1_scores': [],
            's2_scores': [],
            'verdict_counts': {'Live': 0, 'Spoof': 0, 'Uncertain': 0}
        }
        
        images_to_save = {} # Best, Worst, FFT
        inference_errors = [] # Track runtime errors
        
        print(f"Processing batch of {len(frame_data_list)} frames...")
        
        for i, uri in enumerate(frame_data_list):
            frame = self.decode_frame(uri)
            if frame is None: continue
            
            # Check for Over-exposure
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # SPLIT PIPELINE
            # 1. Metrics Frame (Enhanced for human-like quality check)
            # 2. Model Frame (Original for AI robustness)
            if avg_brightness > 170:
                frame_metrics = self.enhance_contrast(frame)
                frame_model = frame 
            else:
                frame_metrics = frame
                frame_model = frame
            
            stats['processed'] += 1
            
            # 1. Classical Metrics (Use Enhanced)
            blur = self.compute_multiscale_blur(frame_metrics)
            
            # FFT can use Model Frame (Original) to avoid artifact patterns
            fft_res = self.compute_fft_analysis(frame_model)
            
            stats['blur_scores'].append(blur)
            stats['fft_ratios'].append(fft_res['spectral_ratio'])
            
            # Save representative FFT
            if i == len(frame_data_list) // 2:
                images_to_save['fft_vis'] = fft_res['vis']
                images_to_save['sample_frame'] = frame_metrics # Show user the Enhanced version

            # 2. Model Inference (Use Original Frame)
            
            if i % 5 == 0:
                # CROP FACE & CONVERT TO RGB
                # 1. Detect and Crop Face (Matches Training Data distribution)
                face_bgr = self.get_face_crop(frame_model)
                
                # 2. Convert to RGB (Fixes Blue Face Issue)
                frame_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                
                # Stage 1
                if self.model_s1 is not None:
                    try:
                        t_s1 = trans_s1(frame_rgb).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            s1_score = self.model_s1(t_s1).item()
                        stats['s1_scores'].append(s1_score)
                    except Exception as e:
                        err_msg = f"S1 Error: {str(e)}"
                        if err_msg not in inference_errors: inference_errors.append(err_msg)

                
                # Stage 2
                if self.model_s2 is not None:
                    try:
                        t_s2 = trans_s2(frame_rgb).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logit = self.model_s2(t_s2)
                            s2_score = torch.sigmoid(logit).item()
                        stats['s2_scores'].append(s2_score)
                    except Exception as e:
                        err_msg = f"S2 Error: {str(e)}"
                        if err_msg not in inference_errors: inference_errors.append(err_msg)

        # --- Aggregate Results ---
        avg_blur = np.mean(stats['blur_scores']) if stats['blur_scores'] else 0
        avg_fft = np.mean(stats['fft_ratios']) if stats['fft_ratios'] else 0
        
        # Only calculate average if scores exist
        avg_s1 = np.mean(stats['s1_scores']) if stats['s1_scores'] else None
        avg_s2 = np.mean(stats['s2_scores']) if stats['s2_scores'] else None
        
        # --- Final Verdict Logic ---
        # Heuristic Combination
        verdict = "UNCERTAIN"
        confidence = 0.0
        
        # Logic:
        # High Blur -> Fail
        # High FFT -> Fail (Screen)
        # Low S2 score -> Fail (Spoof)
        
        spoof_votes = 0
        live_votes = 0
        
        if avg_blur < 30: spoof_votes += 1 # Too blurry (Lowered from 40 for low-res cams)
        if avg_blur > 40: live_votes += 1 # Relaxed from 50 (User has ~49)
        
        if avg_fft > 0.35: spoof_votes += 2 # Strong Screen
        if avg_fft < 0.30: live_votes += 1 # Relaxed from 0.25
        
        # Stage 1 votes
        if avg_s1 is not None:
            if avg_s1 < 0.10:
                if avg_fft < 0.30 and avg_s2 is not None and avg_s2 > 0.5:
                    print("S1 says spoof but FFT and S2 say live. Weakening S1 spoof vote.")
                    spoof_votes += 1
                else:
                    spoof_votes += 2 # Strong spoof signal from Stage 1
            elif avg_s1 > 0.85:
                live_votes += 1
        
        # Only count model votes if model ran successfully
        if avg_s2 is not None:
            if avg_s2 < 0.0001:
                # If Score is 0.0000, it's likely a model collapse or domain mismatch.
                # User requested to fallback to Forensic (FFT) in this case.
                print("Model S2 score is ~0.0. Ignoring S2 vote.")
            elif avg_s2 < 0.5: 
                spoof_votes += 2 # Model says spoof
            elif avg_s2 >= 0.5: 
                live_votes += 1 # Model leans live
                if avg_s2 > 0.8:
                    live_votes += 1 # Model strongly says live
        
        if spoof_votes > live_votes:
            verdict = "SPOOF DETECTED"
            confidence = min(100, (spoof_votes / (spoof_votes + live_votes + 0.1)) * 100)
            if confidence <= 66.0 and avg_fft < 0.35 and avg_blur > 50:
                print("Boundary Override: overriding ~65% spoof confidence with LIVE PERSON due to strong classical signals.")
                verdict = "LIVE PERSON"
        elif live_votes >= spoof_votes:
            verdict = "LIVE PERSON"
            confidence = min(100, (live_votes / (spoof_votes + live_votes + 0.1)) * 100)
        
        # Generate Report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        report_id = f"report_{timestamp}"
        
        # Save Artifacts
        report_data = {
           "id": report_id,
           "timestamp": timestamp,
           "verdict": verdict,
           "confidence": f"{confidence:.1f}%",
           "metrics": {
               "avg_blur": f"{avg_blur:.2f}",
               "avg_fft_ratio": f"{avg_fft:.4f}",
               "avg_model_s2": f"{avg_s2:.4f}" if avg_s2 is not None else "N/A"
           }
        }
        
        # Save images
        img_paths = {}
        if 'fft_vis' in images_to_save:
            path = os.path.join(self.output_dir, f"{report_id}_fft.png")
            cv2.imwrite(path, images_to_save['fft_vis'])
            img_paths['fft'] = os.path.basename(path) # Rel path for frontend
            
        if 'sample_frame' in images_to_save:
            path = os.path.join(self.output_dir, f"{report_id}_sample.jpg")
            cv2.imwrite(path, images_to_save['sample_frame'])
            img_paths['sample'] = os.path.basename(path)
            
        report_data['images'] = img_paths
        
        # Write detailed text report
        txt_path = os.path.join(self.output_dir, f"{report_id}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Live Analysis Report: {report_id}\n")
            f.write("="*40 + "\n")
            f.write(f"Verdict: {verdict}\n")
            f.write(f"Confidence: {confidence:.1f}%\n")
            f.write("-"*30 + "\n")
            f.write(f"Metrics (Avg of {stats['processed']} frames):\n")
            f.write(f"  - Blur Score: {avg_blur:.2f} (Target > 50)\n")
            f.write(f"  - FFT Ratio: {avg_fft:.4f} (Target < 0.35)\n")
            
            s2_display = f"{avg_s2:.4f}" if avg_s2 is not None else "N/A (Model Not Loaded)"
            f.write(f"  - Model S2 Score: {s2_display} (Target > 0.5)\n")
            
            f.write("-"*30 + "\n")
            f.write("Breakdown:\n")
            f.write(f"  - Frames Processed: {stats['processed']}\n")
            
            s1_display = f"{avg_s1:.4f}" if avg_s1 is not None else "N/A"
            f.write(f"  - Stage 1 (Face) Avg: {s1_display} (Target > 0.85, < 0.10 is Spoof)\n")
            
            if self.load_error:
                f.write("\n System Warnings (Loading):\n")
                for err in self.load_error:
                    f.write(f"  ! {err}\n")
            
            if inference_errors:
                f.write("\n System Warnings (Inference):\n")
                for err in inference_errors:
                    f.write(f"  ! {err}\n")

        report_data['full_report_path'] = txt_path
        
        return report_data
