import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIG ---
# Set this to the path of the image you want to process
INPUT_IMAGE_PATH = r"C:\Users\manos\Videos\Final Year\dataset\live\FEI\originalimages_part1\1-02.jpg" 
OUTPUT_DIR = "output_analysis"

def compute_multiscale_blur(frame):
    """
    Multi-scale Laplacian blur detection.
    Returns composite blur score and individual scale scores.
    Low variance indicates a blurry image.
    """
    if frame is None:
        return 0.0, []
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scales = [1.0, 0.75, 0.5]
    blur_scores = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            resized = gray
        # Laplacian calculation
        laplacian = cv2.Laplacian(resized, cv2.CV_64F)
        variance = laplacian.var()
        blur_scores.append(variance)
    
    # Weighted average of scales
    weights = [0.5, 0.3, 0.2]
    combined_blur = sum(s * w for s, w in zip(blur_scores, weights))
    return combined_blur, blur_scores

def compute_fft_analysis(frame):
    """
    FFT-based frequency domain analysis.
    detects spoof-specific artifacts such as periodic printing noise, 
    screen refresh patterns, and Moiré effects.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Normalize for visualization (Log scale)
    magnitude_log = 20 * np.log(magnitude_spectrum + 1e-8)
    magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # --- Analysis Metrics ---
    # Linear magnitude for calculation
    magnitude_spectrum_lin = magnitude_spectrum / (magnitude_spectrum.max() + 1e-8)
    
    h, w = magnitude_spectrum_lin.shape
    center_y, center_x = h // 2, w // 2
    # Define frequency bands
    low_freq_radius = min(h, w) // 8
    high_freq_radius = min(h, w) // 3
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    low_freq_mask = distance <= low_freq_radius
    high_freq_mask = (distance > low_freq_radius) & (distance <= high_freq_radius)
    
    # Calculate power in bands
    # low_freq_power = np.sum(magnitude_spectrum_lin[low_freq_mask]) # Unused in this metric but useful context
    high_freq_power = np.sum(magnitude_spectrum_lin[high_freq_mask])
    total_power = np.sum(magnitude_spectrum_lin)
    
    # Spectral metrics
    spectral_ratio = high_freq_power / (total_power + 1e-8)
    high_freq_region = magnitude_spectrum_lin[high_freq_mask]
    moire_score = np.std(high_freq_region) if len(high_freq_region) > 0 else 0
    
    return {
        'spectral_ratio': spectral_ratio,
        'moire_score': moire_score,
        'magnitude_spectrum_vis': magnitude_norm
    }

def main():
    print(f"Working Directory: {os.getcwd()}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Potential paths to check
    possible_paths = [
        INPUT_IMAGE_PATH,
        os.path.join(script_dir, INPUT_IMAGE_PATH),
        os.path.join(script_dir, "..", os.path.basename(INPUT_IMAGE_PATH)),
        os.path.join(os.getcwd(), INPUT_IMAGE_PATH)
    ]
    
    target_image = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            target_image = os.path.abspath(p)
            break
            
    if target_image is None:
        print("Error: Input image not found.")
        print(f"Searched in: {possible_paths}")
        print("Please edit 'INPUT_IMAGE_PATH' in the script or place 'test_image.jpg' in the script folder or parent folder.")
        return

    # Create output directory
    # Save outputs in the same folder as this script, inside 'output_analysis'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Load Image
    print(f"Loading image: {target_image}")
    image = cv2.imread(target_image)
    if image is None:
        print("Failed to load image. Check format/permissions.")
        return
        
    print(f"Image shape: {image.shape}")
    
    # 1. Blur Detection
    print("\n--- 1. Laplacian Blur Detection ---")
    blur_score, blur_scales = compute_multiscale_blur(image)
    print(f"Blur Score (High=Sharp, Low=Blurry): {blur_score:.2f}")
    print(f"Detailed Scales: {blur_scales}")
    
    # 2. FFT Analysis
    print("\n--- 2. FFT Frequency Analysis ---")
    fft_results = compute_fft_analysis(image)
    print(f"Spectral Ratio: {fft_results['spectral_ratio']:.4f}")
    print(f"Moire/Noise Score: {fft_results['moire_score']:.4f}")
    
    # --- SAVE OUTPUTS ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save FFT Spectrum Image
    fft_filename = f"fft_spectrum_{timestamp}.png"
    fft_path = os.path.join(final_output_dir, fft_filename)
    cv2.imwrite(fft_path, fft_results['magnitude_spectrum_vis'])
    print(f"\n[Saved] FFT Spectrum: {fft_path}")
    
    # Annotate Original Image with scores
    annotated_img = image.copy()
    # Add a header background for readability
    cv2.rectangle(annotated_img, (0, 0), (600, 110), (0, 0, 0), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_img, f"Blur Score: {blur_score:.2f}", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_img, f"Spectral Ratio: {fft_results['spectral_ratio']:.4f}", (10, 60), font, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_img, f"Moire Score: {fft_results['moire_score']:.4f}", (10, 90), font, 0.8, (0, 255, 255), 2)
    
    annotated_filename = f"processed_result_{timestamp}.jpg"
    annotated_path = os.path.join(final_output_dir, annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    print(f"[Saved] Annotated Image: {annotated_path}")
    
    # Save Text Report
    report_filename = f"analysis_report_{timestamp}.txt"
    report_path = os.path.join(final_output_dir, report_filename)
    with open(report_path, "w") as f:
        f.write("Single Image Analysis Report\n")
        f.write("============================\n")
        f.write(f"Source Image: {target_image}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. Laplacian Blur Detection (Sharpness Check)\n")
        f.write("--------------------------------------------\n")
        f.write(f"Combined Blur Score: {blur_score:.2f}\n")
        f.write(f"Scale Breakdown: {[round(s, 2) for s in blur_scales]}\n\n")
        f.write("Explanation:\n")
        f.write("  - This technique uses the Variance of Laplacian to measure image sharpness.\n")
        f.write("  - Low scores (typically < 50-100) are considered BLURRY. This often happens in replay attacks\n")
        f.write("    (out of focus) or low-quality printed spoofs to hide texture details.\n")
        f.write("  - High scores (> 100) indicate a detailed, SHARP image, which is expected for genuine live faces.\n")
        f.write("  - Note: Very high scores (> 1000) might indicate artificial sharpening or extreme noise.\n\n")

        f.write("2. FFT Frequency Analysis (Artifact Check)\n")
        f.write("------------------------------------------\n")
        f.write(f"Spectral Ratio: {fft_results['spectral_ratio']:.6f}\n")
        f.write(f"Moire/Noise Score: {fft_results['moire_score']:.6f}\n\n")
        f.write("Explanation:\n")
        f.write("  - We analyze the image in the Frequency Domain (FFT) to find repeating patterns.\n")
        f.write("  - Spectral Ratio: Measures the proportion of energy in high frequencies.\n")
        f.write("    screen replay attacks often introduce pixel grid patterns (Moiré pattern),\n")
        f.write("    which appear as high-frequency noise.\n")
        f.write("  - Moire Score: A specific variance check in the high-frequency band.\n")
        f.write("  - Thresholds: Ratios > 0.15 typically indicate potential spoofing artifacts (screens/prints).\n\n")
        
        f.write("3. Automated Verdict & Type Classification\n")
        f.write("------------------------------------------\n")
        
        # Classification Logic
        prediction_type = "Unknown"
        confidence_level = "Low"
        
        # 1. Check for Screen/Replay artifacts first (Dominant feature)
        # Thresholds increased to avoid false positives on sharp live images (User image was ~0.25)
        # inference_v3 uses 0.4 as hard cutoff.
        if fft_results['spectral_ratio'] > 0.35 or fft_results['moire_score'] > 0.5: 
            prediction_type = "POTENTIAL SCREEN / REPLAY ATTACK"
            confidence_level = "High" if fft_results['spectral_ratio'] > 0.45 else "Medium"
            details = "Strong high-frequency grid patterns detected (Moiré effect), typical of filming digital screens."
            
        # 2. Check for Blur/Quality issues (Secondary)
        elif blur_score < 40:
            prediction_type = "LOW QUALITY / POTENTIAL PRINT ATTACK"
            confidence_level = "Medium"
            details = "Image is significantly blurry, lacking skin texture details. Common in low-res prints or defocus."
            
        # 3. If neither, assume Live
        # Slightly wider range for 'Live' to account for variations
        elif blur_score > 50:
            prediction_type = "LIKELY LIVE / BONA FIDE"
            confidence_level = "Medium"
            details = "Image has natural frequency distribution and sufficient sharpness."
            
            if fft_results['spectral_ratio'] > 0.15:
                details += "\n  (Note: Somewhat high frequency content detected, but below spoof threshold. Likely texture details.)"
        else:
            prediction_type = "UNCERTAIN / BORDERLINE"
            details = "Metrics are inconclusive. No strong spoof artifacts, but quality is mediocre."

        f.write(f"Predicted Type: {prediction_type}\n")
        f.write(f"Confidence:     {confidence_level}\n\n")
        f.write(f"Reasoning:\n  {details}\n")
        
        f.write("\nNOTE: This is a heuristic check and not a deep learning prediction.\n")
        f.write("For high-accuracy results, please use the main deep learning model (inference files).\n")
    
    print(f"[Saved] Text Report: {report_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
