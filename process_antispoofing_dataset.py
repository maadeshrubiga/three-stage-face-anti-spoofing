"""
Process the Anti-Spoofing dataset from Kaggle (tapakah68).
Extracts frames from videos and organizes everything into live/spoof folders.
"""

import os
import cv2
import glob
import shutil

# CONFIGURATION
SOURCE_DIR = 'dataset/spoof/all'  # Where the downloaded dataset is
LIVE_OUTPUT = 'dataset/live/antispoofing'
SPOOF_OUTPUT = 'dataset/spoof/antispoofing'

FRAME_INTERVAL = 15  # Extract every Nth frame
MAX_FRAMES_PER_VIDEO = 30  # Limit frames per video

# Map folders to labels
LIVE_FOLDERS = ['live_selfie', 'live_video']
SPOOF_FOLDERS = ['replay', 'printouts', 'cut-out printouts']


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_frames_from_video(video_path, output_dir, frame_interval=15, max_frames=30):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [ERROR] Could not open: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Clean filename for Windows compatibility
    video_name = "".join(c for c in video_name if c.isalnum() or c in ('_', '-'))[:50]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if saved_count >= max_frames:
                break
            
            frame_filename = f"{video_name}_f{saved_count:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def copy_images(source_folder, output_dir):
    """Copy image files to output directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    copied = 0
    
    for ext in extensions:
        for img_path in glob.glob(os.path.join(source_folder, ext)):
            filename = os.path.basename(img_path)
            # Clean filename
            name, ext_part = os.path.splitext(filename)
            clean_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))[:50]
            new_filename = f"{clean_name}{ext_part}"
            
            dest_path = os.path.join(output_dir, new_filename)
            shutil.copy2(img_path, dest_path)
            copied += 1
    
    return copied


def process_folder(folder_name, is_live=False):
    """Process a folder from the anti-spoofing dataset."""
    source_path = os.path.join(SOURCE_DIR, folder_name)
    
    if not os.path.exists(source_path):
        print(f"  [SKIP] Folder not found: {source_path}")
        return 0
    
    # Determine output directory
    if is_live:
        output_dir = os.path.join(LIVE_OUTPUT, folder_name.replace(' ', '_'))
    else:
        output_dir = os.path.join(SPOOF_OUTPUT, folder_name.replace(' ', '_'))
    
    ensure_dir(output_dir)
    
    total_processed = 0
    
    # Process videos
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MOV', '*.MP4']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(source_path, ext)))
    
    if video_files:
        print(f"  Processing {len(video_files)} videos...")
        for video_path in video_files:
            frames = extract_frames_from_video(video_path, output_dir, FRAME_INTERVAL, MAX_FRAMES_PER_VIDEO)
            total_processed += frames
    
    # Copy images directly
    image_count = copy_images(source_path, output_dir)
    total_processed += image_count
    
    if image_count > 0:
        print(f"  Copied {image_count} images")
    
    return total_processed


def main():
    print("=" * 60)
    print("Processing Anti-Spoofing Dataset (tapakah68)")
    print("=" * 60)
    
    # Create output directories
    ensure_dir(LIVE_OUTPUT)
    ensure_dir(SPOOF_OUTPUT)
    
    total_live = 0
    total_spoof = 0
    
    # Process LIVE folders
    print("\n[LIVE SAMPLES]")
    for folder in LIVE_FOLDERS:
        print(f"\nProcessing: {folder}")
        count = process_folder(folder, is_live=True)
        total_live += count
        print(f"  -> Total: {count} samples")
    
    # Process SPOOF folders
    print("\n[SPOOF SAMPLES]")
    for folder in SPOOF_FOLDERS:
        print(f"\nProcessing: {folder}")
        count = process_folder(folder, is_live=False)
        total_spoof += count
        print(f"  -> Total: {count} samples")
    
    print("\n" + "=" * 60)
    print(f"DONE!")
    print(f"  Live samples added:  {total_live}")
    print(f"  Spoof samples added: {total_spoof}")
    print(f"\nOutput directories:")
    print(f"  Live:  {LIVE_OUTPUT}")
    print(f"  Spoof: {SPOOF_OUTPUT}")
    print(f"\nNext step: Run 'python data_splitter.py' to update training splits.")
    print("=" * 60)


if __name__ == "__main__":
    main()
