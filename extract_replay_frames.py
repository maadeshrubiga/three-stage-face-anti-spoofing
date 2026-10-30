"""
Extract frames from replay attack video files (.mp4) for training.
Saves frames as .jpg images in the same directory structure.
"""

import os
import cv2
import glob

# CONFIGURATION
REPLAY_ROOT = 'dataset/spoof/replay/samples'
FRAME_INTERVAL = 10  # Extract every Nth frame (adjust to control number of frames)
OUTPUT_FORMAT = 'jpg'
MAX_FRAMES_PER_VIDEO = 50  # Limit frames per video to avoid imbalance

def extract_frames_from_video(video_path, output_dir, frame_interval=10, max_frames=50):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract per video
    
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            if saved_count >= max_frames:
                break
                
            frame_filename = f"{video_name}_frame_{saved_count:04d}.{OUTPUT_FORMAT}"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return saved_count


def main():
    print(f"Scanning for video files in: {REPLAY_ROOT}")
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(REPLAY_ROOT, '**', ext), recursive=True))
    
    print(f"Found {len(video_files)} video files.\n")
    
    if len(video_files) == 0:
        print("No video files found. Check the path.")
        return
    
    total_frames_extracted = 0
    
    for i, video_path in enumerate(video_files, 1):
        video_dir = os.path.dirname(video_path)
        print(f"[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        
        frames_extracted = extract_frames_from_video(
            video_path=video_path,
            output_dir=video_dir,  # Save frames in same folder as video
            frame_interval=FRAME_INTERVAL,
            max_frames=MAX_FRAMES_PER_VIDEO
        )
        
        print(f"  -> Extracted {frames_extracted} frames")
        total_frames_extracted += frames_extracted
    
    print(f"\n{'='*50}")
    print(f"DONE! Extracted {total_frames_extracted} total frames from {len(video_files)} videos.")
    print(f"\nNext step: Run 'python data_splitter.py' to update the training splits.")


if __name__ == "__main__":
    main()
