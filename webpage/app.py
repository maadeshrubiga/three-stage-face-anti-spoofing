from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import cv2
import base64
import tempfile
from analysis_engine import AnalysisEngine

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize Analysis Engine
# Pass PARENT_DIR so it can find 'weights/' and 'models.py'
engine = AnalysisEngine(PARENT_DIR)


# Error handler for file too large
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_frames_from_video(video_path, num_frames=60):
    """Extract evenly spaced frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        # If video has fewer frames, take all of them
        step = 1
        num_frames = total_frames
    else:
        # Calculate step to evenly sample frames
        step = total_frames // num_frames
    
    frame_indices = [i * step for i in range(num_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame to base64 data URI (same format as webcam capture)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_frame}"
            frames.append(data_uri)
    
    cap.release()
    return frames

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze an uploaded video file for spoof detection."""
    try:
        # Check if video file is in the request
        if 'video' not in request.files:
            return jsonify({"error": "No video file in request"}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        if not allowed_video_file(video_file.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"}), 400
        
        # Save video to temporary file (Windows-compatible approach)
        import uuid
        tmp_filename = f"video_upload_{uuid.uuid4().hex}.mp4"
        tmp_path = os.path.join(tempfile.gettempdir(), tmp_filename)
        video_file.save(tmp_path)
        
        try:
            # Extract frames from video
            print(f"Extracting frames from uploaded video: {video_file.filename}")
            frames = extract_frames_from_video(tmp_path, num_frames=60)
            
            if not frames:
                return jsonify({"error": "Could not extract frames from video"}), 400
            
            print(f"Extracted {len(frames)} frames from video for analysis.")
            
            # Run analysis using the same engine
            report = engine.analyze_batch(frames)
            
            return jsonify({"status": "success", "report": report})
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"Error during video analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({"error": "No frames received"}), 400
            
        print(f"Received {len(frames)} frames for analysis.")
        
        # Run analysis
        report = engine.analyze_batch(frames)
        
        return jsonify({"status": "success", "report": report})
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(os.path.join(BASE_DIR, "processed_reports"), filename)

if __name__ == '__main__':
    # Run securely on 0.0.0.0 (accessible locally) or localhost
    # Debug=True for development
    print("Starting Web Server...")
    print("Go to http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=True)
