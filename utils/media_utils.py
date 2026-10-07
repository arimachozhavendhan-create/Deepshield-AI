import os
import cv2
import config

def allowed_file(filename, media_type='image'):
    """Validates uploaded file extensions."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if media_type == 'image':
        return ext in config.ALLOWED_IMAGE_EXTENSIONS
    elif media_type == 'video':
        return ext in config.ALLOWED_VIDEO_EXTENSIONS
    return False

def extract_video_metadata(video_path):
    """
    Extracts video metadata using OpenCV VideoCapture.
    Returns dict: duration (s), fps, total_frames, resolution (WxH).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np_isnan(fps):
        fps = 30.0  # Fallback FPS
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    return {
        'filename': os.path.basename(video_path),
        'duration': round(duration, 2),
        'fps': round(fps, 2),
        'total_frames': total_frames,
        'resolution': f"{width}x{height}",
        'width': width,
        'height': height
    }

def np_isnan(val):
    try:
        import numpy as np
        return np.isnan(val)
    except Exception:
        return False
