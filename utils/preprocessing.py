import cv2
import numpy as np
from PIL import Image
import config

def preprocess_face_for_cnn(face_bgr, target_size=config.IMAGE_SIZE):
    """
    Preprocesses face crop BGR image array for EfficientNetB0 input.
    - Resizes to target_size (224, 224)
    - Converts BGR to RGB
    - Expands dimensions to (1, 224, 224, 3)
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    
    # Resize image
    resized = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert to float array (EfficientNetB0 handles preprocessing internally or expects 0-255 inputs depending on tf version)
    rgb_batch = np.expand_dims(rgb.astype(np.float32), axis=0)
    return rgb_batch

def convert_bgr_to_grayscale(img_bgr):
    """Converts a BGR image to Grayscale."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
