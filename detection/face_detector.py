import cv2
import numpy as np
import mediapipe as mp
import config

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            model_selection=1 # 1 is best for full range/portrait
        )
        # Fallback Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_primary_face(self, image_bgr, draw_bbox=True, is_suspicious=False):
        """
        Detects the primary (largest) face in an image.
        Returns:
            face_crop (np.ndarray or None): BGR cropped face image
            bbox (tuple or None): (x, y, w, h) bounding box coordinates
            annotated_img (np.ndarray): Image with drawn face bounding box
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, None, image_bgr

        h_img, w_img, _ = image_bgr.shape
        annotated_img = image_bgr.copy()
        
        # 1. Try MediaPipe Face Detection
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        primary_bbox = None
        max_area = 0

        if results.detections:
            for detection in results.detections:
                bbox_relative = detection.location_data.relative_bounding_box
                x = int(bbox_relative.xmin * w_img)
                y = int(bbox_relative.ymin * h_img)
                w = int(bbox_relative.width * w_img)
                h = int(bbox_relative.height * h_img)
                
                # Clip bounds to image dimensions
                x = max(0, x)
                y = max(0, y)
                w = min(w_img - x, w)
                h = min(h_img - y, h)

                area = w * h
                if area > max_area and w > 10 and h > 10:
                    max_area = area
                    primary_bbox = (x, y, w, h)

        # 2. Fallback to Haar Cascade if MediaPipe found no face
        if primary_bbox is None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    primary_bbox = (x, y, w, h)

        if primary_bbox is None:
            return None, None, annotated_img

        x, y, w, h = primary_bbox
        
        # Add slight margin padding for crop
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x_start = max(0, x - pad_x)
        y_start = max(0, y - pad_y)
        x_end = min(w_img, x + w + pad_x)
        y_end = min(h_img, y + h + pad_y)

        face_crop = image_bgr[y_start:y_end, x_start:x_end]

        if draw_bbox:
            # Color: Red if high suspicion, Cyan/Blue if normal
            color = (0, 0, 238) if is_suspicious else (235, 175, 0)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 3)
            
            # Label banner
            label = "SUSPICIOUS FACE REGION" if is_suspicious else "PRIMARY FACE ANALYZED"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(annotated_img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return face_crop, primary_bbox, annotated_img

    def close(self):
        self.face_detection.close()
