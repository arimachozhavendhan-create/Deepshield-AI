import os
import numpy as np
import tensorflow as tf
import cv2
import config
from utils.preprocessing import preprocess_face_for_cnn

class CNNDetector:
    """
    Visual Evidence Detector powered by EfficientNetB0 architecture.
    Handles trained Keras model loading or safe demonstration fallback.
    """
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.custom_model_loaded = False
        self.model = None
        self.base_feature_extractor = None

        self._initialize_model()

    def _initialize_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"[DeepShield] Loading custom deepfake model from {self.model_path}...")
                self.model = tf.keras.models.load_model(self.model_path)
                self.custom_model_loaded = True
                print("[DeepShield] Custom EfficientNetB0 DeepShield model loaded successfully!")
                return
            except Exception as e:
                print(f"[DeepShield WARNING] Failed to load custom model: {e}")

        # Fallback to pretrained ImageNet feature extractor for demonstration mode
        print("[DeepShield INFO] Custom weights not found. Initializing EfficientNetB0 ImageNet feature extractor for Demonstration Mode.")
        try:
            self.base_feature_extractor = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
        except Exception as e:
            print(f"[DeepShield ERROR] Failed to load base EfficientNetB0: {e}")

    def analyze_face(self, face_bgr):
        """
        Analyzes a face crop BGR image.
        Returns dict:
            cnn_score (float): 0.0 to 1.0 (Higher = Higher Visual Inconsistency/Suspicion)
            is_demo (bool): True if using ImageNet feature heuristic demo mode
            status_text (str): Explanation of model state
        """
        if face_bgr is None or face_bgr.size == 0:
            return {'cnn_score': 0.0, 'is_demo': not self.custom_model_loaded, 'status': 'Invalid Crop'}

        batch_input = preprocess_face_for_cnn(face_bgr)
        if batch_input is None:
            return {'cnn_score': 0.0, 'is_demo': not self.custom_model_loaded, 'status': 'Preprocessing Error'}

        # Case 1: Custom Trained DeepShield Model
        if self.custom_model_loaded and self.model is not None:
            try:
                preds = self.model.predict(batch_input, verbose=0)
                # Assume binary output (0 = Real, 1 = Fake score)
                score = float(preds[0][0]) if preds.ndim > 1 else float(preds[0])
                score = max(0.0, min(1.0, score))
                return {
                    'cnn_score': round(score, 4),
                    'is_demo': False,
                    'status': 'Trained EfficientNetB0 Analysis'
                }
            except Exception as e:
                print(f"[DeepShield ERROR] Model prediction error: {e}")

        # Case 2: Demonstration Feature Heuristics (ImageNet base)
        try:
            # Extract EfficientNetB0 feature embeddings
            features = self.base_feature_extractor.predict(batch_input, verbose=0)[0]
            
            # Spatial variance heuristic based on Laplacian edge sharpness + feature dispersion
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            feat_std = np.std(features)
            
            # Calculate a normalized score for demonstration
            # Deepfakes often suffer from unnatural smoothing or edge artifacts
            smoothness_factor = max(0.0, 1.0 - (laplacian_var / 800.0))
            dispersion_factor = max(0.0, min(1.0, feat_std / 1.5))
            
            demo_score = 0.5 * smoothness_factor + 0.5 * dispersion_factor
            demo_score = round(max(0.15, min(0.95, demo_score)), 4)

            return {
                'cnn_score': demo_score,
                'is_demo': True,
                'status': 'DEMO MODE (EfficientNetB0 ImageNet Heuristics)'
            }
        except Exception as e:
            return {
                'cnn_score': 0.50,
                'is_demo': True,
                'status': f'Demo Fallback Error ({e})'
            }
