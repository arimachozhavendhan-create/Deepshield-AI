import cv2
import numpy as np
import os
import config

class FFTAnalyzer:
    """
    Frequency Domain Anomaly Detector using 2D Discrete Fast Fourier Transform (FFT).
    Analyzes high-frequency spectral energy distributions for synthetic blending artifacts.
    """
    def __init__(self):
        pass

    def analyze_face(self, face_bgr, save_visualization_path=None):
        """
        Calculates 2D FFT magnitude spectrum statistics on facial crop.
        Returns:
            dict: {
                'fft_score': float (0.0 to 1.0),
                'high_freq_ratio': float,
                'spectrum_path': str or None
            }
        """
        if face_bgr is None or face_bgr.size == 0:
            return {'fft_score': 0.0, 'high_freq_ratio': 0.0, 'spectrum_path': None}

        # Convert to Grayscale
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Perform 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate Magnitude Spectrum
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

        # Calculate Low-Frequency vs High-Frequency Energy
        center_y, center_x = h // 2, w // 2
        r = min(h, w) // 8  # Radius for low-frequency center mask

        y, x = np.ogrid[:h, :w]
        mask_low = (x - center_x)**2 + (y - center_y)**2 <= r**2

        low_freq_energy = np.sum(np.abs(f_shift)[mask_low])
        total_energy = np.sum(np.abs(f_shift)) + 1e-8
        high_freq_energy = total_energy - low_freq_energy

        high_freq_ratio = float(high_freq_energy / total_energy)

        # Deepfakes often exhibit abnormal high-frequency noise spikes or unnatural smoothing
        # Normalize into a suspicion score between 0.0 and 1.0
        # Expected natural face high_freq_ratio is around 0.65 - 0.85
        if high_freq_ratio > 0.85:
            fft_score = min(1.0, 0.70 + (high_freq_ratio - 0.85) * 2.0)
        elif high_freq_ratio < 0.55:
            fft_score = min(1.0, 0.70 + (0.55 - high_freq_ratio) * 2.0)
        else:
            # Normal range
            fft_score = max(0.1, (abs(high_freq_ratio - 0.72) / 0.15) * 0.5)

        fft_score = round(max(0.0, min(1.0, float(fft_score))), 4)

        spectrum_saved_path = None
        if save_visualization_path:
            try:
                # Normalize magnitude spectrum to 0-255 uint8 for visual output
                norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                norm_spectrum = np.uint8(norm_spectrum)
                # Apply Color Map for visually attractive forensic display
                colored_spectrum = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_JET)
                cv2.imwrite(save_visualization_path, colored_spectrum)
                spectrum_saved_path = save_visualization_path
            except Exception as e:
                print(f"[FFT WARNING] Failed to save spectrum image: {e}")

        return {
            'fft_score': fft_score,
            'high_freq_ratio': round(high_freq_ratio, 4),
            'spectrum_path': spectrum_saved_path
        }
