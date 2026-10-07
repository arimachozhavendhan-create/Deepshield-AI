import config

class FusionEngine:
    """
    Multi-Signal Fusion Engine.
    Combines visual (CNN), frequency (FFT), and temporal (Blink) evidence scores.
    Re-normalizes weights when a modality (e.g. eye blinking in static images) is missing.
    """
    def __init__(self, cnn_weight=config.CNN_WEIGHT, fft_weight=config.FFT_WEIGHT, blink_weight=config.BLINK_WEIGHT):
        self.w_cnn = cnn_weight
        self.w_fft = fft_weight
        self.w_blink = blink_weight

    def fuse(self, cnn_score, fft_score, blink_score=None):
        """
        Fuses individual modality scores into an overall suspicion score (0.0 to 1.0).
        """
        # Validate score bounds
        c_score = max(0.0, min(1.0, float(cnn_score)))
        f_score = max(0.0, min(1.0, float(fft_score)))

        if blink_score is not None:
            # Full Video Modality Fusion
            b_score = max(0.0, min(1.0, float(blink_score)))
            total_weight = self.w_cnn + self.w_fft + self.w_blink
            
            fused_score = (self.w_cnn * c_score + self.w_fft * f_score + self.w_blink * b_score) / total_weight
            weights_used = {'cnn': self.w_cnn, 'fft': self.w_fft, 'blink': self.w_blink}
        else:
            # Image Mode (Missing Eye Blinking) -> Re-normalize CNN and FFT weights
            total_available = self.w_cnn + self.w_fft
            norm_cnn = self.w_cnn / total_available
            norm_fft = self.w_fft / total_available

            fused_score = norm_cnn * c_score + norm_fft * f_score
            weights_used = {'cnn': round(norm_cnn, 3), 'fft': round(norm_fft, 3), 'blink': 0.0}

        fused_score = round(max(0.0, min(1.0, float(fused_score))), 4)
        verdict = self._determine_verdict(fused_score)
        explanation = self._generate_explanation(c_score, f_score, blink_score, fused_score, verdict)

        return {
            'fusion_score': fused_score,
            'verdict': verdict,
            'weights': weights_used,
            'explanation': explanation
        }

    def _determine_verdict(self, score):
        if score >= config.SUSPICION_THRESHOLD:
            return 'FAKE'
        elif score >= 0.45:
            return 'SUSPICIOUS'
        else:
            return 'REAL'

    def _generate_explanation(self, cnn_sc, fft_sc, blink_sc, fused_sc, verdict):
        """
        Generates authentic explainable evidence breakdown based strictly on calculated scores.
        """
        reasons = []

        # CNN evidence text
        if cnn_sc >= 0.70:
            reasons.append("High spatial Inconsistency: EfficientNetB0 detected significant facial texture anomalies.")
        elif cnn_sc >= 0.45:
            reasons.append("Moderate spatial variations detected in facial boundary regions.")
        else:
            reasons.append("Natural visual facial characteristics detected.")

        # FFT evidence text
        if fft_sc >= 0.70:
            reasons.append("Abnormal frequency spectrum: FFT analysis indicates high-frequency synthetic blending artifacts.")
        elif fft_sc >= 0.45:
            reasons.append("Minor frequency spectrum noise detected.")
        else:
            reasons.append("Standard spatial frequency distribution observed.")

        # Blink evidence text (if applicable)
        if blink_sc is not None:
            if blink_sc >= 0.70:
                reasons.append("Behavioral Anomaly: Temporal eye aspect ratio indicates unnatural blinking frequency or static eye closures.")
            elif blink_sc >= 0.45:
                reasons.append("Slightly irregular blinking rhythm observed across frames.")
            else:
                reasons.append("Natural eye blinking pattern and eye movement verified.")

        if verdict == 'FAKE':
            summary = f"VERDICT: HIGH LIKELIHOOD OF MANIPULATION (Overall Suspicion: {round(fused_sc*100, 1)}%). " + " ".join(reasons)
        elif verdict == 'SUSPICIOUS':
            summary = f"VERDICT: MODERATE ANOMALY DETECTED (Overall Suspicion: {round(fused_sc*100, 1)}%). " + " ".join(reasons)
        else:
            summary = f"VERDICT: LIKELY AUTHENTIC MEDIA (Overall Suspicion: {round(fused_sc*100, 1)}%). " + " ".join(reasons)

        return summary
