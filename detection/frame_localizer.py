import config

class FrameLocalizer:
    """
    Forensic Localization Engine.
    Identifies peak suspicious frame, groups consecutive suspicious frames into time segments,
    and structures timeline data for Chart.js dashboard visualization.
    """
    def __init__(self, threshold=config.SUSPICION_THRESHOLD):
        self.threshold = threshold

    def localize(self, frame_records):
        """
        Analyzes a sequence of evaluated frame records.
        Returns:
            dict: {
                'peak_frame': int,
                'peak_timestamp': float,
                'peak_score': float,
                'peak_record': dict,
                'suspicious_frames_count': int,
                'suspicious_segments': list of dicts,
                'timeline_data': list of dicts
            }
        """
        if not frame_records:
            return {
                'peak_frame': 0,
                'peak_timestamp': 0.0,
                'peak_score': 0.0,
                'peak_record': None,
                'suspicious_frames_count': 0,
                'suspicious_segments': [],
                'timeline_data': []
            }

        peak_score = -1.0
        peak_record = frame_records[0]
        suspicious_frames = []

        # Find Peak Frame and flag suspicious frames
        for rec in frame_records:
            sc = rec['fusion_score']
            if sc > peak_score:
                peak_score = sc
                peak_record = rec

            if sc >= self.threshold:
                suspicious_frames.append(rec)

        # Detect consecutive suspicious frame time-segments
        segments = self._group_suspicious_segments(suspicious_frames)

        return {
            'peak_frame': peak_record['frame_number'],
            'peak_timestamp': round(peak_record['timestamp'], 2),
            'peak_score': round(peak_record['fusion_score'], 4),
            'peak_record': peak_record,
            'suspicious_frames_count': len(suspicious_frames),
            'suspicious_segments': segments,
            'timeline_data': frame_records
        }

    def _group_suspicious_segments(self, suspicious_frames):
        """
        Groups consecutive suspicious frames into start/end time ranges.
        """
        if not suspicious_frames:
            return []

        segments = []
        current_seg = [suspicious_frames[0]]

        for i in range(1, len(suspicious_frames)):
            prev = suspicious_frames[i-1]
            curr = suspicious_frames[i]

            # If frames are close in sequence (allowing gap of max 15 sampled frames)
            if curr['frame_number'] - prev['frame_number'] <= config.FRAME_SAMPLE_RATE + 5:
                current_seg.append(curr)
            else:
                segments.append(self._format_segment(current_seg))
                current_seg = [curr]

        if current_seg:
            segments.append(self._format_segment(current_seg))

        return segments

    def _format_segment(self, frame_list):
        start_ts = frame_list[0]['timestamp']
        end_ts = frame_list[-1]['timestamp']
        
        def format_time(seconds):
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins:02d}:{secs:05.2f}"

        return {
            'start': format_time(start_ts),
            'end': format_time(end_ts),
            'start_seconds': round(start_ts, 2),
            'end_seconds': round(end_ts, 2),
            'start_frame': frame_list[0]['frame_number'],
            'end_frame': frame_list[-1]['frame_number'],
            'frame_count': len(frame_list)
        }
