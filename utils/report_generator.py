import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import config

def generate_forensic_pdf(analysis_result, output_pdf_path):
    """
    Generates a professional forensic PDF report for an image or video analysis.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e293b')
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0284c7')
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    verdict_color = colors.HexColor('#ef4444') if analysis_result.get('verdict') == 'FAKE' else (
        colors.HexColor('#f97316') if analysis_result.get('verdict') == 'SUSPICIOUS' else colors.HexColor('#22c55e')
    )

    # Header section
    story.append(Paragraph("DEEPSHIELD", title_style))
    story.append(Paragraph("Adaptive Explainable Deepfake Detection & Forensic Localization Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=15))

    # General Information Table
    media_type = analysis_result.get('media_type', 'Image').capitalize()
    verdict = analysis_result.get('verdict', 'UNKNOWN')
    score_pct = f"{analysis_result.get('fusion_score', 0) * 100:.1f}%"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info_data = [
        [Paragraph("<b>Report ID:</b>", body_style), Paragraph(analysis_result.get('id', 'N/A'), body_style),
         Paragraph("<b>Date/Time:</b>", body_style), Paragraph(timestamp_str, body_style)],
        [Paragraph("<b>File Name:</b>", body_style), Paragraph(analysis_result.get('filename', 'N/A'), body_style),
         Paragraph("<b>Media Type:</b>", body_style), Paragraph(media_type, body_style)],
        [Paragraph("<b>Verdict:</b>", body_style), Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", body_style),
         Paragraph("<b>Suspicion Score:</b>", body_style), Paragraph(f"<b>{score_pct}</b>", body_style)]
    ]

    if media_type == 'Video':
        info_data.append([
            Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{analysis_result.get('duration', 0)}s", body_style),
            Paragraph("<b>Frames Analyzed:</b>", body_style), Paragraph(str(analysis_result.get('frames_analyzed', 0)), body_style)
        ])
        info_data.append([
            Paragraph("<b>Peak Frame:</b>", body_style), Paragraph(f"Frame {analysis_result.get('peak_frame', 'N/A')}", body_style),
            Paragraph("<b>Peak Timestamp:</b>", body_style), Paragraph(f"{analysis_result.get('peak_timestamp', 0):.2f}s", body_style)
        ])

    info_table = Table(info_data, colWidths=[110, 150, 110, 150])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # Evidence Breakdown Section
    story.append(Paragraph("1. Multi-Signal Evidence Breakdown", heading_style))
    cnn_sc = f"{analysis_result.get('cnn_score', 0) * 100:.1f}%"
    fft_sc = f"{analysis_result.get('fft_score', 0) * 100:.1f}%"
    blink_sc = f"{analysis_result.get('blink_score', 0) * 100:.1f}%" if media_type == 'Video' else 'N/A (Image Mode)'

    ev_data = [
        ["Signal Modality", "Technique Used", "Suspicion Level"],
        ["Visual Evidence", "EfficientNetB0 Spatial CNN", cnn_sc],
        ["Frequency Evidence", "2D FFT Spectrum Energy Ratio", fft_sc],
        ["Behavioral Evidence", "MediaPipe Temporal Eye Blinking (EAR)", blink_sc]
    ]

    ev_table = Table(ev_data, colWidths=[160, 240, 120])
    ev_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ev_table)
    story.append(Spacer(1, 15))

    # Peak Frame & Bounding Box Section (if available)
    peak_frame_path = analysis_result.get('peak_frame_path')
    if peak_frame_path and os.path.exists(peak_frame_path):
        story.append(Paragraph("2. Forensic Peak Frame Localization", heading_style))
        story.append(Paragraph(
            f"The image below illustrates the frame with the highest suspicion score ({score_pct}) "
            f"extracted at timestamp <b>{analysis_result.get('peak_timestamp', 0):.2f}s</b> (Frame {analysis_result.get('peak_frame', 'N/A')}). "
            "The primary detected facial region is bounded below:", body_style
        ))
        story.append(Spacer(1, 8))
        try:
            img = RLImage(peak_frame_path, width=240, height=240)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"<i>Image preview unavailable ({str(e)})</i>", body_style))
        story.append(Spacer(1, 15))

    # Suspicious Time Segments (if Video)
    suspicious_segments = analysis_result.get('suspicious_segments', [])
    if media_type == 'Video':
        story.append(Paragraph("3. Suspicious Time Segments Detected", heading_style))
        if suspicious_segments:
            seg_text = ", ".join([f"<b>{s['start']} → {s['end']}</b> (Frames {s['start_frame']}-{s['end_frame']})" for s in suspicious_segments])
            story.append(Paragraph(f"Consecutive frame anomalies localized at time interval(s): {seg_text}", body_style))
        else:
            story.append(Paragraph("No continuous time segment exceeded the suspicion threshold (0.70).", body_style))
        story.append(Spacer(1, 15))

    # Interpretative Evidence Explanation
    story.append(Paragraph("4. Explainable Evidence Interpretation", heading_style))
    exp_text = analysis_result.get('explanation', "Analysis complete. Multi-signal evidence evaluated.")
    story.append(Paragraph(f"<i>\"{exp_text}\"</i>", body_style))
    story.append(Spacer(1, 15))

    # System Limitations & Disclaimers
    story.append(Paragraph("5. Forensic Disclaimers & System Limitations", heading_style))
    disclaimer_text = (
        "<b>Notice:</b> DeepShield provides an AI-assisted detection score based on multi-signal feature heuristics, "
        "spatial CNN patterns, and frequency domain anomalies. This output does not constitute absolute legal or "
        "forensic proof of manipulation. Low-resolution media, extreme compression, or irregular lighting conditions "
        "may affect facial feature extraction accuracy."
    )
    story.append(Paragraph(disclaimer_text, body_style))

    doc.build(story)
    return output_pdf_path
