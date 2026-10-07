/* ==========================================================================
   DEEPSHIELD — Digital Forensic Evidence Workstation Script
   ========================================================================== */

let timelineChartInstance = null;
let currentTimelineData = [];

/**
 * Computes Cryptographic SHA-256 Hash of File ArrayBuffer
 */
async function computeFileSHA256(file) {
    try {
        const buffer = await file.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } catch (e) {
        return 'UNAVAILABLE_HASH';
    }
}

/**
 * Image Intake & Forensic Processing Handler
 */
function initImageDetectionPage() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const metaContainer = document.getElementById('file-meta-container');
    const btnSubmit = document.getElementById('btn-submit-image');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    async function handleFileSelection(file) {
        metaContainer.classList.remove('hidden');
        document.getElementById('meta-filename-val').textContent = file.name;
        document.getElementById('meta-format-val').textContent = file.type || file.name.split('.').pop().toUpperCase();
        document.getElementById('meta-size-val').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

        // Compute Dimensions
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            document.getElementById('meta-dim-val').textContent = `${img.width}x${img.height} PX`;
        };

        // Compute SHA-256
        document.getElementById('meta-hash-val').textContent = 'Computing SHA-256...';
        const hash = await computeFileSHA256(file);
        document.getElementById('meta-hash-val').textContent = hash.substring(0, 32) + '...';
    }

    btnSubmit.addEventListener('click', () => {
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select an image file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        document.getElementById('upload-panel').classList.add('hidden');
        document.getElementById('loading-spinner').classList.remove('hidden');
        document.getElementById('result-container').classList.add('hidden');

        fetch('/api/analyze/image', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) return res.json().then(e => { throw new Error(e.error || 'Server error'); });
            return res.json();
        })
        .then(data => {
            document.getElementById('loading-spinner').classList.add('hidden');
            renderImageResult(data);
        })
        .catch(err => {
            document.getElementById('loading-spinner').classList.add('hidden');
            document.getElementById('upload-panel').classList.remove('hidden');
            alert(`Analysis Error: ${err.message}`);
        });
    });
}

/**
 * Render Image Case Report Results
 */
function renderImageResult(data) {
    document.getElementById('result-container').classList.remove('hidden');

    if (data.is_demo) {
        document.getElementById('demo-mode-badge').classList.remove('hidden');
    }

    // Case Header
    document.getElementById('case-id-display').textContent = `CASE DS-${data.id.toUpperCase()}`;
    
    const verdictEl = document.getElementById('verdict-badge');
    verdictEl.textContent = data.verdict === 'FAKE' ? 'LIKELY MANIPULATED' : (data.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'LIKELY AUTHENTIC');
    verdictEl.className = `badge-verdict ${data.verdict.toLowerCase()}`;

    const scorePct = (data.fusion_score * 100).toFixed(1) + '%';
    document.getElementById('fusion-score-val').textContent = scorePct;
    document.getElementById('fusion-recap-val').textContent = scorePct;

    const barFill = document.getElementById('confidence-bar-fill');
    barFill.style.width = scorePct;
    barFill.className = `confidence-bar-fill ${data.verdict.toLowerCase()}`;

    // Sub-signal scores
    const cnnPct = (data.cnn_score * 100).toFixed(1) + '%';
    document.getElementById('cnn-score-val').textContent = cnnPct;
    setSeverityPill('cnn-severity-pill', data.cnn_score);

    const fftPct = (data.fft_score * 100).toFixed(1) + '%';
    document.getElementById('fft-score-val').textContent = fftPct;
    setSeverityPill('fft-severity-pill', data.fft_score);

    // Explanation
    document.getElementById('explanation-text').textContent = data.explanation;

    // Previews
    if (data.face_crop_filename) {
        document.getElementById('face-crop-img').src = `/results/${data.face_crop_filename}`;
    }
    if (data.fft_spectrum_filename) {
        document.getElementById('fft-spectrum-img').src = `/results/${data.fft_spectrum_filename}`;
    }

    // PDF Link
    document.getElementById('btn-download-pdf').href = `/api/report/${data.id}`;
}

/**
 * Video Intake & Forensic Processing Handler
 */
function initVideoDetectionPage() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const metaContainer = document.getElementById('file-meta-container');
    const btnSubmit = document.getElementById('btn-submit-video');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleVideoSelection(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleVideoSelection(fileInput.files[0]);
        }
    });

    async function handleVideoSelection(file) {
        metaContainer.classList.remove('hidden');
        document.getElementById('meta-filename-val').textContent = file.name;
        document.getElementById('meta-format-val').textContent = file.type || file.name.split('.').pop().toUpperCase();
        document.getElementById('meta-size-val').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

        document.getElementById('meta-hash-val').textContent = 'Computing SHA-256...';
        const hash = await computeFileSHA256(file);
        document.getElementById('meta-hash-val').textContent = hash.substring(0, 32) + '...';
    }

    btnSubmit.addEventListener('click', () => {
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select a video file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        document.getElementById('upload-panel').classList.add('hidden');
        document.getElementById('loading-spinner').classList.remove('hidden');
        document.getElementById('result-container').classList.add('hidden');

        // Progress line animation
        let pct = 20;
        const pInterval = setInterval(() => {
            if (pct < 90) {
                pct += 15;
                document.getElementById('analysis-progress-fill').style.width = pct + '%';
            }
        }, 1200);

        fetch('/api/analyze/video', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            clearInterval(pInterval);
            if (!res.ok) return res.json().then(e => { throw new Error(e.error || 'Server error'); });
            return res.json();
        })
        .then(data => {
            document.getElementById('loading-spinner').classList.add('hidden');
            renderVideoResult(data);
        })
        .catch(err => {
            clearInterval(pInterval);
            document.getElementById('loading-spinner').classList.add('hidden');
            document.getElementById('upload-panel').classList.remove('hidden');
            alert(`Video Forensic Error: ${err.message}`);
        });
    });
}

/**
 * Render Video Case Report Results & Timeline Scrubber
 */
function renderVideoResult(data) {
    document.getElementById('result-container').classList.remove('hidden');
    currentTimelineData = data.timeline_data || [];

    if (data.is_demo) {
        document.getElementById('demo-mode-badge').classList.remove('hidden');
    }

    // Header Case Info
    document.getElementById('case-id-display').textContent = `CASE DS-${data.id.toUpperCase()}`;

    const verdictEl = document.getElementById('verdict-badge');
    verdictEl.textContent = data.verdict === 'FAKE' ? 'LIKELY MANIPULATED' : (data.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'LIKELY AUTHENTIC');
    verdictEl.className = `badge-verdict ${data.verdict.toLowerCase()}`;

    const scorePct = (data.fusion_score * 100).toFixed(1) + '%';
    document.getElementById('fusion-score-val').textContent = scorePct;
    document.getElementById('fusion-recap-val').textContent = scorePct;

    const barFill = document.getElementById('confidence-bar-fill');
    barFill.style.width = scorePct;
    barFill.className = `confidence-bar-fill ${data.verdict.toLowerCase()}`;

    // Sub-signal scores
    document.getElementById('cnn-score-val').textContent = (data.cnn_score * 100).toFixed(1) + '%';
    setSeverityPill('cnn-severity-pill', data.cnn_score);

    document.getElementById('fft-score-val').textContent = (data.fft_score * 100).toFixed(1) + '%';
    setSeverityPill('fft-severity-pill', data.fft_score);

    document.getElementById('blink-score-val').textContent = (data.blink_score * 100).toFixed(1) + '%';
    setSeverityPill('blink-severity-pill', data.blink_score);

    // Explanation
    document.getElementById('explanation-text').textContent = data.explanation;

    // PDF Link
    document.getElementById('btn-download-pdf').href = `/api/report/${data.id}`;

    // Peak Frame Image & Inspector initial load
    if (data.peak_frame_filename) {
        document.getElementById('peak-frame-img').src = `/results/${data.peak_frame_filename}`;
    }

    updateInspectorFrame({
        frame_number: data.peak_frame,
        timestamp: data.peak_timestamp,
        fusion_score: data.peak_score,
        cnn_score: data.cnn_score,
        fft_score: data.fft_score,
        blink_score: data.blink_score
    });

    // Render Time Segments Badges
    const segContainer = document.getElementById('segments-list');
    segContainer.innerHTML = '';
    if (data.suspicious_segments && data.suspicious_segments.length > 0) {
        data.suspicious_segments.forEach(seg => {
            const span = document.createElement('span');
            span.style.cssText = 'display: inline-block; background-color: var(--accent-red-bg); border: 1px solid var(--accent-red); color: var(--accent-red); padding: 2px 8px; font-family: var(--font-mono); font-size: 0.7rem; margin-right: 6px; margin-bottom: 4px;';
            span.textContent = `[ SEGMENT: ${seg.start} → ${seg.end} (Frames ${seg.start_frame}-${seg.end_frame}) ]`;
            segContainer.appendChild(span);
        });
    } else {
        segContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.7rem;">NO CONTINUOUS TIME SEGMENT EXCEEDED 0.70 THRESHOLD</span>';
    }

    // Render SIGNATURE VIDEO TIMELINE SCRUBBER & MARKERS
    renderTimelineScrubber(data);

    // Render Chart.js Timeline Graph
    renderTimelineChart(data.timeline_data);
}

/**
 * SIGNATURE UI ELEMENT: Professional Video Timeline Scrubber with Markers
 */
function renderTimelineScrubber(data) {
    const track = document.getElementById('timeline-track-container');
    if (!track) return;

    // Remove old markers
    const oldMarkers = track.querySelectorAll('.timeline-marker');
    oldMarkers.forEach(m => m.remove());

    const totalDuration = data.duration || 1.0;
    const timelineData = data.timeline_data || [];

    // Filter frames with suspicion >= 0.70 (or top peak frames)
    const suspiciousFrames = timelineData.filter(d => d.fusion_score >= 0.70);

    // If no frame >= 0.70, show peak frame marker
    const framesToMark = suspiciousFrames.length > 0 ? suspiciousFrames : timelineData.filter(d => d.frame_number === data.peak_frame);

    framesToMark.forEach(rec => {
        const pctPos = Math.max(2, Math.min(98, (rec.timestamp / totalDuration) * 100));

        const marker = document.createElement('div');
        marker.className = 'timeline-marker';
        marker.style.left = `${pctPos}%`;

        const label = document.createElement('div');
        label.className = 'timeline-marker-label';
        label.textContent = `#${rec.frame_number} (${rec.timestamp}s)`;
        marker.appendChild(label);

        marker.addEventListener('click', () => {
            // Unselect all markers
            track.querySelectorAll('.timeline-marker').forEach(m => m.classList.remove('selected'));
            marker.classList.add('selected');

            // Find gallery image or peak image for this frame if available
            let frameImgFilename = data.peak_frame_filename;
            if (data.top_gallery) {
                const galleryItem = data.top_gallery.find(g => g.frame_number === rec.frame_number);
                if (galleryItem) frameImgFilename = galleryItem.image_filename;
            }

            if (frameImgFilename) {
                document.getElementById('peak-frame-img').src = `/results/${frameImgFilename}`;
            }

            updateInspectorFrame(rec);
        });

        track.appendChild(marker);
    });
}

/**
 * Updates Frame Inspector Metadata Table
 */
function updateInspectorFrame(rec) {
    document.getElementById('inspector-frame-label').textContent = `FRAME #${rec.frame_number}`;
    document.getElementById('inspector-frame-no').textContent = rec.frame_number;
    document.getElementById('inspector-timestamp').textContent = `${rec.timestamp}s`;

    const fScorePct = (rec.fusion_score * 100).toFixed(1) + '%';
    document.getElementById('inspector-fusion-score').textContent = fScorePct;

    document.getElementById('inspector-cnn-score').textContent = (rec.cnn_score * 100).toFixed(1) + '%';
    document.getElementById('inspector-fft-score').textContent = (rec.fft_score * 100).toFixed(1) + '%';
    document.getElementById('inspector-blink-score').textContent = rec.blink_score ? (rec.blink_score * 100).toFixed(1) + '%' : 'N/A';

    const sevEl = document.getElementById('inspector-severity');
    if (rec.fusion_score >= 0.70) {
        sevEl.textContent = 'HIGH ANOMALY';
        sevEl.className = 'badge-verdict fake';
    } else if (rec.fusion_score >= 0.45) {
        sevEl.textContent = 'MODERATE SUSPICION';
        sevEl.className = 'badge-verdict suspicious';
    } else {
        sevEl.textContent = 'NORMAL';
        sevEl.className = 'badge-verdict real';
    }
}

function setSeverityPill(elementId, score) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (score >= 0.70) {
        el.textContent = 'HIGH';
        el.className = 'severity-pill high';
    } else if (score >= 0.45) {
        el.textContent = 'MEDIUM';
        el.className = 'severity-pill medium';
    } else {
        el.textContent = 'NORMAL';
        el.className = 'severity-pill normal';
    }
}

/**
 * Render Chart.js Suspicion Line Graph
 */
function renderTimelineChart(timelineData) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    if (timelineChartInstance) {
        timelineChartInstance.destroy();
    }

    const labels = timelineData.map(d => `${d.timestamp}s`);
    const fusionScores = timelineData.map(d => d.fusion_score);
    const thresholdLine = timelineData.map(() => 0.70);

    timelineChartInstance = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Fusion Suspicion Score',
                    data: fusionScores,
                    borderColor: '#CF443B',
                    backgroundColor: 'rgba(207, 68, 59, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 3,
                    pointHoverRadius: 6
                },
                {
                    label: 'Threshold (0.70)',
                    data: thresholdLine,
                    borderColor: '#D9822B',
                    borderDash: [4, 4],
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#8B929E', font: { family: 'JetBrains Mono', size: 10 } } }
            },
            scales: {
                x: {
                    grid: { color: '#2A2D31' },
                    ticks: { color: '#8B929E', font: { family: 'JetBrains Mono', size: 10 } }
                },
                y: {
                    min: 0,
                    max: 1.0,
                    grid: { color: '#2A2D31' },
                    ticks: { color: '#8B929E', font: { family: 'JetBrains Mono', size: 10 } }
                }
            }
        }
    });
}

/**
 * Loads Case History Registry Table
 */
function loadHistoryRecords() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    fetch('/api/history')
    .then(res => res.json())
    .then(data => {
        tbody.innerHTML = '';
        if (!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align: center; padding: 1.5rem;">No archived cases found.</td></tr>';
            return;
        }

        data.forEach(rec => {
            const tr = document.createElement('tr');
            const verdictClass = rec.verdict === 'FAKE' ? 'fake' : (rec.verdict === 'SUSPICIOUS' ? 'suspicious' : 'real');
            const verdictLabel = rec.verdict === 'FAKE' ? 'LIKELY MANIPULATED' : (rec.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'AUTHENTIC');
            const scorePct = (rec.fusion_score * 100).toFixed(1) + '%';
            const peakStr = rec.peak_frame ? `Frame #${rec.peak_frame} (${rec.peak_timestamp}s)` : 'N/A';

            tr.innerHTML = `
                <td><strong>DS-${rec.id.toUpperCase()}</strong></td>
                <td>${rec.timestamp || 'N/A'}</td>
                <td>${rec.filename}</td>
                <td>${(rec.media_type || 'image').toUpperCase()}</td>
                <td><span class="badge-verdict ${verdictClass}">${verdictLabel}</span></td>
                <td><strong>${scorePct}</strong></td>
                <td>${peakStr}</td>
                <td>
                    <a href="/api/report/${rec.id}" target="_blank" class="btn-forensic btn-sm">PDF Report</a>
                </td>
            `;
            tbody.appendChild(tr);
        });
    })
    .catch(err => {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--accent-red); padding: 1.5rem;">Error loading archive: ${err.message}</td></tr>`;
    });
}
