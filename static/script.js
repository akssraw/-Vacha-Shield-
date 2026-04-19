/**
 * VACHA-SHIELD 2.0 — Premium Hackathon UI
 * Main Application Script
 */

(function () {
    'use strict';

    // ===== STATE =====
    const state = {
        selectedFile: null,
        recordedBlob: null,
        isRecording: false,
        mediaRecorder: null,
        audioChunks: [],
        analysisProfile: 'strict',
        socket: null,
        monitorActive: false,
        monitorStream: null,
        monitorAudioCtx: null,
        monitorProcessor: null,
        waveformAnimId: null,
        waveformData: new Float32Array(256),
    };

    // ===== DOM REFS =====
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const DOM = {
        viewUpload: $('#view-upload'),
        viewResults: $('#view-results'),
        viewMonitor: $('#view-monitor'),
        loader: $('#loader-overlay'),

        // Upload
        dropZone: $('#drop-zone'),
        fileInput: $('#audio-input'),
        browseBtn: $('#browse-btn'),
        fileInfo: $('#file-info'),
        fileName: $('#file-name'),
        fileMeta: $('#file-meta'),
        analyzeBtn: $('#analyze-btn'),
        recordBtn: $('#record-btn'),
        recordingStatus: $('#recording-status'),
        demoToggle: $('#demo-mode-toggle'),

        // Profile
        profileTabs: $$('.profile-tab'),

        // Results
        alertDanger: $('#alert-danger'),
        alertSafe: $('#alert-safe'),
        alertWarning: $('#alert-warning'),
        alertFraud: $('#alert-fraud'),
        gaugeFill: $('#gauge-fill'),
        gaugeValue: $('#gauge-value'),
        humanScore: $('#human-score'),
        humanBar: $('#human-bar'),
        synthScore: $('#synth-score'),
        synthBar: $('#synth-bar'),
        modelScore: $('#model-score'),
        modelBar: $('#model-bar'),
        artifactScore: $('#artifact-score'),
        artifactBar: $('#artifact-bar'),
        fraudScore: $('#fraud-score'),
        fraudBar: $('#fraud-bar'),
        keywordPills: $('#keyword-pills'),
        spectrogramImg: $('#spectrogram-img'),
        spectrogramPlaceholder: $('#spectrogram-placeholder'),
        verdictBox: $('#verdict-box'),
        verdictIcon: $('#verdict-icon'),
        verdictTitle: $('#verdict-title'),
        verdictSummary: $('#verdict-summary'),
        transcriptPreview: $('#transcript-preview'),
        detailProfile: $('#detail-profile'),
        detailThreshold: $('#detail-threshold'),
        detailChunks: $('#detail-chunks'),
        detailMode: $('#detail-mode'),
        resetBtn: $('#reset-btn'),
        feedbackSection: $('#feedback-section'),
        feedbackButtons: $('#feedback-buttons'),
        feedbackThanks: $('#feedback-thanks'),
        btnFeedbackHuman: $('#btn-feedback-human'),
        btnFeedbackAi: $('#btn-feedback-ai'),

        // Monitor
        startMonitorBtn: $('#start-monitor-btn'),
        stopMonitorBtn: $('#stop-monitor-btn'),
        liveBadge: $('#live-badge'),
        waveformCanvas: $('#waveform-canvas'),
        monitorVoiceCard: $('#monitor-voice-card'),
        monitorVoiceStatus: $('#monitor-voice-status'),
        monitorVoiceSub: $('#monitor-voice-sub'),
        monitorVoiceBar: $('#monitor-voice-bar'),
        monitorScamCard: $('#monitor-scam-card'),
        monitorScamStatus: $('#monitor-scam-status'),
        monitorScamSub: $('#monitor-scam-sub'),
        monitorKeywords: $('#monitor-keywords'),
        monitorTranscript: $('#monitor-transcript'),
        threatRingFill: $('#threat-ring-fill'),
        threatRingValue: $('#threat-ring-value'),
        statSampleRate: $('#stat-sample-rate'),
        statBuffer: $('#stat-buffer'),
        statLatency: $('#stat-latency'),
        statProcessing: $('#stat-processing'),
        statVerdict: $('#stat-verdict'),

        // Footer
        modelStatus: $('#model-status'),
        deviceBadge: $('#device-badge'),
        footerDevice: $('#footer-device'),
    };

    // ===== INIT =====
    function init() {
        checkHealth();
        bindUploadEvents();
        bindProfileTabs();
        bindResultsEvents();
        bindMonitorEvents();
        initSocket();
    }

    // ===== HEALTH CHECK =====
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            if (data.model_loaded) {
                DOM.modelStatus.innerHTML = '<span class="dot"></span><span>Model Ready</span>';
                DOM.modelStatus.classList.remove('offline');
            } else {
                DOM.modelStatus.innerHTML = '<span class="dot"></span><span>Model Error</span>';
                DOM.modelStatus.classList.add('offline');
            }
            const device = data.device || 'CPU';
            DOM.deviceBadge.textContent = device;
            DOM.footerDevice.textContent = device;

            // Set the default profile from the server
            if (data.default_analysis_profile) {
                setActiveProfile(data.default_analysis_profile);
            }
        } catch (e) {
            DOM.modelStatus.innerHTML = '<span class="dot"></span><span>Offline</span>';
            DOM.modelStatus.classList.add('offline');
        }
    }

    // ===== VIEW SWITCHING =====
    function showView(view) {
        [DOM.viewUpload, DOM.viewResults, DOM.viewMonitor].forEach(v => v.classList.remove('active'));
        view.classList.add('active');
    }

    function showLoader(show) {
        if (show) {
            DOM.loader.classList.add('active');
        } else {
            DOM.loader.classList.remove('active');
        }
    }

    // ===== UPLOAD & FILE HANDLING =====
    function bindUploadEvents() {
        // Browse button
        DOM.browseBtn.addEventListener('click', () => DOM.fileInput.click());

        // File input change
        DOM.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectFile(e.target.files[0]);
            }
        });

        // Drop zone
        DOM.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.add('dragover');
        });
        DOM.dropZone.addEventListener('dragleave', () => {
            DOM.dropZone.classList.remove('dragover');
        });
        DOM.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                selectFile(e.dataTransfer.files[0]);
            }
        });
        DOM.dropZone.addEventListener('click', (e) => {
            if (e.target.closest('.btn')) return;
            DOM.fileInput.click();
        });

        // Analyze button
        DOM.analyzeBtn.addEventListener('click', analyzeFile);

        // Record button
        DOM.recordBtn.addEventListener('click', toggleRecording);
    }

    function selectFile(file) {
        state.selectedFile = file;
        state.recordedBlob = null;
        DOM.fileName.textContent = file.name;
        DOM.fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB • ${file.type || 'audio'}`;
        DOM.fileInfo.classList.remove('hidden');
    }

    // ===== RECORDING =====
    async function toggleRecording() {
        if (state.isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            state.audioChunks = [];
            state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            state.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) state.audioChunks.push(e.data);
            };

            state.mediaRecorder.onstop = () => {
                const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
                state.recordedBlob = blob;
                state.selectedFile = null;
                stream.getTracks().forEach(t => t.stop());

                DOM.fileName.textContent = 'recorded_audio.webm';
                DOM.fileMeta.textContent = `${(blob.size / 1024).toFixed(1)} KB • Recorded`;
                DOM.fileInfo.classList.remove('hidden');
            };

            state.mediaRecorder.start(250);
            state.isRecording = true;
            DOM.recordBtn.classList.add('recording');
            DOM.recordingStatus.classList.remove('hidden');
        } catch (err) {
            alert('Microphone access denied. Please allow microphone permissions.');
        }
    }

    function stopRecording() {
        if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
            state.mediaRecorder.stop();
        }
        state.isRecording = false;
        DOM.recordBtn.classList.remove('recording');
        DOM.recordingStatus.classList.add('hidden');
    }

    // ===== ANALYSIS =====
    async function analyzeFile() {
        const formData = new FormData();
        if (state.selectedFile) {
            formData.append('file', state.selectedFile);
        } else if (state.recordedBlob) {
            formData.append('file', state.recordedBlob, 'recorded_audio.webm');
        } else {
            return;
        }

        formData.append('analysis_profile', state.analysisProfile);
        formData.append('enable_transcript_analysis', 'true');

        if (DOM.demoToggle.checked) {
            formData.append('force_alert', 'true');
        }

        showLoader(true);

        try {
            const res = await fetch('/detect_voice', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) {
                alert('Analysis error: ' + data.error);
                showLoader(false);
                return;
            }

            showLoader(false);
            renderResults(data);
            showView(DOM.viewResults);
        } catch (err) {
            showLoader(false);
            alert('Failed to connect to the server. Is the backend running?');
        }
    }

    // ===== RESULTS RENDERING =====
    function renderResults(data) {
        const synthProb = data.synthetic_probability || 0;
        const humanProb = data.human_probability || 0;
        const modelProb = data.model_probability || 0;
        const artifactProb = data.artifact_probability || 0;
        const fraudProb = data.fraud_language_probability || 0;
        const alert = data.alert || false;
        const verdict = data.verdict || 'human';

        // Reset alert banners
        [DOM.alertDanger, DOM.alertSafe, DOM.alertWarning, DOM.alertFraud].forEach(b => b.classList.remove('visible'));

        // Show appropriate banner
        if (verdict === 'fraud_language') {
            DOM.alertFraud.classList.add('visible');
        } else if (alert) {
            DOM.alertDanger.classList.add('visible');
        } else if (verdict === 'borderline_human') {
            DOM.alertWarning.classList.add('visible');
        } else {
            DOM.alertSafe.classList.add('visible');
        }

        // Gauge
        animateGauge(synthProb);

        // Progress bars
        animateBar(DOM.humanBar, DOM.humanScore, humanProb, 'safe');
        animateBar(DOM.synthBar, DOM.synthScore, synthProb, synthProb > 0.5 ? 'danger' : 'neutral');
        animateBar(DOM.modelBar, DOM.modelScore, modelProb, 'neutral');
        animateBar(DOM.artifactBar, DOM.artifactScore, artifactProb, 'neutral');
        animateBar(DOM.fraudBar, DOM.fraudScore, fraudProb, fraudProb > 0.4 ? 'danger' : 'neutral');

        // Verdict box
        updateVerdict(verdict, data.decision_summary);

        // Spectrogram
        if (data.spectrogram_base64) {
            DOM.spectrogramImg.src = data.spectrogram_base64;
            DOM.spectrogramImg.style.display = 'block';
            DOM.spectrogramPlaceholder.style.display = 'none';
        } else {
            DOM.spectrogramImg.style.display = 'none';
            DOM.spectrogramPlaceholder.style.display = 'flex';
        }

        // Keywords
        renderKeywords(DOM.keywordPills, data.fraud_keywords || []);

        // Transcript
        if (data.transcript_preview) {
            DOM.transcriptPreview.textContent = data.transcript_preview;
        } else {
            DOM.transcriptPreview.innerHTML = '<span style="color: var(--text-muted)">No transcript available</span>';
        }

        // Details
        DOM.detailProfile.textContent = data.analysis_profile || state.analysisProfile;
        DOM.detailThreshold.textContent = (data.threshold || 0).toFixed(4);
        DOM.detailChunks.textContent = data.chunk_count || '—';
        DOM.detailMode.textContent = data.decision_mode || '—';

        // Reset feedback
        DOM.feedbackButtons.style.display = 'flex';
        DOM.feedbackThanks.classList.remove('visible');
    }

    function animateGauge(value) {
        const pct = Math.round(value * 100);
        const arcLength = 251.2; // Approximate arc length
        const offset = arcLength * (1 - value);
        const color = value > 0.6 ? 'var(--danger)' : value > 0.4 ? 'var(--warning)' : 'var(--safe)';

        DOM.gaugeFill.style.stroke = color;
        // Delay for animation
        requestAnimationFrame(() => {
            DOM.gaugeFill.style.strokeDashoffset = offset;
        });
        DOM.gaugeValue.textContent = pct + '%';
    }

    function animateBar(barEl, scoreEl, value, colorClass) {
        const pct = (value * 100).toFixed(2);
        scoreEl.textContent = pct + '%';
        scoreEl.className = 'metric-value ' + colorClass;
        requestAnimationFrame(() => {
            barEl.style.width = pct + '%';
        });
    }

    function updateVerdict(verdict, summary) {
        const classes = { human: 'safe', ai_clone: 'danger', borderline_human: 'warning', fraud_language: 'danger' };
        const titles = { human: 'VERIFIED HUMAN', ai_clone: 'AI VOICE CLONE', borderline_human: 'BORDERLINE', fraud_language: 'FRAUD DETECTED' };
        const icons = {
            human: '<polyline points="20 6 9 17 4 12"/>',
            ai_clone: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
            borderline_human: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
            fraud_language: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        };

        const cls = classes[verdict] || 'safe';
        DOM.verdictBox.className = 'verdict-box ' + cls;
        DOM.verdictIcon.className = 'verdict-icon ' + cls;
        DOM.verdictIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">' + (icons[verdict] || icons.human) + '</svg>';
        DOM.verdictTitle.textContent = titles[verdict] || verdict.toUpperCase();
        DOM.verdictSummary.textContent = summary || '';
    }

    function renderKeywords(container, keywords) {
        if (!keywords || keywords.length === 0) {
            container.innerHTML = '<span class="keyword-pill safe">No keywords detected</span>';
            return;
        }
        const HIGH_RISK = ['otp', 'pin', 'cvv', 'gift_card', 'remote_access', 'urgent_payment', 'transaction_blocked'];
        container.innerHTML = keywords.map(kw => {
            const cls = HIGH_RISK.includes(kw) ? 'danger' : 'warning';
            return `<span class="keyword-pill ${cls}">${kw.replace(/_/g, ' ')}</span>`;
        }).join('');
    }

    // ===== PROFILE TABS =====
    function bindProfileTabs() {
        DOM.profileTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                setActiveProfile(tab.dataset.profile);
            });
        });
    }

    function setActiveProfile(profile) {
        state.analysisProfile = profile;
        DOM.profileTabs.forEach(t => {
            t.classList.toggle('active', t.dataset.profile === profile);
        });
    }

    // ===== RESULTS EVENTS =====
    function bindResultsEvents() {
        DOM.resetBtn.addEventListener('click', () => {
            state.selectedFile = null;
            state.recordedBlob = null;
            DOM.fileInfo.classList.add('hidden');
            DOM.fileInput.value = '';
            showView(DOM.viewUpload);
        });

        DOM.btnFeedbackHuman.addEventListener('click', () => submitFeedback('human'));
        DOM.btnFeedbackAi.addEventListener('click', () => submitFeedback('ai'));
    }

    async function submitFeedback(label) {
        const formData = new FormData();
        if (state.selectedFile) {
            formData.append('file', state.selectedFile);
        } else if (state.recordedBlob) {
            formData.append('file', state.recordedBlob, 'recorded_audio.webm');
        } else {
            return;
        }
        formData.append('label', label);

        try {
            await fetch('/feedback', { method: 'POST', body: formData });
        } catch (e) { /* silent */ }

        DOM.feedbackButtons.style.display = 'none';
        DOM.feedbackThanks.classList.add('visible');
    }

    // ===== SOCKET.IO =====
    function initSocket() {
        state.socket = io({ transports: ['websocket', 'polling'] });

        state.socket.on('call_monitor_ready', (data) => {
            if (data && data.device) {
                DOM.deviceBadge.textContent = data.device;
                DOM.footerDevice.textContent = data.device;
            }
        });

        state.socket.on('call_monitor_status', (data) => {
            // status update
        });

        state.socket.on('call_monitor_result', (data) => {
            updateMonitorUI(data);
        });

        state.socket.on('call_monitor_error', (data) => {
            console.error('Monitor error:', data);
        });
    }

    // ===== LIVE MONITOR =====
    function bindMonitorEvents() {
        DOM.startMonitorBtn.addEventListener('click', startLiveMonitor);
        DOM.stopMonitorBtn.addEventListener('click', stopLiveMonitor);
    }

    async function startLiveMonitor() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
            });
            state.monitorStream = stream;
            state.monitorAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            const source = state.monitorAudioCtx.createMediaStreamSource(stream);
            const analyser = state.monitorAudioCtx.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);

            // ScriptProcessor for sending audio data
            const bufferSize = 4096;
            state.monitorProcessor = state.monitorAudioCtx.createScriptProcessor(bufferSize, 1, 1);
            source.connect(state.monitorProcessor);
            state.monitorProcessor.connect(state.monitorAudioCtx.destination);

            state.monitorProcessor.onaudioprocess = (e) => {
                if (!state.monitorActive) return;
                const audioData = e.inputBuffer.getChannelData(0);
                const chunk = new Float32Array(audioData);
                state.socket.emit('call_monitor_chunk', {
                    audio: chunk.buffer,
                    captured_at_ms: Date.now()
                });
            };

            // Start socket monitor
            state.socket.emit('call_monitor_start', {
                sample_rate: state.monitorAudioCtx.sampleRate,
                analysis: { analysis_profile: state.analysisProfile }
            });

            state.monitorActive = true;
            state.statSampleRate = state.monitorAudioCtx.sampleRate;
            DOM.statSampleRate.textContent = (state.monitorAudioCtx.sampleRate / 1000).toFixed(0) + 'kHz';

            showView(DOM.viewMonitor);
            startWaveformAnimation(analyser);

        } catch (err) {
            alert('Microphone access denied or unavailable.');
        }
    }

    function stopLiveMonitor() {
        state.monitorActive = false;
        state.socket.emit('call_monitor_stop');

        if (state.monitorProcessor) {
            state.monitorProcessor.disconnect();
            state.monitorProcessor = null;
        }
        if (state.monitorAudioCtx) {
            state.monitorAudioCtx.close();
            state.monitorAudioCtx = null;
        }
        if (state.monitorStream) {
            state.monitorStream.getTracks().forEach(t => t.stop());
            state.monitorStream = null;
        }
        if (state.waveformAnimId) {
            cancelAnimationFrame(state.waveformAnimId);
            state.waveformAnimId = null;
        }

        showView(DOM.viewUpload);
    }

    function updateMonitorUI(data) {
        const synthProb = data.synthetic_probability || 0;
        const pct = Math.round(synthProb * 100);
        const isAlert = data.alert || false;
        const fraudAlert = data.fraud_language_alert || false;
        const color = synthProb > 0.6 ? 'var(--danger)' : synthProb > 0.4 ? 'var(--warning)' : 'var(--safe)';

        // Threat ring
        const circumference = 408.4;
        const offset = circumference * (1 - synthProb);
        DOM.threatRingFill.style.strokeDashoffset = offset;
        DOM.threatRingFill.style.stroke = color;
        DOM.threatRingValue.textContent = pct + '%';

        // Voice card
        DOM.monitorVoiceStatus.textContent = isAlert ? 'ALERT' : synthProb > 0.4 ? 'CAUTION' : 'SAFE';
        DOM.monitorVoiceStatus.style.color = color;
        DOM.monitorVoiceSub.textContent = `Synthetic: ${pct}%`;
        DOM.monitorVoiceBar.style.width = pct + '%';
        DOM.monitorVoiceBar.className = 'progress-fill ' + (synthProb > 0.6 ? 'danger-fill' : synthProb > 0.4 ? 'warning-fill' : 'safe-fill');
        DOM.monitorVoiceCard.classList.toggle('alert', isAlert);

        // Scam card
        const fraudProb = data.fraud_language_probability || 0;
        const fraudPct = Math.round(fraudProb * 100);
        DOM.monitorScamStatus.textContent = fraudAlert ? 'THREAT DETECTED' : fraudProb > 0.3 ? 'CAUTION' : 'NO THREATS';
        DOM.monitorScamStatus.style.color = fraudAlert ? 'var(--danger)' : fraudProb > 0.3 ? 'var(--warning)' : 'var(--safe)';
        DOM.monitorScamSub.textContent = `Fraud Score: ${fraudPct}%`;
        DOM.monitorScamCard.classList.toggle('alert', fraudAlert);
        renderKeywords(DOM.monitorKeywords, data.fraud_keywords || []);

        // Transcript
        if (data.transcript_preview) {
            DOM.monitorTranscript.textContent = data.transcript_preview;
        }

        // Stats
        if (data.buffered_seconds !== undefined) {
            DOM.statBuffer.textContent = data.buffered_seconds.toFixed(1) + 's';
        }
        if (data.latency_ms !== undefined) {
            DOM.statLatency.textContent = Math.round(data.latency_ms) + 'ms';
        }
        if (data.processing_ms !== undefined) {
            DOM.statProcessing.textContent = Math.round(data.processing_ms) + 'ms';
        }
        DOM.statVerdict.textContent = (data.verdict || '—').replace(/_/g, ' ').toUpperCase();
        DOM.statVerdict.style.color = isAlert ? 'var(--danger)' : 'var(--safe)';
    }

    // ===== WAVEFORM ANIMATION =====
    function startWaveformAnimation(analyser) {
        const canvas = DOM.waveformCanvas;
        const ctx = canvas.getContext('2d');

        function resize() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = 120 * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        resize();
        window.addEventListener('resize', resize);

        const bufferLen = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLen);

        function draw() {
            if (!state.monitorActive) return;
            state.waveformAnimId = requestAnimationFrame(draw);

            analyser.getByteTimeDomainData(dataArray);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;

            ctx.clearRect(0, 0, w, h);

            // Background glow
            const gradient = ctx.createLinearGradient(0, 0, w, 0);
            gradient.addColorStop(0, 'rgba(0, 242, 254, 0.8)');
            gradient.addColorStop(0.5, 'rgba(79, 172, 254, 0.8)');
            gradient.addColorStop(1, 'rgba(124, 58, 237, 0.8)');

            ctx.lineWidth = 2;
            ctx.strokeStyle = gradient;
            ctx.shadowColor = 'rgba(0, 242, 254, 0.5)';
            ctx.shadowBlur = 10;

            ctx.beginPath();
            const sliceWidth = w / bufferLen;
            let x = 0;

            for (let i = 0; i < bufferLen; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * h) / 2;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.lineTo(w, h / 2);
            ctx.stroke();

            // Zero line
            ctx.shadowBlur = 0;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
        }

        draw();
    }

    // ===== BOOT =====
    document.addEventListener('DOMContentLoaded', init);

})();
