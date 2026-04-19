document.addEventListener('DOMContentLoaded', () => {

    const recordBtn = document.getElementById('record-btn');
    const radarContainer = document.querySelector('.radar-container');
    const navStatus = document.getElementById('nav-status');
    const liveDashboard = document.getElementById('live-dashboard');

    // UI Elements
    const analysisStatus = document.getElementById('analysis-status');
    const barHuman = document.getElementById('bar-human');
    const barSynth = document.getElementById('bar-synth');
    const metricHuman = document.getElementById('metric-human');
    const metricSynth = document.getElementById('metric-synth');

    // Global Alert UI
    const alertOverlay = document.getElementById('alert-overlay');
    const overlayProb = document.getElementById('overlay-prob');
    const dismissBtn = document.getElementById('dismiss-btn');

    let isMonitoring = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let monitorInterval = null;
    let wakeLock = null; // Prevent screen from sleeping

    // --- Hackathon Presentation Mode ---
    let demoMode = false;
    let clickCount = 0;

    const handleSecretTap = (e) => {
        // Only prevent default on touch to stop double-firing with click
        if (e.type === 'touchstart') e.preventDefault();

        clickCount++;
        if (clickCount >= 3) {
            demoMode = !demoMode;
            clickCount = 0;
            // Visual indicator: Logo turns red when armed!
            if (demoMode) {
                document.querySelector('.app-logo svg').setAttribute('stroke', '#ff4d4d');
            } else {
                document.querySelector('.app-logo svg').setAttribute('stroke', '#00f2fe');
            }
        }
    };

    const logoEl = document.querySelector('.app-logo');
    logoEl.addEventListener('click', handleSecretTap);
    logoEl.addEventListener('touchstart', handleSecretTap, { passive: false });

    // Audio Visualizer Contexts
    let audioContext = null;
    let analyser = null;
    let animationId = null;

    // Configuration
    const CHUNK_TIME_MS = 5000; // 5 seconds

    // Request Screen Wake Lock
    async function requestWakeLock() {
        if ('wakeLock' in navigator) {
            try {
                wakeLock = await navigator.wakeLock.request('screen');
                console.log('Screen Wake Lock acquired.');

                // Re-acquire if page visibility changes (like pulling down notification shade)
                document.addEventListener('visibilitychange', async () => {
                    if (wakeLock !== null && document.visibilityState === 'visible') {
                        wakeLock = await navigator.wakeLock.request('screen');
                    }
                });
            } catch (err) {
                console.error(`Wake Lock failed: ${err.name}, ${err.message}`);
            }
        }
    }

    function releaseWakeLock() {
        if (wakeLock !== null) {
            wakeLock.release()
                .then(() => {
                    wakeLock = null;
                    console.log('Screen Wake Lock released.');
                });
        }
    }

    // --- Phase 12: Simulated OS Call Routing ---
    const standbyView = document.getElementById('standby-view');
    const activeView = document.getElementById('active-view');
    const simCallBtn = document.getElementById('sim-call-btn');
    const callTypeSubtitle = document.getElementById('call-type-subtitle');

    simCallBtn.addEventListener('click', async () => {
        standbyView.classList.add('hidden');
        activeView.classList.remove('hidden');
        callTypeSubtitle.textContent = "Secure Incoming Call";
        await startMonitoring();
    });

    // End Call disconnects microphone and goes to Standby
    recordBtn.addEventListener('click', () => {
        stopMonitoring();
        activeView.classList.add('hidden');
        standbyView.classList.remove('hidden');
    });

    async function startMonitoring() {
        try {
            // Request Native Microphone Permission
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStream = stream; // Save to global so we can mathematically kill it on End Call

            isMonitoring = true;
            radarContainer.classList.add('active');
            navStatus.textContent = 'MONITORING';
            navStatus.classList.add('live');
            liveDashboard.classList.remove('hidden');
            document.getElementById('viz-container').classList.remove('hidden');
            document.querySelector('.btn-text').innerHTML = "STOP<br>MONITOR";
            document.querySelector('.btn-subtext').textContent = "Tap to disconnect";

            // Prevent screen from going to sleep during the call
            await requestWakeLock();

            // Setup Visualizer
            setupVisualizer(stream);

            // Initialize Recorder
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = []; // Clear sequence

                if (isMonitoring) {
                    analysisStatus.textContent = "Analyzing segment...";
                    await analyzeSegment(audioBlob);
                    // Start next segment immediately
                    mediaRecorder.start();
                    setTimeout(() => {
                        if (mediaRecorder.state === 'recording') mediaRecorder.stop();
                    }, CHUNK_TIME_MS);
                }
            };

            // Kick off first cycle
            mediaRecorder.start();
            setTimeout(() => {
                if (mediaRecorder.state === 'recording') mediaRecorder.stop();
            }, CHUNK_TIME_MS);

        } catch (err) {
            console.error("Microphone Access Denied:", err);
            alert("Vacha-Shield requires microphone access to monitor calls.");
        }
    }

    function stopMonitoring() {
        isMonitoring = false;

        // Auto-Disarm Demo Mode to prevent infinite false-positives
        demoMode = false;
        clickCount = 0;
        document.querySelector('.app-logo svg').setAttribute('stroke', '#00f2fe');

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        // --- CRITICAL PRIVACY PATCH: Explicitly Kill Hardware Microphone ---
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
            console.log("Privacy Enforced: Microphone hardware completely disconnected.");
        }

        // Release the screen wake lock
        releaseWakeLock();

        // Stop Visualizer
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // UI Reset
        radarContainer.classList.remove('active');
        navStatus.textContent = 'OFFLINE';
        navStatus.classList.remove('live');
        document.querySelector('.btn-text').innerHTML = "START<br>MONITOR";
        document.querySelector('.btn-subtext').textContent = "Tap to connect";
        analysisStatus.textContent = "System offline.";
        liveDashboard.classList.add('hidden');
        document.getElementById('viz-container').classList.add('hidden');
    }

    // --- Audio Visualizer Logic ---
    function setupVisualizer(stream) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        const canvas = document.getElementById('audio-visualizer');
        const canvasCtx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        // Resize Canvas nicely
        canvas.width = canvas.parentElement.clientWidth;

        function drawVisualizer() {
            if (!isMonitoring) return;
            animationId = requestAnimationFrame(drawVisualizer);

            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = 'rgba(10, 14, 23, 1)'; // Matches #0a0e17 background
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = '#00f2fe';

            canvasCtx.beginPath();

            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvas.height / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
        }

        drawVisualizer();
    }
    // ------------------------------

    async function analyzeSegment(audioBlob) {
        const formData = new FormData();
        // Flask expects 'file' named 'ambient.webm'
        formData.append('file', audioBlob, 'ambient.webm');

        // Pass the Presentation Mode forced flag
        if (demoMode) {
            formData.append('force_alert', 'true');
        }

        try {
            const response = await fetch('/detect_voice', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                updateDashboard(data);
            }
        } catch (error) {
            console.error("Inference Error:", error);
            analysisStatus.textContent = "Connection interrupted.";
        }
    }

    function updateDashboard(data) {
        const h_pct = (data.human_probability * 100).toFixed(1);
        const s_pct = (data.synthetic_probability * 100).toFixed(1);

        metricHuman.textContent = `${h_pct}%`;
        metricSynth.textContent = `${s_pct}%`;

        barHuman.style.width = `${h_pct}%`;
        barSynth.style.width = `${s_pct}%`;

        const syn_prob = data.synthetic_probability;

        // 3-Category Classification
        if (syn_prob < 0.40) {
            analysisStatus.textContent = "Human Voice Detected";
            analysisStatus.style.color = "#10b981"; // Safe Green
        } else if (syn_prob >= 0.40 && syn_prob <= 0.75) {
            analysisStatus.innerHTML = "AI Assistant Detected<br><span style='font-size: 0.8em; opacity: 0.8;'>Likely customer support bot</span>";
            analysisStatus.style.color = "#fbbf24"; // Warning Yellow
        } else {
            // > 0.75 is caught by data.alert below
            analysisStatus.textContent = "POSSIBLE AI VOICE CLONE";
            analysisStatus.style.color = "#ff4d4d"; // Danger Red
        }

        // Trigger Massive Alert Overlay
        if (data.alert) {
            triggerDeepfakeAlert(s_pct);
        }
    }

    function triggerDeepfakeAlert(probability) {
        // Auto-Disarm Demo Mode to prevent infinite false-positives
        demoMode = false;
        clickCount = 0;
        document.querySelector('.app-logo svg').setAttribute('stroke', '#00f2fe');

        // Vibrate Smartphone natively if supported (SOS Pattern)
        if ("vibrate" in navigator) {
            navigator.vibrate([400, 200, 400, 200, 800, 400]);
        }

        overlayProb.textContent = `${probability}%`;
        alertOverlay.classList.remove('hidden');

        // Sever the microphone connection to protect privacy
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        // Get the active stream and kill all hardware tracks immediately
        if (mediaRecorder && mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        isMonitoring = false;

        // Stop Visualizer Execution
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Release Screen Wake Lock
        releaseWakeLock();

        // Update UI to reflect severed connection
        radarContainer.classList.remove('active');
        navStatus.textContent = 'MIC SEVERED';
        navStatus.classList.remove('live');
        navStatus.style.color = '#ff4d4d'; // Danger Red

        document.querySelector('.btn-text').innerHTML = "SYSTEM<br>LOCKED";
        document.querySelector('.btn-subtext').textContent = "Threat neutralized";

        analysisStatus.textContent = "CRITICAL: DEEPFAKE DETECTED. MIC SEVERED.";
    }

    // Dismiss Alert
    dismissBtn.addEventListener('click', () => {
        alertOverlay.classList.add('hidden');
        if ("vibrate" in navigator) navigator.vibrate(0); // Stop vibration

        // Reset nav status color for next use
        navStatus.style.color = '';
    });

});
