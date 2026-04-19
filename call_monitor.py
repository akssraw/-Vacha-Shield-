import collections
import datetime
import json
import os
import queue
import shutil
import struct
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyaudio
import torch
from dotenv import load_dotenv

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

# =============================================================================
# VACHA-SHIELD CALL MONITOR v2.0
# Changes from v1.0:
#   - Integrated Sarvam AI live transcription (Hindi/Hinglish/English)
#   - Shared audio buffer between deepfake detector and transcription engine
#   - Scam keyword detection on live transcript
#   - SARVAM_API_KEY loaded from environment variable or .env file
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
TEMP_FILE = BASE_DIR / "ambient_temp_call.wav"
FLAGGED_DIR = BASE_DIR / "flagged_calls"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"
MODEL_PATH = BASE_DIR / "model.pth"

# Deepfake detection thresholds (unchanged from v1)
ALERT_FLOOR_THRESHOLD = 0.55
CONSECUTIVE_ALERTS_REQUIRED = 2
ALERT_COOLDOWN_SECONDS = 12
MONITOR_SENSITIVITY = 0.72
MONITOR_MODEL_WEIGHT = 0.76
MONITOR_ARTIFACT_WEIGHT = 0.24
MONITOR_CHUNK_SECONDS = 0.9
MONITOR_HOP_SECONDS = 0.35

# =============================================================================
# ── NEW: Sarvam Transcription Config ─────────────────────────────────────────
# Set your API key in .env file as: SARVAM_API_KEY=your_key_here
# OR set environment variable before running:
#   Windows:  set SARVAM_API_KEY=your_key_here
#   Linux/Mac: export SARVAM_API_KEY=your_key_here
# Get your key at: https://dashboard.sarvam.ai/
# =============================================================================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_ENDPOINT = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text").strip()
SARVAM_MODEL = "saaras:v3"
SARVAM_MODE = "translit"       # Hindi/Hinglish → Roman script
SARVAM_LANGUAGE = "hi-IN"      # Change to "en-IN" for English only
TRANSCRIPTION_ENABLED = bool(SARVAM_API_KEY)

# Transcription audio settings
SILENCE_THRESHOLD = 500        # RMS level below which audio is "silence"
SILENCE_DURATION = 1.0         # seconds of silence to end an utterance
MAX_UTTERANCE_SEC = 12         # max seconds per transcription chunk
PREROLL_SEC = 0.35             # seconds of audio before speech starts
STREAM_INTERVAL = 1.2          # seconds between live transcription updates
MIN_AUDIO_SEC = 0.35           # minimum audio length to attempt transcription
MAX_PARALLEL_TRANSCRIPTIONS = 4

# Scam keywords to watch for in transcript
SCAM_KEYWORDS = [
    "otp", "one time password",
    "kyc", "re kyc",
    "upi", "upi pin",
    "cvv", "card number",
    "bank account", "bank verification",
    "refund", "claim refund",
    "remote access", "anydesk", "teamviewer",
    "gift card", "amazon voucher",
    "screen share",
    "police case", "arrest warrant",
    "courier", "customs",
    "lottery", "you won",
    "verify your account",
]

FLAGGED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ── Existing functions (UNCHANGED) ───────────────────────────────────────────
# =============================================================================

def load_base_threshold(default: float = 0.5) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", default))
        return threshold if 0.1 <= threshold <= 0.9 else default
    except Exception:
        return default


def load_model() -> AudioCNN | None:
    try:
        if not MODEL_PATH.exists():
            print("[!] model.pth not found. Train model first.")
            return None
        model = AudioCNN(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        print(f"[!] Failed to load model: {exc}")
        return None


def record_chunk(temp_file: Path) -> bool:
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        frames = []
        total_frames = int(SAMPLE_RATE / FRAMES_PER_BUFFER * CHUNK_SECONDS)
        for _ in range(total_frames):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return True
    except OSError as exc:
        print(f"[!] Microphone error: {exc}")
        return False
    finally:
        p.terminate()


def log_flagged_clip(temp_file: Path, probability: float) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prob = int(probability * 100)
    destination = FLAGGED_DIR / f"deepfake_log_{timestamp}_prob{safe_prob}.wav"
    shutil.copy(str(temp_file), str(destination))
    print(f"[+] Forensic clip saved: {destination}")


# =============================================================================
# ── NEW: Transcription Engine (from worknd.py) ────────────────────────────────
# These functions are taken from worknd.py and adapted for terminal output.
# No UI/Tkinter needed here — transcripts print directly to console.
# =============================================================================

def _rms(data: bytes) -> float:
    """Calculate RMS energy of audio chunk to detect speech vs silence."""
    count = len(data) // 2
    if not count:
        return 0.0
    shorts = struct.unpack(f"{count}h", data)
    return (sum(s * s for s in shorts) / count) ** 0.5


def _frames_to_wav(frames: list) -> str:
    """Save audio frames to a temporary WAV file, return file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return tmp.name


def _transcribe_frames(frames: list) -> str:
    """
    Send audio frames to Sarvam AI and return transcript text.
    Returns empty string if transcription fails or API key not set.
    """
    if not SARVAM_API_KEY:
        return ""
    if len(frames) * FRAMES_PER_BUFFER / SAMPLE_RATE < MIN_AUDIO_SEC:
        return ""

    path = _frames_to_wav(frames)
    try:
        with open(path, "rb") as f:
            resp = httpx.post(
                SARVAM_ENDPOINT,
                headers={"api-subscription-key": SARVAM_API_KEY},
                data={
                    "model": SARVAM_MODEL,
                    "mode": SARVAM_MODE,
                    "language_code": SARVAM_LANGUAGE,
                },
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=15,
            )
        resp.raise_for_status()
        return resp.json().get("transcript", "").strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _check_scam_keywords(text: str) -> list[str]:
    """Check transcript for scam keywords. Returns list of matched keywords."""
    normalized = text.lower()
    return [kw for kw in SCAM_KEYWORDS if kw in normalized]


class TranscriptionThread(threading.Thread):
    """
    Runs in background alongside the main deepfake detection loop.
    Listens to mic independently using VAD (Voice Activity Detection),
    sends speech segments to Sarvam AI, and prints transcripts to console.

    This is the core of worknd.py adapted for terminal use (no Tkinter UI).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSCRIPTIONS)
        self._latest_transcript = ""
        self._scam_hits: list[str] = []
        self._lock = threading.Lock()

    def stop(self):
        self._running = False

    def get_latest(self) -> tuple[str, list[str]]:
        """Returns (latest_transcript, scam_keywords_found)"""
        with self._lock:
            return self._latest_transcript, list(self._scam_hits)

    def _on_utterance(self, frames: list, is_final: bool) -> None:
        """Called in thread pool — transcribes and prints result."""
        text = _transcribe_frames(frames)
        if not text:
            return

        scam_hits = _check_scam_keywords(text)

        with self._lock:
            self._latest_transcript = text
            if scam_hits:
                self._scam_hits = scam_hits

        prefix = "[TRANSCRIPT]" if is_final else "[LIVE]      "
        print(f"\n{prefix} {text}")

        if scam_hits:
            print(f"[⚠ SCAM KEYWORDS] {', '.join(scam_hits).upper()}")

    def run(self):
        self._running = True
        pa = pyaudio.PyAudio()

        try:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )
        except Exception as exc:
            print(f"[!] Transcription mic error: {exc}")
            pa.terminate()
            return

        preroll = collections.deque(
            maxlen=int(SAMPLE_RATE / FRAMES_PER_BUFFER * PREROLL_SEC)
        )
        sil_needed = int(SAMPLE_RATE / FRAMES_PER_BUFFER * SILENCE_DURATION)
        max_chunks = int(SAMPLE_RATE / FRAMES_PER_BUFFER * MAX_UTTERANCE_SEC)
        stream_every = int(SAMPLE_RATE / FRAMES_PER_BUFFER * STREAM_INTERVAL)

        utterance: list = []
        silent_cnt = 0
        speaking = False
        since_stream = 0

        while self._running:
            try:
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            except Exception:
                continue

            level = _rms(data)

            if level > SILENCE_THRESHOLD:
                if not speaking:
                    utterance = list(preroll) + [data]
                    speaking = True
                else:
                    utterance.append(data)
                    silent_cnt = 0
                    since_stream += 1

                # Send live preview every STREAM_INTERVAL seconds
                if since_stream >= stream_every:
                    self._executor.submit(self._on_utterance, list(utterance), False)
                    since_stream = 0

                # Force-end utterance if too long
                if len(utterance) >= max_chunks:
                    self._executor.submit(self._on_utterance, list(utterance), True)
                    utterance = []
                    speaking = False
                    since_stream = 0

            elif speaking:
                utterance.append(data)
                silent_cnt += 1

                if silent_cnt >= sil_needed:
                    # Silence detected — send final utterance
                    self._executor.submit(self._on_utterance, list(utterance), True)
                    utterance = []
                    speaking = False
                    silent_cnt = 0
                else:
                    preroll.append(data)
            else:
                preroll.append(data)

        stream.stop_stream()
        stream.close()
        pa.terminate()
        self._executor.shutdown(wait=False)


# =============================================================================
# ── Main Monitor (MODIFIED to include transcription) ─────────────────────────
# Changes from v1:
#   1. Starts TranscriptionThread before the main loop
#   2. Prints transcript alongside deepfake scores each cycle
#   3. Stops TranscriptionThread on exit
# =============================================================================

def simulate_call_monitor() -> None:
    model = load_model()
    if model is None:
        return

    base_threshold = load_base_threshold(0.5)
    print("=" * 64)
    print("VACHA-SHIELD CALL MONITOR v2.0")
    print("=" * 64)
    print(f"[*] Device:                    {DEVICE}")
    print(f"[*] Base threshold:            {base_threshold:.2f}")
    print(f"[*] Alert floor threshold:     {ALERT_FLOOR_THRESHOLD:.2f}")
    print(f"[*] Consecutive alerts needed: {CONSECUTIVE_ALERTS_REQUIRED}")

    # ── NEW: Show transcription status ────────────────────────────────────────
    if TRANSCRIPTION_ENABLED:
        print(f"[*] Transcription:             ON (Sarvam AI · {SARVAM_LANGUAGE})")
    else:
        print("[*] Transcription:             OFF (set SARVAM_API_KEY to enable)")
    print("=" * 64)

    try:
        input("Press ENTER to start call monitoring...")
    except KeyboardInterrupt:
        return

    # ── NEW: Start transcription thread ───────────────────────────────────────
    transcription_thread = None
    if TRANSCRIPTION_ENABLED:
        transcription_thread = TranscriptionThread()
        transcription_thread.start()
        print("[*] Transcription engine started.")

    streak = 0
    cycle = 1
    last_alert_time = 0.0

    try:
        while True:
            if not record_chunk(TEMP_FILE):
                break

            result = predict_deepfake_from_file(
                audio_path=str(TEMP_FILE),
                model=model,
                device=DEVICE,
                threshold=base_threshold,
                chunk_seconds=MONITOR_CHUNK_SECONDS,
                hop_seconds=MONITOR_HOP_SECONDS,
                sensitivity=MONITOR_SENSITIVITY,
                model_weight=MONITOR_MODEL_WEIGHT,
                artifact_weight=MONITOR_ARTIFACT_WEIGHT,
            )

            effective_threshold = max(
                float(result.get("threshold", base_threshold)),
                ALERT_FLOOR_THRESHOLD,
            )
            synthetic = float(result["synthetic_probability"])
            human = float(result["human_probability"])

            if synthetic > effective_threshold:
                streak += 1
            else:
                streak = 0

            now = time.time()
            can_alert = (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS

            # ── NEW: Get latest transcript for this cycle ──────────────────
            transcript = ""
            scam_hits: list[str] = []
            if transcription_thread:
                transcript, scam_hits = transcription_thread.get_latest()

            if streak >= CONSECUTIVE_ALERTS_REQUIRED and can_alert:
                print(
                    f"\n[DEEPFAKE ALERT] cycle={cycle} | AI={synthetic:.3f} | "
                    f"Human={human:.3f} | threshold={effective_threshold:.3f} | "
                    f"streak={streak}"
                )
                # ── NEW: Show scam keywords on alert ──────────────────────
                if scam_hits:
                    print(f"[⚠ SCAM KEYWORDS DETECTED] {', '.join(scam_hits).upper()}")
                if transcript:
                    print(f"[LAST TRANSCRIPT] {transcript}")

                log_flagged_clip(TEMP_FILE, synthetic)
                last_alert_time = now
                streak = 0
            else:
                # ── MODIFIED: Show transcript in normal cycle output ───────
                transcript_display = f" | transcript: {transcript[:40]}..." if transcript else ""
                print(
                    f"[cycle {cycle}] human={human:.3f} ai={synthetic:.3f} "
                    f"thr={effective_threshold:.3f} streak={streak}"
                    f"{transcript_display}",
                    end="\r",
                )

            if TEMP_FILE.exists():
                TEMP_FILE.unlink(missing_ok=True)

            cycle += 1

    except KeyboardInterrupt:
        print("\n[*] Monitoring stopped by user.")
    finally:
        # ── NEW: Stop transcription thread on exit ─────────────────────────
        if transcription_thread:
            transcription_thread.stop()
            print("[*] Transcription engine stopped.")
        if TEMP_FILE.exists():
            TEMP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    simulate_call_monitor()
