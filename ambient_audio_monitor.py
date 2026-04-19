import datetime
import json
import os
import shutil
import time
import wave
from pathlib import Path

import pyaudio
import torch

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

BASE_DIR = Path(__file__).resolve().parent
TEMP_FILE = BASE_DIR / "ambient_temp.wav"
FLAGGED_DIR = BASE_DIR / "flagged_calls"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"
MODEL_PATH = BASE_DIR / "model.pth"

ALERT_FLOOR_THRESHOLD = 0.55
CONSECUTIVE_ALERTS_REQUIRED = 3
MONITOR_SENSITIVITY = 0.72
MONITOR_MODEL_WEIGHT = 0.76
MONITOR_ARTIFACT_WEIGHT = 0.24
MONITOR_CHUNK_SECONDS = 0.9
MONITOR_HOP_SECONDS = 0.35

FLAGGED_DIR.mkdir(parents=True, exist_ok=True)


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
            print("[!] model.pth missing.")
            return None
        model = AudioCNN(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        print(f"[!] Failed loading model: {exc}")
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
        print(f"[!] Audio capture error: {exc}")
        return False
    finally:
        p.terminate()


def log_flagged_clip(temp_file: Path, probability: float) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prob = int(probability * 100)
    destination = FLAGGED_DIR / f"ambient_alert_{timestamp}_prob{safe_prob}.wav"
    shutil.copy(str(temp_file), str(destination))
    print(f"\n[+] Saved suspicious ambient clip: {destination}")


def start_ambient_monitor() -> None:
    model = load_model()
    if model is None:
        return

    base_threshold = load_base_threshold(0.5)

    print("=" * 60)
    print("VACHA-SHIELD AMBIENT MONITOR")
    print("=" * 60)
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Base threshold: {base_threshold:.2f} | Alert floor: {ALERT_FLOOR_THRESHOLD:.2f}")
    print("[*] Press Ctrl+C to stop")

    cycle = 1
    streak = 0

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

            synthetic = float(result["synthetic_probability"])
            human = float(result["human_probability"])
            effective_threshold = max(float(result.get("threshold", base_threshold)), ALERT_FLOOR_THRESHOLD)

            if synthetic > effective_threshold:
                streak += 1
            else:
                streak = 0

            if streak >= CONSECUTIVE_ALERTS_REQUIRED:
                print(
                    f"\n[AMBIENT ALERT] cycle={cycle} ai={synthetic:.3f} human={human:.3f} "
                    f"thr={effective_threshold:.3f}"
                )
                log_flagged_clip(TEMP_FILE, synthetic)
                streak = 0
            else:
                print(
                    f"[cycle {cycle}] human={human:.3f} ai={synthetic:.3f} "
                    f"thr={effective_threshold:.3f} streak={streak}",
                    end="\r",
                )

            if TEMP_FILE.exists():
                TEMP_FILE.unlink(missing_ok=True)

            cycle += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[*] Ambient monitor stopped.")
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    start_ambient_monitor()
