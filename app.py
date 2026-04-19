from __future__ import annotations
import imageio_ffmpeg as _iffmpeg
import os as _os
_os.environ["PATH"] += _os.pathsep + _os.path.dirname(_iffmpeg.get_ffmpeg_exe())
import base64
import datetime
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from groq import Groq
from werkzeug.utils import secure_filename

from deepfake_detector import predict_deepfake_from_file, predict_deepfake_from_waveform
from model import AudioCNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(tempfile.gettempdir()) / "vacha_shield_runtime"
ENV_FILE = os.getenv("VACHA_ENV_FILE")
if ENV_FILE:
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv(dotenv_path=BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_STT_URL = os.getenv("SARVAM_STT_URL", "").strip()
SARVAM_LANGUAGE_CODE = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN").strip()
GOOGLE_STT_LANGUAGE_CODES = tuple(
    code.strip()
    for code in os.getenv("GOOGLE_STT_LANGUAGE_CODES", "en-IN,hi-IN,en-US").split(",")
    if code.strip()
) or ("en-IN", "hi-IN", "en-US")

UPLOAD_DIR = RUNTIME_DIR / "temp_uploads"
FLAGGED_DIR = BASE_DIR / "flagged_calls"
CONTINUOUS_DATASET_DIR = BASE_DIR / "continuous_learning_dataset"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"
STT_COMPATIBLE_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".aifc"}
ALERT_DEFAULT_FLOOR = float(os.getenv("ALERT_MIN_THRESHOLD", "0.62"))
DEFAULT_ANALYSIS_PROFILE = os.getenv("DEFAULT_ANALYSIS_PROFILE", "strict").strip().lower()
SEMANTIC_OVERRIDE_ENABLED = os.getenv("SEMANTIC_OVERRIDE_ENABLED", "false").strip().lower() == "true"
VACHA_API_KEY = os.getenv("VACHA_API_KEY", "").strip()
VIDEO_COMPATIBLE_EXTENSIONS = {".webm", ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg"}
PRIMARY_RESAMPLE_TYPE = "kaiser_fast"
FALLBACK_RESAMPLE_TYPE = "polyphase"


def _coerce_env_int(raw_value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_env_float(raw_value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_cors_origins(raw_value: str | None) -> str | list[str]:
    text = (raw_value or "").strip()
    if not text or text == "*":
        return "*"
    origins = [item.strip() for item in text.split(",") if item.strip()]
    return origins or "*"


MAX_UPLOAD_MB = _coerce_env_int(os.getenv("MAX_UPLOAD_MB"), 40, 5, 500)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_FILES = _coerce_env_int(os.getenv("MAX_BATCH_FILES"), 8, 1, 50)
PHRASE_SEARCH_WINDOW_SECONDS = _coerce_env_float(os.getenv("PHRASE_SEARCH_WINDOW_SECONDS"), 5.0, 1.5, 15.0)
PHRASE_SEARCH_HOP_SECONDS = _coerce_env_float(os.getenv("PHRASE_SEARCH_HOP_SECONDS"), 2.5, 0.5, 8.0)
PHRASE_SEARCH_MIN_AMPLITUDE = _coerce_env_float(os.getenv("PHRASE_SEARCH_MIN_AMPLITUDE"), 0.008, 0.001, 0.08)
PHRASE_SEARCH_MAX_MATCHES = _coerce_env_int(os.getenv("PHRASE_SEARCH_MAX_MATCHES"), 25, 1, 200)
CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS", "*"))

# Drift-resistant inference is fixed to 3-second model windows with a 1-second stride.
INFERENCE_WINDOW_SECONDS = 3.0
INFERENCE_STRIDE_SECONDS = 1.0
INFERENCE_POOLING_METHOD = os.getenv("INFERENCE_POOLING_METHOD", "median").strip().lower()
if INFERENCE_POOLING_METHOD not in {"mean", "median"}:
    INFERENCE_POOLING_METHOD = "median"
HAB_PITCH_VARIANCE_THRESHOLD = _coerce_env_float(os.getenv("HAB_PITCH_VARIANCE_THRESHOLD"), 0.30, 0.0, 1.0)
HAB_AI_PROBABILITY_PENALTY = _coerce_env_float(os.getenv("HAB_AI_PROBABILITY_PENALTY"), 0.15, 0.0, 0.9)

ANALYSIS_PROFILES: dict[str, dict[str, float]] = {
    "conservative": {
        "sensitivity": 0.40,
        "decision_floor": 0.66,
        "model_weight": 0.84,
        "artifact_weight": 0.16,
        "chunk_seconds": 1.0,
        "hop_seconds": 0.5,
        "borderline_margin": 0.03,
    },
    "balanced": {
        "sensitivity": 0.48,
        "decision_floor": ALERT_DEFAULT_FLOOR,
        "model_weight": 0.82,
        "artifact_weight": 0.18,
        "chunk_seconds": 1.0,
        "hop_seconds": 0.5,
        "borderline_margin": 0.04,
    },
    "strict": {
        "sensitivity": 0.62,
        "decision_floor": 0.58,
        "model_weight": 0.78,
        "artifact_weight": 0.22,
        "chunk_seconds": 0.95,
        "hop_seconds": 0.4,
        "borderline_margin": 0.06,
    },
    "forensic": {
        "sensitivity": 0.80,
        "decision_floor": 0.58,
        "model_weight": 0.72,
        "artifact_weight": 0.28,
        "chunk_seconds": 0.8,
        "hop_seconds": 0.3,
        "borderline_margin": 0.08,
    },
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _find_lovable_dist() -> Path | None:
    override = os.getenv("LOVABLE_DIST_DIR", "").strip()
    if override:
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = BASE_DIR / override_path
        if override_path.exists() and override_path.is_dir():
            return override_path

    bundled_dist = BASE_DIR / "lovable-dist"
    if bundled_dist.exists() and bundled_dist.is_dir():
        return bundled_dist

    candidates = sorted(
        BASE_DIR.glob("lovable-project-*/dist"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


LOVABLE_DIST_DIR = _find_lovable_dist()

# Serve built Lovable UI when available, otherwise keep legacy static folder behavior.
app = Flask(
    __name__,
    static_folder=str(LOVABLE_DIST_DIR) if LOVABLE_DIST_DIR else "static",
    static_url_path="/static",
    template_folder="templates",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/*": {"origins": CORS_ALLOWED_ORIGINS}})
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ALLOWED_ORIGINS,
    async_mode="threading",
    ping_interval=10,
    ping_timeout=30,
    max_http_buffer_size=2_000_000,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Inference API on: {DEVICE}")

def _load_base_threshold(default: float = 0.50) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", default))
        if 0.1 <= threshold <= 0.9:
            print(f"Loaded calibrated threshold from model_calibration.json: {threshold:.3f}")
            return threshold
    except Exception as e:
        print(f"Could not load model calibration file: {e}")
    return default



def _load_duration_penalty_coefficient(default: float = 0.08) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        coefficient = float(payload.get("duration_penalty_coefficient", default))
        if 0.0 <= coefficient <= 0.5:
            print(f"Loaded duration penalty coefficient from model_calibration.json: {coefficient:.3f}")
            return coefficient
    except Exception as e:
        print(f"Could not load duration penalty coefficient: {e}")
    return default
def _coerce_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return float(np.clip(parsed, minimum, maximum))


def _resolve_analysis_profile(profile_name: str | None) -> tuple[str, dict[str, float]]:
    candidate = (profile_name or DEFAULT_ANALYSIS_PROFILE).strip().lower()
    if candidate not in ANALYSIS_PROFILES:
        candidate = DEFAULT_ANALYSIS_PROFILE if DEFAULT_ANALYSIS_PROFILE in ANALYSIS_PROFILES else "balanced"
    return candidate, dict(ANALYSIS_PROFILES[candidate])


def _extract_analysis_params(form_data) -> dict[str, float | str]:
    profile_name, preset = _resolve_analysis_profile(form_data.get("analysis_profile"))

    chunk_seconds = INFERENCE_WINDOW_SECONDS
    hop_seconds = INFERENCE_STRIDE_SECONDS

    model_weight = _coerce_float(form_data.get("model_weight"), preset["model_weight"], 0.05, 0.95)
    artifact_weight = _coerce_float(form_data.get("artifact_weight"), preset["artifact_weight"], 0.05, 0.95)

    return {
        "profile": profile_name,
        "sensitivity": _coerce_float(form_data.get("sensitivity"), preset["sensitivity"], 0.0, 1.0),
        "decision_floor": _coerce_float(form_data.get("decision_floor"), preset["decision_floor"], 0.35, 0.8),
        "model_weight": model_weight,
        "artifact_weight": artifact_weight,
        "chunk_seconds": chunk_seconds,
        "hop_seconds": hop_seconds,
        "window_seconds": chunk_seconds,
        "stride_seconds": hop_seconds,
        "pooling_method": INFERENCE_POOLING_METHOD,
        "hab_pitch_variance_threshold": HAB_PITCH_VARIANCE_THRESHOLD,
        "hab_penalty": HAB_AI_PROBABILITY_PENALTY,
        "borderline_margin": _coerce_float(form_data.get("borderline_margin"), preset["borderline_margin"], 0.02, 0.2),
    }

# Base decision threshold. The detector applies small per-clip adaptive changes.
DEEPFAKE_THRESHOLD = _load_base_threshold(0.50)
DURATION_PENALTY_COEFFICIENT = _load_duration_penalty_coefficient(0.08)

class ModelRuntime:
    """Process-wide singleton that keeps the PyTorch model hot in memory."""

    def __init__(self, model_path: Path, device: torch.device):
        self.model_path = model_path
        self.device = device
        self._model: AudioCNN | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._load_model_once()

    def _load_model_once(self) -> AudioCNN | None:
        if self._model is not None or self._load_error is not None:
            return self._model

        with self._lock:
            if self._model is not None or self._load_error is not None:
                return self._model

            try:
                loaded_model = AudioCNN(num_classes=1)
                if self.model_path.exists():
                    loaded_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    print("Loaded model weights from 'model.pth'.")
                else:
                    print("WARNING: 'model.pth' not found. The model is using random weights.")

                loaded_model.to(self.device)
                loaded_model.eval()
                self._model = loaded_model
            except Exception as e:
                self._load_error = str(e)
                print(f"Critical error loading model: {e}")

        return self._model

    @property
    def model(self) -> AudioCNN | None:
        return self._load_model_once()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_file(self, audio_path: str, **kwargs) -> dict:
        if self.model is None:
            raise RuntimeError(self._load_error or "AI model failed to load.")
        return predict_deepfake_from_file(audio_path=audio_path, model=self.model, device=self.device, **kwargs)

    def predict_waveform(self, audio: np.ndarray, sample_rate: int, **kwargs) -> dict:
        if self.model is None:
            raise RuntimeError(self._load_error or "AI model failed to load.")
        return predict_deepfake_from_waveform(
            audio=audio,
            sample_rate=sample_rate,
            model=self.model,
            device=self.device,
            **kwargs,
        )


MODEL_RUNTIME = ModelRuntime(model_path=BASE_DIR / "model.pth", device=DEVICE)


def generate_spectrogram_base64(audio_path: Path) -> str | None:
    """Generate a compact base64 mel spectrogram image for UI rendering."""
    try:
        try:
            y, sr_rate = librosa.load(str(audio_path), sr=16000, res_type=PRIMARY_RESAMPLE_TYPE)
        except Exception as load_error:
            if not (isinstance(load_error, ModuleNotFoundError) and "resampy" in str(load_error).lower()):
                raise
            y, sr_rate = librosa.load(str(audio_path), sr=16000, res_type=FALLBACK_RESAMPLE_TYPE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr_rate, n_mels=30)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(8, 3))
        plt.style.use("dark_background")
        librosa.display.specshow(mel_db, sr=sr_rate, x_axis="time", y_axis="mel", cmap="magma")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", transparent=True, dpi=72, bbox_inches="tight")
        plt.close("all")

        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


def _semantic_keyword_hit(transcript: str) -> bool:
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", transcript.lower())).strip()
    phrases = [
        "i am an ai",
        "i'm an ai",
        "i am a bot",
        "i'm a bot",
        "i am a virtual assistant",
        "i am an artificial intelligence",
        "this is an ai voice",
        "this voice is generated",
    ]
    return any(phrase in text for phrase in phrases)


SCAM_KEYWORD_PATTERNS: dict[str, tuple[str, ...]] = {
    "otp": ("otp", "one time password"),
    "pin": ("pin", "atm pin"),
    "cvv": ("cvv", "card verification value"),
    "bank": ("bank account", "bank verification", "bank officer", "bank manager"),
    "kyc": ("kyc", "re kyc", "kyc update", "kyc verification"),
    "upi": ("upi", "upi id", "upi pin"),
    "refund": ("refund", "claim refund", "tax refund"),
    "gift_card": ("gift card", "amazon voucher", "google play card"),
    "screen_share": ("screen share", "share your screen"),
    "remote_access": ("remote access", "anydesk", "teamviewer", "quicksupport"),
    "urgent_payment": ("pay now", "send money now", "make payment immediately"),
    "transaction_blocked": ("transaction blocked", "account blocked", "account suspended"),
    "lottery": ("lottery", "you won", "prize money"),
    "police": ("police case", "police complaint", "cyber cell", "arrest warrant"),
    "courier": ("courier", "customs clearance", "parcel held"),
    "verification": ("verify your account", "confirm your identity", "verification code"),
}

HIGH_RISK_SCAM_KEYWORDS = {
    "otp",
    "pin",
    "cvv",
    "gift_card",
    "remote_access",
    "urgent_payment",
    "transaction_blocked",
}


def _extract_scam_keyword_hits(transcript: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", transcript.lower())).strip()
    if not normalized:
        return []

    hits: list[str] = []
    for label, phrases in SCAM_KEYWORD_PATTERNS.items():
        if any(phrase in normalized for phrase in phrases):
            hits.append(label)
    return hits


def _score_transcript_scam_risk(keyword_hits: list[str]) -> float:
    unique_hits = set(keyword_hits)
    if not unique_hits:
        return 0.0

    score = 0.0
    score += min(0.26, 0.09 * len(unique_hits))
    high_risk_count = sum(1 for keyword in unique_hits if keyword in HIGH_RISK_SCAM_KEYWORDS)
    if high_risk_count:
        score += min(0.28, 0.11 * high_risk_count)
    if len(unique_hits) >= 3:
        score += 0.12
    return float(np.clip(score, 0.0, 0.95))


def _extract_text_from_json(payload) -> str:
    if isinstance(payload, dict):
        for key in (
            "transcript",
            "text",
            "output",
            "result",
            "recognized_text",
            "best_transcript",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("data", "response", "results"):
            nested = payload.get(key)
            extracted = _extract_text_from_json(nested)
            if extracted:
                return extracted
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_text_from_json(item)
            if extracted:
                return extracted
    return ""


def _transcribe_with_sarvam(audio_path: Path) -> str:
    if not SARVAM_API_KEY or not SARVAM_STT_URL:
        return ""
    if not audio_path.exists() or not audio_path.is_file():
        return ""

    boundary = f"----vacha-sarvam-{uuid.uuid4().hex}"
    content = audio_path.read_bytes()
    filename = audio_path.name
    lines: list[bytes] = []

    def _append_field(name: str, value: str) -> None:
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        lines.append(f"{value}\r\n".encode("utf-8"))

    _append_field("language_code", SARVAM_LANGUAGE_CODE)
    _append_field("model", "saaras:v2")

    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    lines.append(b"Content-Type: audio/wav\r\n\r\n")
    lines.append(content)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(lines)
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    request_obj = urllib.request.Request(SARVAM_STT_URL, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request_obj, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        transcript = _extract_text_from_json(payload)
        if transcript:
            print("[TRANSCRIPT] Sarvam STT succeeded.")
        return transcript
    except urllib.error.HTTPError as e:
        print(f"Sarvam STT HTTP error: {e.code}")
    except urllib.error.URLError as e:
        print(f"Sarvam STT connection error: {e}")
    except Exception as e:
        print(f"Sarvam STT error: {e}")
    return ""


def _transcribe_audio_file(audio_path: Path) -> str:
    transcript = _transcribe_with_sarvam(audio_path)
    if transcript:
        return transcript

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio_data = recognizer.record(source)
        transcript = _recognize_google_with_language_fallback(recognizer, audio_data)
        if transcript:
            return transcript

        boosted_path = _create_stt_boosted_wav(audio_path)
        if boosted_path is None:
            return ""

        try:
            with sr.AudioFile(str(boosted_path)) as source:
                boosted_audio = recognizer.record(source)
            return _recognize_google_with_language_fallback(recognizer, boosted_audio)
        finally:
            _cleanup_paths(boosted_path)
    except sr.RequestError as e:
        print(f"STT unavailable: {e}")
        return ""
    except Exception as e:
        print(f"Transcript fallback error: {e}")
        return ""


def _recognize_google_with_language_fallback(recognizer: sr.Recognizer, audio_data: sr.AudioData) -> str:
    for language_code in GOOGLE_STT_LANGUAGE_CODES:
        try:
            text = recognizer.recognize_google(audio_data, language=language_code).strip()
            if text:
                return text
        except sr.UnknownValueError:
            continue
    return ""


def _create_stt_boosted_wav(audio_path: Path) -> Path | None:
    if audio_path.suffix.lower() not in STT_COMPATIBLE_EXTENSIONS:
        return None

    tmp_path: Path | None = None
    try:
        with wave.open(str(audio_path), "rb") as wav_handle:
            channels = wav_handle.getnchannels()
            sample_width = wav_handle.getsampwidth()
            frame_rate = wav_handle.getframerate()
            frames = wav_handle.readframes(wav_handle.getnframes())

        if channels != 1 or sample_width != 2 or not frames:
            return None

        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        if peak < 0.003 or peak >= 0.32:
            return None

        gain = min(14.0, 0.72 / max(peak, 1e-6))
        boosted = np.clip(pcm * gain, -1.0, 1.0)
        boosted16 = (boosted * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)

        with wave.open(str(tmp_path), "wb") as wav_handle:
            wav_handle.setnchannels(1)
            wav_handle.setsampwidth(2)
            wav_handle.setframerate(frame_rate)
            wav_handle.writeframes(boosted16.tobytes())

        return tmp_path
    except Exception as e:
        print(f"STT boost skipped: {e}")
        _cleanup_paths(*(path for path in [tmp_path] if path is not None))
        return None


def _transcribe_waveform(audio_window: np.ndarray, sample_rate: int) -> str:
    if audio_window is None or audio_window.size == 0:
        return ""
    peak = float(np.max(np.abs(audio_window)))
    if peak < 0.004:
        return ""

    recognizer = sr.Recognizer()
    tmp_path: Path | None = None
    try:
        pcm = np.clip(np.asarray(audio_window, dtype=np.float32), -1.0, 1.0)
        if peak < 0.32:
            pcm = np.clip(pcm * min(14.0, 0.72 / max(peak, 1e-6)), -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)

        with wave.open(str(tmp_path), "wb") as wav_handle:
            wav_handle.setnchannels(1)
            wav_handle.setsampwidth(2)
            wav_handle.setframerate(int(sample_rate))
            wav_handle.writeframes(pcm16.tobytes())

        with sr.AudioFile(str(tmp_path)) as source:
            audio_data = recognizer.record(source)
        return _recognize_google_with_language_fallback(recognizer, audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Live STT unavailable: {e}")
        return ""
    except Exception as e:
        print(f"Live transcript error: {e}")
        return ""
    finally:
        _cleanup_paths(*(path for path in [tmp_path] if path is not None))


def _normalize_search_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()
    return normalized


def _query_present_in_transcript(query_normalized: str, transcript_normalized: str) -> bool:
    if not query_normalized or not transcript_normalized:
        return False
    if " " in query_normalized:
        return query_normalized in transcript_normalized
    return re.search(rf"\b{re.escape(query_normalized)}\b", transcript_normalized) is not None


def _extract_api_key_candidate() -> str:
    header_value = (request.headers.get("X-API-Key") or "").strip()
    if header_value:
        return header_value
    authorization = (request.headers.get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _require_api_key_if_enabled() -> tuple[dict, int] | None:
    if not VACHA_API_KEY:
        return None
    supplied = _extract_api_key_candidate()
    if supplied and hmac.compare_digest(supplied, VACHA_API_KEY):
        return None
    return jsonify({"error": "Unauthorized. Provide a valid API key."}), 401


def _is_decode_error_message(message: str) -> bool:
    lowered = message.lower()
    decode_error_tokens = (
        "ffmpeg",
        "webm",
        "decode",
        "invalid data",
        "moov",
        "could not decode",
    )
    return any(token in lowered for token in decode_error_tokens)


def _build_inconclusive_chunk_response() -> dict:
    return {
        "synthetic_probability": 0.5,
        "human_probability": 0.5,
        "alert": False,
        "threshold": DEEPFAKE_THRESHOLD,
        "verdict": "inconclusive",
        "processing_state": "inconclusive_chunk",
        "decision_mode": "fallback_decode_guard",
        "decision_summary": "Unreadable audio chunk skipped. Continuing with the next live chunk.",
    }


def semantic_deepfake_check(audio_path: Path) -> bool:
    """
    Optional semantic override:
    if transcript explicitly self-identifies as AI, raise a guaranteed flag.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(audio_path)) as source:
            audio_data = recognizer.record(source)
            text = _recognize_google_with_language_fallback(recognizer, audio_data)
            print(f"[STT] '{text}'")

        if not text:
            return False

        if _semantic_keyword_hit(text):
            print("[SEMANTIC] Explicit AI self-identification detected via keyword rules.")
            return True

        if not GROQ_API_KEY:
            return False

        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            "Classify this transcript as TRUE or FALSE only. "
            "TRUE if the speaker explicitly identifies as AI, bot, or virtual assistant. "
            f"Transcript: {text!r}"
        )
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=5,
        )

        result = completion.choices[0].message.content.strip().upper()
        semantic_hit = "TRUE" in result
        if semantic_hit:
            print("[SEMANTIC] LLM override triggered.")
        return semantic_hit
    except sr.UnknownValueError:
        return False
    except sr.RequestError as e:
        print(f"STT unavailable: {e}")
        return False
    except Exception as e:
        print(f"Semantic engine error: {e}")
        return False


def _cleanup_paths(*paths: Path) -> None:
    for path in paths:
        if path and path.exists():
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _convert_media_to_wav(source_media: Path, target_wav: Path) -> Path:
    """Convert media files to WAV format with improved error handling and logging."""
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[CONVERSION] Converting {source_media.name} to WAV format...")

        completed = subprocess.run(
            [
                ffmpeg_exe,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source_media),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(target_wav),
            ],
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            print(f"[CONVERSION] FFmpeg error: {stderr}")
            raise RuntimeError(f"Media conversion failed: {stderr or f'ffmpeg exited with code {completed.returncode}'}")

        if not target_wav.exists():
            raise RuntimeError("Conversion completed but output file not found")

        file_size = target_wav.stat().st_size
        if file_size <= 44:  # WAV header is 44 bytes minimum
            raise RuntimeError(f"Output file too small ({file_size} bytes) - likely conversion failed")

        print(f"[CONVERSION] Successfully converted to {target_wav.name} ({file_size} bytes)")
        return target_wav

    except ImportError as e:
        print(f"[CONVERSION] Missing dependency: {e}")
        raise RuntimeError("FFmpeg not available for media conversion")
    except Exception as e:
        print(f"[CONVERSION] Media conversion failed: {e}")
        # Don't fall back to original - let the error propagate for proper handling
        raise RuntimeError(f"Could not convert media file: {e}")


def _finalize_detection_result(
    result: dict,
    analysis_params: dict[str, float | str],
    *,
    process_path: Path | None = None,
    force_alert: bool = False,
    include_spectrogram: bool = False,
    include_transcript_analysis: bool = False,
    persist_alert: bool = False,
) -> dict:
    finalized = dict(result)
    semantic_override_applied = False

    if force_alert:
        finalized["synthetic_probability"] = 0.98
        finalized["human_probability"] = 0.02
        finalized["alert"] = True
        finalized["threshold"] = max(DEEPFAKE_THRESHOLD, float(analysis_params["decision_floor"]))
        print("[DEMO MODE] Forced alert enabled.")
    else:
        if (
            SEMANTIC_OVERRIDE_ENABLED
            and process_path is not None
            and process_path.suffix.lower() in STT_COMPATIBLE_EXTENSIONS
            and finalized.get("max_amplitude", 0.0) >= 0.005
            and semantic_deepfake_check(process_path)
        ):
            finalized["synthetic_probability"] = 0.99
            finalized["human_probability"] = 0.01
            finalized["alert"] = True
            semantic_override_applied = True
            print("[SEMANTIC OVERRIDE] Speaker claimed to be AI.")

    effective_threshold = max(
        float(finalized.get("threshold", DEEPFAKE_THRESHOLD)),
        float(analysis_params["decision_floor"]),
    )
    finalized["threshold"] = round(effective_threshold, 4)
    finalized["alert"] = bool(finalized["synthetic_probability"] > effective_threshold)
    finalized["analysis_profile"] = str(analysis_params["profile"])

    merged_analysis_params = dict(finalized.get("analysis_parameters") or {})
    merged_analysis_params.update(
        {
            "decision_floor": round(float(analysis_params["decision_floor"]), 4),
            "borderline_margin": round(float(analysis_params["borderline_margin"]), 4),
        }
    )
    finalized["analysis_parameters"] = merged_analysis_params

    if finalized["alert"]:
        finalized["verdict"] = "ai_clone"
    elif finalized["synthetic_probability"] >= (
        effective_threshold - float(analysis_params["borderline_margin"])
    ):
        finalized["verdict"] = "borderline_human"
    else:
        finalized["verdict"] = "human"

    finalized["decision_mode"] = "model_artifact_fusion"
    finalized["semantic_override_enabled"] = SEMANTIC_OVERRIDE_ENABLED
    finalized["semantic_override_applied"] = semantic_override_applied
    finalized["decision_margin"] = round(float(finalized["synthetic_probability"]) - effective_threshold, 4)
    if semantic_override_applied:
        finalized["decision_summary"] = "Semantic override raised the verdict after AI self-identification cues."
    elif finalized["alert"]:
        finalized["decision_summary"] = "Voice risk exceeded the alert threshold using model and artifact fusion."
    elif finalized["verdict"] == "borderline_human":
        finalized["decision_summary"] = "Voice stayed just below the alert threshold and remains borderline."
    else:
        finalized["decision_summary"] = "Voice stayed below the alert threshold on its own model reading."

    transcript_preview = ""
    keyword_hits: list[str] = []
    fraud_language_probability = 0.0
    fraud_language_alert = False
    alert_reasons = list(finalized.get("alert_reasons") or [])
    if finalized["alert"] and "voice_clone" not in alert_reasons:
        alert_reasons.append("voice_clone")

    if (
        include_transcript_analysis
        and process_path is not None
        and process_path.suffix.lower() in STT_COMPATIBLE_EXTENSIONS
        and float(finalized.get("max_amplitude", 0.0)) >= 0.008
    ):
        transcript_preview = _transcribe_audio_file(process_path)
        keyword_hits = _extract_scam_keyword_hits(transcript_preview)
        fraud_language_probability = _score_transcript_scam_risk(keyword_hits)
        fraud_language_alert = fraud_language_probability >= LIVE_SCAM_ALERT_THRESHOLD
        if fraud_language_alert:
            if not finalized["alert"]:
                finalized["alert"] = True
                finalized["verdict"] = "fraud_language"
            if "fraud_language" not in alert_reasons:
                alert_reasons.append("fraud_language")

    finalized["transcript_preview"] = transcript_preview
    finalized["fraud_keywords"] = sorted(set(keyword_hits))
    finalized["fraud_language_probability"] = round(fraud_language_probability, 4)
    finalized["fraud_language_alert"] = bool(fraud_language_alert)
    finalized["alert_reasons"] = alert_reasons

    if include_spectrogram and process_path is not None and process_path.suffix.lower() != ".webm":
        finalized["spectrogram_base64"] = generate_spectrogram_base64(process_path)

    if persist_alert and finalized["alert"] and process_path is not None:
        FLAGGED_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = uuid.uuid4().hex[:8]
        safe_prob = int(float(finalized["synthetic_probability"]) * 100)
        flagged_path = FLAGGED_DIR / f"deepfake_log_{timestamp}_{request_id}_prob{safe_prob}.wav"
        shutil.copy(process_path, flagged_path)
        print(f"[FORENSICS] Logged suspicious call: {flagged_path}")

    return finalized


def _run_detection_for_uploaded_file(
    file,
    analysis_params: dict[str, float | str],
    *,
    force_alert: bool,
    include_transcript_analysis: bool,
    include_spectrogram: bool,
    persist_alert: bool,
) -> dict:
    request_id = uuid.uuid4().hex
    ext = Path(file.filename or "").suffix.lower() or ".wav"
    raw_path = UPLOAD_DIR / f"{request_id}{ext}"
    wav_path = UPLOAD_DIR / f"{request_id}.wav"

    process_path = raw_path

    try:
        file.save(raw_path)

        if ext in VIDEO_COMPATIBLE_EXTENSIONS:
            process_path = _convert_media_to_wav(raw_path, wav_path)

        result = MODEL_RUNTIME.predict_file(
            audio_path=str(process_path),
            threshold=DEEPFAKE_THRESHOLD,
            chunk_seconds=float(analysis_params["chunk_seconds"]),
            hop_seconds=float(analysis_params["hop_seconds"]),
            sensitivity=float(analysis_params["sensitivity"]),
            model_weight=float(analysis_params["model_weight"]),
            artifact_weight=float(analysis_params["artifact_weight"]),
            pooling_method=str(analysis_params["pooling_method"]),
            duration_penalty_coefficient=DURATION_PENALTY_COEFFICIENT,
            hab_pitch_variance_threshold=float(analysis_params["hab_pitch_variance_threshold"]),
            hab_penalty=float(analysis_params["hab_penalty"]),
        )
        result = _finalize_detection_result(
            result,
            analysis_params,
            process_path=process_path,
            force_alert=force_alert,
            include_spectrogram=include_spectrogram,
            include_transcript_analysis=include_transcript_analysis,
            persist_alert=persist_alert,
        )
        return result
    finally:
        if process_path == raw_path:
            _cleanup_paths(raw_path)
        else:
            _cleanup_paths(raw_path, process_path)


LIVE_MONITOR_WINDOW_SECONDS = float(os.getenv("LIVE_MONITOR_WINDOW_SECONDS", "2.0"))
LIVE_MONITOR_HOP_SECONDS = float(os.getenv("LIVE_MONITOR_HOP_SECONDS", "0.35"))
LIVE_MONITOR_MAX_BUFFER_SECONDS = float(os.getenv("LIVE_MONITOR_MAX_BUFFER_SECONDS", "8.0"))
LIVE_MONITOR_MIN_SECONDS = float(os.getenv("LIVE_MONITOR_MIN_SECONDS", "0.6"))
LIVE_SEMANTIC_WINDOW_SECONDS = float(os.getenv("LIVE_SEMANTIC_WINDOW_SECONDS", "2.8"))
LIVE_SEMANTIC_HOP_SECONDS = float(os.getenv("LIVE_SEMANTIC_HOP_SECONDS", "2.2"))
LIVE_SCAM_ALERT_THRESHOLD = float(os.getenv("LIVE_SCAM_ALERT_THRESHOLD", "0.62"))
LIVE_SEMANTIC_ENABLED = os.getenv("LIVE_SEMANTIC_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
LIVE_MONITOR_SESSIONS_LOCK = threading.Lock()


@dataclass
class LiveMonitorSession:
    sid: str
    sample_rate: int
    analysis_params: dict[str, float | str]
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    total_samples: int = 0
    last_analyzed_total_samples: int = 0
    inference_in_flight: bool = False
    semantic_in_flight: bool = False
    last_chunk_received_at: float = field(default_factory=time.time)
    last_semantic_total_samples: int = 0
    latest_transcript: str = ""
    latest_keywords: list[str] = field(default_factory=list)
    keyword_counts: dict[str, int] = field(default_factory=dict)
    fraud_language_probability: float = 0.0
    fraud_language_alert: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enqueue_chunk(
        self,
        chunk: np.ndarray,
        *,
        captured_at_ms: float | None = None,
    ) -> tuple[np.ndarray, int, dict[str, float | str], float | None] | None:
        samples = np.asarray(chunk, dtype=np.float32)
        if samples.size == 0:
            return None

        with self.lock:
            self.buffer = np.concatenate([self.buffer, samples])
            self.total_samples += int(samples.size)
            self.last_chunk_received_at = time.time()

            max_samples = int(self.sample_rate * LIVE_MONITOR_MAX_BUFFER_SECONDS)
            if self.buffer.size > max_samples:
                self.buffer = self.buffer[-max_samples:]

            if self.inference_in_flight:
                return None

            if self.buffer.size < int(self.sample_rate * LIVE_MONITOR_MIN_SECONDS):
                return None

            min_hop_samples = int(self.sample_rate * LIVE_MONITOR_HOP_SECONDS)
            if (self.total_samples - self.last_analyzed_total_samples) < min_hop_samples:
                return None

            window_samples = min(int(self.sample_rate * LIVE_MONITOR_WINDOW_SECONDS), self.buffer.size)
            analysis_audio = self.buffer[-window_samples:].copy()
            self.last_analyzed_total_samples = self.total_samples
            self.inference_in_flight = True
            return analysis_audio, self.sample_rate, dict(self.analysis_params), captured_at_ms

    def mark_inference_complete(self) -> None:
        with self.lock:
            self.inference_in_flight = False

    def maybe_semantic_window(self) -> tuple[np.ndarray, int] | None:
        with self.lock:
            if self.semantic_in_flight:
                return None

            if self.buffer.size < int(self.sample_rate * 1.2):
                return None

            semantic_hop_samples = int(self.sample_rate * LIVE_SEMANTIC_HOP_SECONDS)
            if (self.total_samples - self.last_semantic_total_samples) < semantic_hop_samples:
                return None

            window_samples = min(int(self.sample_rate * LIVE_SEMANTIC_WINDOW_SECONDS), self.buffer.size)
            semantic_audio = self.buffer[-window_samples:].copy()
            self.last_semantic_total_samples = self.total_samples
            self.semantic_in_flight = True
            return semantic_audio, self.sample_rate

    def mark_semantic_complete(self) -> None:
        with self.lock:
            self.semantic_in_flight = False

    def update_semantic_state(self, transcript: str, keyword_hits: list[str]) -> dict:
        with self.lock:
            cleaned_hits = sorted(set(keyword_hits))
            if transcript:
                self.latest_transcript = transcript
            self.latest_keywords = cleaned_hits

            for keyword in cleaned_hits:
                self.keyword_counts[keyword] = self.keyword_counts.get(keyword, 0) + 1

            repeated_keywords = sorted(
                [keyword for keyword, count in self.keyword_counts.items() if count >= 2],
                key=lambda item: (-self.keyword_counts[item], item),
            )
            max_repeat = max(self.keyword_counts.values(), default=0)
            unique_keywords = len(self.keyword_counts)
            total_keyword_hits = sum(self.keyword_counts.values())
            high_risk_seen = any(keyword in HIGH_RISK_SCAM_KEYWORDS for keyword in self.keyword_counts)
            high_risk_repeated = any(keyword in HIGH_RISK_SCAM_KEYWORDS for keyword in repeated_keywords)

            score = 0.0
            if unique_keywords >= 1:
                score += 0.18
            if unique_keywords >= 2:
                score += 0.12
            if repeated_keywords:
                score += 0.18
            if len(repeated_keywords) >= 2:
                score += 0.10
            if max_repeat >= 3:
                score += 0.16
            if high_risk_seen:
                score += 0.10
            if high_risk_repeated:
                score += 0.12
            if total_keyword_hits >= 4:
                score += 0.08

            self.fraud_language_probability = float(np.clip(score, 0.0, 0.99))
            self.fraud_language_alert = self.fraud_language_probability >= LIVE_SCAM_ALERT_THRESHOLD

            return {
                "transcript_preview": self.latest_transcript,
                "fraud_keywords": list(self.latest_keywords),
                "repeated_keywords": repeated_keywords,
                "scam_keyword_counts": dict(sorted(self.keyword_counts.items(), key=lambda item: (-item[1], item[0]))),
                "fraud_language_probability": round(self.fraud_language_probability, 4),
                "fraud_language_alert": bool(self.fraud_language_alert),
            }

    def semantic_snapshot(self) -> dict:
        with self.lock:
            repeated_keywords = sorted(
                [keyword for keyword, count in self.keyword_counts.items() if count >= 2],
                key=lambda item: (-self.keyword_counts[item], item),
            )
            return {
                "transcript_preview": self.latest_transcript,
                "fraud_keywords": list(self.latest_keywords),
                "repeated_keywords": repeated_keywords,
                "scam_keyword_counts": dict(sorted(self.keyword_counts.items(), key=lambda item: (-item[1], item[0]))),
                "fraud_language_probability": round(self.fraud_language_probability, 4),
                "fraud_language_alert": bool(self.fraud_language_alert),
            }


LIVE_MONITOR_SESSIONS: dict[str, LiveMonitorSession] = {}


def _get_live_monitor_session(sid: str) -> LiveMonitorSession | None:
    with LIVE_MONITOR_SESSIONS_LOCK:
        return LIVE_MONITOR_SESSIONS.get(sid)


def _remove_live_monitor_session(sid: str) -> None:
    with LIVE_MONITOR_SESSIONS_LOCK:
        LIVE_MONITOR_SESSIONS.pop(sid, None)


def _process_live_monitor_window(
    sid: str,
    audio_window: np.ndarray,
    sample_rate: int,
    analysis_params: dict[str, float | str],
    captured_at_ms: float | None,
) -> None:
    session = _get_live_monitor_session(sid)
    if session is None:
        return

    started_at = time.perf_counter()
    try:
        result = MODEL_RUNTIME.predict_waveform(
            audio=audio_window,
            sample_rate=sample_rate,
            threshold=DEEPFAKE_THRESHOLD,
            chunk_seconds=float(analysis_params["chunk_seconds"]),
            hop_seconds=float(analysis_params["hop_seconds"]),
            sensitivity=float(analysis_params["sensitivity"]),
            model_weight=float(analysis_params["model_weight"]),
            artifact_weight=float(analysis_params["artifact_weight"]),
            pooling_method=str(analysis_params["pooling_method"]),
            duration_penalty_coefficient=DURATION_PENALTY_COEFFICIENT,
            hab_pitch_variance_threshold=float(analysis_params["hab_pitch_variance_threshold"]),
            hab_penalty=float(analysis_params["hab_penalty"]),
        )
        finalized = _finalize_detection_result(
            result,
            analysis_params,
            include_spectrogram=False,
            persist_alert=False,
        )
        semantic_snapshot = session.semantic_snapshot()
        finalized.update(semantic_snapshot)
        alert_reasons: list[str] = []
        if finalized["alert"]:
            alert_reasons.append("voice_clone")
        if semantic_snapshot["fraud_language_alert"]:
            finalized["alert"] = True
            alert_reasons.append("fraud_language")
            if finalized.get("verdict") == "human":
                finalized["verdict"] = "fraud_language"
        finalized["alert_reasons"] = alert_reasons
        finalized["processing_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        finalized["buffered_seconds"] = round(float(len(audio_window) / max(sample_rate, 1)), 3)
        if captured_at_ms is not None:
            finalized["latency_ms"] = round(max(0.0, time.time() * 1000 - float(captured_at_ms)), 2)

        socketio.emit("call_monitor_result", finalized, to=sid)
    except Exception as e:
        socketio.emit(
            "call_monitor_error",
            {"error": str(e), "detail": "Live monitor inference failed. The socket will stay open and retry."},
            to=sid,
        )
    finally:
        session = _get_live_monitor_session(sid)
        if session is not None:
            session.mark_inference_complete()


def _refresh_live_semantic_state(sid: str, audio_window: np.ndarray, sample_rate: int) -> None:
    session = _get_live_monitor_session(sid)
    if session is None:
        return

    try:
        transcript = _transcribe_waveform(audio_window, sample_rate)
        keyword_hits = _extract_scam_keyword_hits(transcript)
        session.update_semantic_state(transcript, keyword_hits)
    except Exception as e:
        print(f"Live semantic refresh failed: {e}")
    finally:
        session = _get_live_monitor_session(sid)
        if session is not None:
            session.mark_semantic_complete()


@socketio.on("connect")
def handle_socket_connect():
    socketio.emit(
        "call_monitor_ready",
        {"ready": MODEL_RUNTIME.is_ready(), "device": str(DEVICE)},
        to=request.sid,
    )


@socketio.on("call_monitor_start")
def handle_call_monitor_start(payload: dict | None = None):
    payload = payload or {}
    sample_rate = int(payload.get("sample_rate") or 16000)
    analysis_payload = payload.get("analysis") or {}
    analysis_params = _extract_analysis_params(analysis_payload)
    session = LiveMonitorSession(
        sid=request.sid,
        sample_rate=max(8000, min(sample_rate, 96000)),
        analysis_params=analysis_params,
    )

    with LIVE_MONITOR_SESSIONS_LOCK:
        LIVE_MONITOR_SESSIONS[request.sid] = session

    socketio.emit(
        "call_monitor_status",
        {
            "state": "active",
            "message": "Microphone connected. Waiting for live speaker audio.",
            "analysis_profile": analysis_params["profile"],
        },
        to=request.sid,
    )


@socketio.on("call_monitor_chunk")
def handle_call_monitor_chunk(payload: dict | None = None):
    session = _get_live_monitor_session(request.sid)
    if session is None or not payload:
        return

    audio_payload = payload.get("audio")
    if audio_payload is None:
        return

    if isinstance(audio_payload, (bytes, bytearray, memoryview)):
        chunk = np.frombuffer(audio_payload, dtype=np.float32)
    else:
        chunk = np.asarray(audio_payload, dtype=np.float32)

    work_item = session.enqueue_chunk(chunk, captured_at_ms=payload.get("captured_at_ms"))
    if work_item is not None:
        socketio.start_background_task(_process_live_monitor_window, request.sid, *work_item)

    if LIVE_SEMANTIC_ENABLED:
        semantic_item = session.maybe_semantic_window()
        if semantic_item is not None:
            socketio.start_background_task(_refresh_live_semantic_state, request.sid, *semantic_item)


@socketio.on("call_monitor_stop")
def handle_call_monitor_stop():
    _remove_live_monitor_session(request.sid)
    socketio.emit(
        "call_monitor_status",
        {"state": "stopped", "message": "Live stream stopped."},
        to=request.sid,
    )


@socketio.on("disconnect")
def handle_socket_disconnect():
    _remove_live_monitor_session(request.sid)


@app.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL_RUNTIME.is_ready(),
        "ui_mode": "lovable-dist" if LOVABLE_DIST_DIR else "legacy-templates",
        "default_analysis_profile": DEFAULT_ANALYSIS_PROFILE,
        "default_alert_floor": ALERT_DEFAULT_FLOOR,
        "semantic_override_enabled": SEMANTIC_OVERRIDE_ENABLED,
        "live_semantic_enabled": LIVE_SEMANTIC_ENABLED,
    }, 200


@app.route("/legacy", methods=["GET"])
def legacy_index():
    return render_template("index.html")


@app.route("/legacy/mobile", methods=["GET"])
def legacy_mobile():
    return render_template("mobile.html")


@app.route("/detect_voice", methods=["POST"])
def detect_voice():
    """Enhanced file scanning with better error handling and validation."""
    print(f"[DETECT_VOICE] Received analysis request at {datetime.datetime.now()}")

    if not MODEL_RUNTIME.is_ready():
        error_msg = MODEL_RUNTIME.load_error or "AI model failed to load."
        print(f"[DETECT_VOICE] Model not ready: {error_msg}")
        return jsonify({"error": error_msg}), 500

    if "file" not in request.files:
        print("[DETECT_VOICE] No file part in request")
        return jsonify({"error": "No 'file' part provided in request."}), 400

    file = request.files["file"]
    if not file.filename:
        print("[DETECT_VOICE] Empty filename")
        return jsonify({"error": "No file selected."}), 400

    request_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower() or ".wav"
    raw_path = UPLOAD_DIR / f"{request_id}{ext}"
    wav_path = UPLOAD_DIR / f"{request_id}.wav"

    force_alert = request.form.get("force_alert", "false").lower() == "true"
    enable_transcript_analysis = request.form.get("enable_transcript_analysis", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    analysis_params = _extract_analysis_params(request.form)

    process_path = raw_path

    try:
        file.save(raw_path)

        if ext == ".webm":
            process_path = _convert_media_to_wav(raw_path, wav_path)

        result = MODEL_RUNTIME.predict_file(
            audio_path=str(process_path),
            threshold=DEEPFAKE_THRESHOLD,
            chunk_seconds=float(analysis_params["chunk_seconds"]),
            hop_seconds=float(analysis_params["hop_seconds"]),
            sensitivity=float(analysis_params["sensitivity"]),
            model_weight=float(analysis_params["model_weight"]),
            artifact_weight=float(analysis_params["artifact_weight"]),
            pooling_method=str(analysis_params["pooling_method"]),
            duration_penalty_coefficient=DURATION_PENALTY_COEFFICIENT,
            hab_pitch_variance_threshold=float(analysis_params["hab_pitch_variance_threshold"]),
            hab_penalty=float(analysis_params["hab_penalty"]),
        )
        result = _finalize_detection_result(
            result,
            analysis_params,
            process_path=process_path,
            force_alert=force_alert,
            include_spectrogram=True,
            include_transcript_analysis=enable_transcript_analysis,
            persist_alert=True,
        )

        print(
            "[INFERENCE] "
            f"synthetic={result['synthetic_probability']:.4f} "
            f"model={result['model_probability']:.4f} "
            f"artifact={result['artifact_probability']:.4f} "
            f"threshold={result['threshold']:.4f} "
            f"chunks={result['chunk_count']} "
            f"profile={analysis_params['profile']} "
            f"sensitivity={analysis_params['sensitivity']:.2f}"
        )

        return jsonify(result)
    except Exception as e:
        message = str(e)
        lowered = message.lower()
        decode_error_tokens = (
            "ffmpeg",
            "webm",
            "decode",
            "invalid data",
            "moov",
            "could not decode",
        )
        if any(token in lowered for token in decode_error_tokens):
            print(f"[INFERENCE] Skipped unreadable live chunk: {message}")
            return jsonify(
                {
                    "synthetic_probability": 0.5,
                    "human_probability": 0.5,
                    "alert": False,
                    "threshold": DEEPFAKE_THRESHOLD,
                    "verdict": "inconclusive",
                    "processing_state": "inconclusive_chunk",
                    "decision_mode": "fallback_decode_guard",
                    "decision_summary": "Unreadable audio chunk skipped. Continuing with the next live chunk.",
                }
            )

        traceback.print_exc()
        return jsonify({"error": message}), 500
    finally:
        if process_path == raw_path:
            _cleanup_paths(raw_path)
        else:
            _cleanup_paths(raw_path, process_path)


@app.route("/feedback", methods=["POST"])
def feedback():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    label = request.form.get("label")
    if label not in {"human", "ai"}:
        return jsonify({"error": "Invalid label. Use 'human' or 'ai'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Invalid file."}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_name = secure_filename(file.filename)
    save_dir = CONTINUOUS_DATASET_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"feedback_{timestamp}_{uuid.uuid4().hex[:8]}_{cleaned_name}"
    file.save(save_path)

    print(f"[CONTINUOUS LEARNING] Stored labeled clip -> {label.upper()} at {save_path}")

    return jsonify({"success": True, "message": "Feedback recorded."})


if LOVABLE_DIST_DIR:

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_lovable(path: str):
        # Protect API routes from being shadowed by the SPA fallback.
        api_prefixes = ("detect_voice", "feedback", "health", "legacy", "socket.io")
        if path.startswith(api_prefixes):
            abort(404)

        if path:
            asset_path = LOVABLE_DIST_DIR / path
            if asset_path.exists() and asset_path.is_file():
                return send_from_directory(str(LOVABLE_DIST_DIR), path)

        return send_from_directory(str(LOVABLE_DIST_DIR), "index.html")

else:

    @app.route("/", methods=["GET"])
    def index_fallback():
        return redirect("/legacy")

    @app.route("/mobile", methods=["GET"])
    def mobile_fallback():
        return redirect("/legacy/mobile")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)
