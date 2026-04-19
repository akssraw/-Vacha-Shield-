import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import torch

from feature_extraction import extract_dual_channel_from_waveform
from utils import segmented_weighted_inference

PRIMARY_RESAMPLE_TYPE = "kaiser_fast"
FALLBACK_RESAMPLE_TYPE = "polyphase"


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), low, high))


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _needs_resampy_fallback(exc: Exception) -> bool:
    return isinstance(exc, ModuleNotFoundError) and "resampy" in str(exc).lower()


def _librosa_load_resilient(audio_path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        return librosa.load(str(audio_path), sr=sample_rate, res_type=PRIMARY_RESAMPLE_TYPE)
    except Exception as exc:
        if not _needs_resampy_fallback(exc):
            raise
        return librosa.load(str(audio_path), sr=sample_rate, res_type=FALLBACK_RESAMPLE_TYPE)


def _librosa_resample_resilient(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        return librosa.resample(
            audio,
            orig_sr=int(orig_sr),
            target_sr=int(target_sr),
            res_type=PRIMARY_RESAMPLE_TYPE,
        )
    except Exception as exc:
        if not _needs_resampy_fallback(exc):
            raise
        return librosa.resample(
            audio,
            orig_sr=int(orig_sr),
            target_sr=int(target_sr),
            res_type=FALLBACK_RESAMPLE_TYPE,
        )


def _chunk_signal(audio: np.ndarray, chunk_samples: int, hop_samples: int) -> list[np.ndarray]:
    if audio is None or len(audio) == 0:
        return []

    chunks = []
    for start in range(0, len(audio), hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        if end >= len(audio):
            break
    return chunks


def _voice_activity_ratio(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
    if rms.size == 0:
        return 0.0
    threshold = max(0.0015, float(np.mean(rms) * 0.45))
    return float(np.mean(rms > threshold))


def _chunk_quality_weight(chunk: np.ndarray) -> float:
    if chunk is None or len(chunk) == 0:
        return 0.2

    rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-8)
    vad_ratio = _voice_activity_ratio(chunk)

    # We prefer chunks that are audible and contain active speech.
    rms_score = float(np.clip(rms * 6.0, 0.2, 1.0))
    vad_score = float(np.clip(vad_ratio * 1.4, 0.2, 1.0))
    return float(np.clip(0.55 * rms_score + 0.45 * vad_score, 0.2, 1.0))


def _convert_with_ffmpeg(source_path: Path, sample_rate: int) -> Path:
    import imageio_ffmpeg

    temp_fd, temp_name = tempfile.mkstemp(prefix="vacha_infer_", suffix=".wav")
    os.close(temp_fd)
    temp_wav_path = Path(temp_name)

    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "wav",
            str(temp_wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not temp_wav_path.exists() or temp_wav_path.stat().st_size <= 44:
        try:
            temp_wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or f"ffmpeg exited with code {completed.returncode}")

    return temp_wav_path


def _load_audio_for_inference(audio_path: str, sample_rate: int) -> tuple[np.ndarray, Path | None]:
    source_path = Path(audio_path)
    load_path = source_path
    temp_wav_path: Path | None = None

    if source_path.suffix.lower() == ".webm":
        try:
            temp_wav_path = _convert_with_ffmpeg(source_path, sample_rate)
            load_path = temp_wav_path
        except Exception as conversion_error:
            try:
                y, _ = _librosa_load_resilient(source_path, sample_rate=sample_rate)
                return np.asarray(y, dtype=np.float32), None
            except Exception as load_error:
                raise RuntimeError(f"Could not decode webm audio chunk: {conversion_error}") from load_error

    y, _ = _librosa_load_resilient(load_path, sample_rate=sample_rate)
    return np.asarray(y, dtype=np.float32), temp_wav_path


def _artifact_probability(audio: np.ndarray, sample_rate: int) -> tuple[float, dict]:
    """
    Handcrafted anti-spoof cues that complement the CNN on unseen clone engines.
    Returns (probability, diagnostics).
    """
    if audio is None or len(audio) == 0:
        return 0.5, {}

    try:
        rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160).flatten()
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=400, hop_length=160).flatten()

        rms_mean = float(np.mean(rms)) if rms.size else 0.0
        rms_std = float(np.std(rms)) if rms.size else 0.0
        rms_cv = rms_std / (rms_mean + 1e-8)
        zcr_std = float(np.std(zcr)) if zcr.size else 0.0

        stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=160)) ** 2
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        high_mask = freqs >= 4000
        total_energy = float(np.sum(stft) + 1e-8)
        high_energy = float(np.sum(stft[high_mask, :])) if stft.size else 0.0
        high_freq_ratio = high_energy / total_energy

        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = float(np.mean(spectral_contrast)) if spectral_contrast.size else 0.0
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85, hop_length=160).flatten()
        rolloff_mean = float(np.mean(rolloff)) if rolloff.size else 0.0
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=160).flatten()
        flatness_mean = float(np.mean(spectral_flatness)) if spectral_flatness.size else 0.0

        vad_ratio = _voice_activity_ratio(audio)
        duration_seconds = float(len(audio) / sample_rate)

        score = 0.5

        # Deep-cloned speech often has lower short-term dynamics and smoother contours.
        if rms_cv < 0.20:
            score += 0.14
        elif rms_cv < 0.28:
            score += 0.07
        elif rms_cv > 0.46:
            score -= 0.10

        if zcr_std < 0.014:
            score += 0.10
        elif zcr_std < 0.020:
            score += 0.05
        elif zcr_std > 0.035:
            score -= 0.08

        if high_freq_ratio < 0.07:
            score += 0.10
        elif high_freq_ratio < 0.11:
            score += 0.05
        elif high_freq_ratio > 0.19:
            score -= 0.06

        if contrast_mean < 15.0:
            score += 0.08
        elif contrast_mean > 21.5:
            score -= 0.06

        if vad_ratio > 0.93:
            score += 0.05
        elif vad_ratio < 0.38:
            score += 0.03

        if duration_seconds < 2.5:
            score += 0.04

        score = float(np.clip(score, 0.05, 0.95))

        diagnostics = {
            "rms_cv": round(rms_cv, 5),
            "zcr_std": round(zcr_std, 5),
            "high_freq_ratio": round(high_freq_ratio, 5),
            "spectral_contrast": round(contrast_mean, 5),
            "rolloff_mean": round(rolloff_mean, 2),
            "spectral_flatness": round(flatness_mean, 5),
            "vad_ratio": round(vad_ratio, 5),
        }

        return score, diagnostics
    except Exception:
        return 0.5, {}


def _round_float_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _predict_deepfake_from_resampled_waveform(
    audio: np.ndarray,
    sample_rate: int,
    model,
    device,
    threshold: float = 0.50,
    chunk_seconds: float = 3.0,
    hop_seconds: float = 1.0,
    sensitivity: float = 0.50,
    model_weight: float = 0.82,
    artifact_weight: float = 0.18,
    low_amplitude_guard: float = 0.01,
    stability_bonus: float = 0.02,
    pooling_method: str = "median",
    duration_penalty_coefficient: float = 0.0,
    hab_pitch_variance_threshold: float = 0.30,
    hab_penalty: float = 0.15,
) -> dict:
    # The model window is intentionally fixed at 3s with a 1s stride for drift-resistant inference.
    chunk_seconds = _clamp(_safe_float(chunk_seconds, 3.0), 3.0, 3.0)
    hop_seconds = _clamp(_safe_float(hop_seconds, 1.0), 1.0, 1.0)
    pooling_method = str(pooling_method or "median").strip().lower()
    if pooling_method not in {"mean", "median"}:
        pooling_method = "median"

    sensitivity = _clamp(_safe_float(sensitivity, 0.5), 0.0, 1.0)
    model_weight = _clamp(_safe_float(model_weight, 0.82), 0.05, 0.95)
    artifact_weight = _clamp(_safe_float(artifact_weight, 0.18), 0.05, 0.95)
    weight_sum = model_weight + artifact_weight
    model_weight = model_weight / weight_sum
    artifact_weight = artifact_weight / weight_sum
    low_amplitude_guard = _clamp(_safe_float(low_amplitude_guard, 0.01), 0.001, 0.05)
    stability_bonus = _clamp(_safe_float(stability_bonus, 0.02), 0.0, 0.08)
    duration_penalty_coefficient = _clamp(_safe_float(duration_penalty_coefficient, 0.0), 0.0, 0.5)
    hab_pitch_variance_threshold = _clamp(_safe_float(hab_pitch_variance_threshold, 0.30), 0.0, 1.0)
    hab_penalty = _clamp(_safe_float(hab_penalty, 0.15), 0.0, 0.9)

    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0:
        return {
            "synthetic_probability": 0.0,
            "human_probability": 1.0,
            "alert": False,
            "threshold": round(threshold, 4),
            "model_probability": 0.0,
            "artifact_probability": 0.5,
            "chunk_probability_mean": 0.0,
            "chunk_probability_std": 0.0,
            "chunk_count": 0,
            "window_probabilities": [],
            "raw_window_probabilities": [],
            "window_weights": [],
            "audio_duration_seconds": 0.0,
            "max_amplitude": 0.0,
            "artifact_signals": {},
            "analysis_parameters": {
                "chunk_seconds": round(chunk_seconds, 3),
                "hop_seconds": round(hop_seconds, 3),
                "window_seconds": round(chunk_seconds, 3),
                "stride_seconds": round(hop_seconds, 3),
                "pooling_method": pooling_method,
                "sensitivity": round(sensitivity, 3),
                "model_weight": round(model_weight, 4),
                "artifact_weight": round(artifact_weight, 4),
                "duration_penalty_coefficient": round(duration_penalty_coefficient, 4),
                "hab_penalty": round(hab_penalty, 4),
                "hab_pitch_variance_threshold": round(hab_pitch_variance_threshold, 4),
            },
        }

    max_amp = float(np.max(np.abs(y)))
    if max_amp > 0.0:
        y = librosa.util.normalize(y)

    segment_result = segmented_weighted_inference(
        audio=y,
        sample_rate=sample_rate,
        model=model,
        device=device,
        feature_extractor=extract_dual_channel_from_waveform,
        window_duration=chunk_seconds,
        stride_duration=hop_seconds,
        pooling_method=pooling_method,
        apply_hab=True,
        hab_pitch_variance_threshold=hab_pitch_variance_threshold,
        hab_penalty=hab_penalty,
        duration_penalty_coefficient=duration_penalty_coefficient,
    )

    chunk_probs = list(segment_result.get("window_probabilities") or [])
    raw_chunk_probs = list(segment_result.get("raw_window_probabilities") or [])
    chunk_weights = list(segment_result.get("window_weights") or [])
    if not chunk_probs:
        chunk_probs = [0.0]
        raw_chunk_probs = [0.0]
        chunk_weights = [1.0]
        segment_result["pooled_probability"] = 0.0

    probs_np = np.asarray(chunk_probs, dtype=np.float32)
    raw_probs_np = np.asarray(raw_chunk_probs, dtype=np.float32)
    model_probability = float(np.clip(float(segment_result.get("pooled_probability", 0.0)), 0.0, 1.0))

    artifact_probability, artifact_diagnostics = _artifact_probability(y, sample_rate=sample_rate)

    fused_probability = float(
        np.clip(model_weight * model_probability + artifact_weight * artifact_probability, 0.0, 1.0)
    )

    duration_seconds = float(len(y) / sample_rate)
    rolloff_mean = float(artifact_diagnostics.get("rolloff_mean", 0.0))
    flatness_mean = float(artifact_diagnostics.get("spectral_flatness", 0.0))
    robustness_penalty = 0.0
    if duration_seconds < 6.5 and model_probability > 0.84 and flatness_mean > 0.018 and rolloff_mean > 2700:
        robustness_penalty = 0.13
        fused_probability = float(np.clip(fused_probability - robustness_penalty, 0.0, 1.0))

    chunk_std = float(np.std(probs_np)) if probs_np.size > 1 else 0.0
    dynamic_threshold = threshold

    if max_amp < low_amplitude_guard:
        dynamic_threshold += 0.05

    if duration_seconds <= 30.0 and probs_np.size >= 4 and chunk_std < 0.05:
        dynamic_threshold -= stability_bonus

    dynamic_threshold += (0.5 - sensitivity) * 0.14
    dynamic_threshold = float(np.clip(dynamic_threshold, 0.40, 0.72))
    alert = bool(fused_probability > dynamic_threshold)

    hab_adjustments = list(segment_result.get("hab_adjustments") or [])
    pitch_variances = list(segment_result.get("pitch_variances") or [])
    duration_penalty = float(segment_result.get("duration_penalty", 0.0))
    hab_adjusted_count = sum(1 for value in hab_adjustments if float(value) > 0.0)

    return {
        "synthetic_probability": round(fused_probability, 4),
        "human_probability": round(1.0 - fused_probability, 4),
        "alert": alert,
        "threshold": round(dynamic_threshold, 4),
        "model_probability": round(model_probability, 4),
        "artifact_probability": round(artifact_probability, 4),
        "chunk_probability_mean": round(float(np.mean(probs_np)), 4),
        "chunk_probability_std": round(chunk_std, 4),
        "raw_chunk_probability_mean": round(float(np.mean(raw_probs_np)), 4),
        "chunk_count": int(len(probs_np)),
        "window_probabilities": _round_float_list(chunk_probs),
        "raw_window_probabilities": _round_float_list(raw_chunk_probs),
        "window_weights": _round_float_list(chunk_weights),
        "hab_adjustments": _round_float_list(hab_adjustments),
        "hab_adjusted_window_count": int(hab_adjusted_count),
        "pitch_variance_mean": round(float(np.mean(pitch_variances)), 4) if pitch_variances else 0.0,
        "duration_penalty": round(duration_penalty, 4),
        "audio_duration_seconds": round(duration_seconds, 3),
        "max_amplitude": round(max_amp, 5),
        "artifact_signals": artifact_diagnostics,
        "analysis_parameters": {
            "chunk_seconds": round(chunk_seconds, 3),
            "hop_seconds": round(hop_seconds, 3),
            "window_seconds": round(chunk_seconds, 3),
            "stride_seconds": round(hop_seconds, 3),
            "pooling_method": pooling_method,
            "sensitivity": round(sensitivity, 3),
            "model_weight": round(model_weight, 4),
            "artifact_weight": round(artifact_weight, 4),
            "low_amplitude_guard": round(low_amplitude_guard, 4),
            "stability_bonus": round(stability_bonus, 4),
            "robustness_penalty": round(robustness_penalty, 4),
            "duration_penalty_coefficient": round(duration_penalty_coefficient, 4),
            "hab_penalty": round(hab_penalty, 4),
            "hab_pitch_variance_threshold": round(hab_pitch_variance_threshold, 4),
        },
    }


def predict_deepfake_from_waveform(
    audio: np.ndarray,
    sample_rate: int,
    model,
    device,
    threshold: float = 0.50,
    chunk_seconds: float = 3.0,
    hop_seconds: float = 1.0,
    sensitivity: float = 0.50,
    model_weight: float = 0.82,
    artifact_weight: float = 0.18,
    low_amplitude_guard: float = 0.01,
    stability_bonus: float = 0.02,
    pooling_method: str = "median",
    duration_penalty_coefficient: float = 0.0,
    hab_pitch_variance_threshold: float = 0.30,
    hab_penalty: float = 0.15,
) -> dict:
    target_sample_rate = 16000
    waveform = np.asarray(audio, dtype=np.float32)

    if waveform.size and sample_rate != target_sample_rate:
        waveform = _librosa_resample_resilient(
            waveform,
            orig_sr=int(sample_rate),
            target_sr=target_sample_rate,
        ).astype(np.float32)

    return _predict_deepfake_from_resampled_waveform(
        audio=waveform,
        sample_rate=target_sample_rate,
        model=model,
        device=device,
        threshold=threshold,
        chunk_seconds=chunk_seconds,
        hop_seconds=hop_seconds,
        sensitivity=sensitivity,
        model_weight=model_weight,
        artifact_weight=artifact_weight,
        low_amplitude_guard=low_amplitude_guard,
        stability_bonus=stability_bonus,
        pooling_method=pooling_method,
        duration_penalty_coefficient=duration_penalty_coefficient,
        hab_pitch_variance_threshold=hab_pitch_variance_threshold,
        hab_penalty=hab_penalty,
    )


def predict_deepfake_from_file(
    audio_path: str,
    model,
    device,
    threshold: float = 0.50,
    chunk_seconds: float = 3.0,
    hop_seconds: float = 1.0,
    sensitivity: float = 0.50,
    model_weight: float = 0.82,
    artifact_weight: float = 0.18,
    low_amplitude_guard: float = 0.01,
    stability_bonus: float = 0.02,
    pooling_method: str = "median",
    duration_penalty_coefficient: float = 0.0,
    hab_pitch_variance_threshold: float = 0.30,
    hab_penalty: float = 0.15,
) -> dict:
    temp_wav_path: Path | None = None
    try:
        y, temp_wav_path = _load_audio_for_inference(audio_path, sample_rate=16000)
        return predict_deepfake_from_waveform(
            audio=y,
            sample_rate=16000,
            model=model,
            device=device,
            threshold=threshold,
            chunk_seconds=chunk_seconds,
            hop_seconds=hop_seconds,
            sensitivity=sensitivity,
            model_weight=model_weight,
            artifact_weight=artifact_weight,
            low_amplitude_guard=low_amplitude_guard,
            stability_bonus=stability_bonus,
            pooling_method=pooling_method,
            duration_penalty_coefficient=duration_penalty_coefficient,
            hab_pitch_variance_threshold=hab_pitch_variance_threshold,
            hab_penalty=hab_penalty,
        )
    finally:
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
