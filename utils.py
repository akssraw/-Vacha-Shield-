from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch


def create_sliding_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_duration: float = 3.0,
    stride_duration: float = 1.0,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Create fixed-duration sliding windows over a waveform.

    Args:
        audio: Input audio waveform
        sample_rate: Audio sample rate
        window_duration: Duration of each window in seconds (default: 3.0)
        stride_duration: Step between window starts in seconds (default: 1.0)

    Returns:
        List of tuples (window_audio, start_sample, end_sample)
    """
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return []

    window_samples = max(1, int(round(window_duration * sample_rate)))
    stride_samples = max(1, int(round(stride_duration * sample_rate)))
    waveform = np.asarray(audio, dtype=np.float32)

    windows: List[Tuple[np.ndarray, int, int]] = []
    start = 0
    while start < len(waveform):
        end = min(start + window_samples, len(waveform))
        window_audio = waveform[start:end]

        if len(window_audio) < window_samples:
            window_audio = np.pad(window_audio, (0, window_samples - len(window_audio)))

        windows.append((window_audio.astype(np.float32), start, end))
        if end >= len(waveform):
            break
        start += stride_samples

    return windows


def calculate_micro_pitch_variance(audio: np.ndarray, sample_rate: int) -> float:
    """
    Calculate micro-pitch variance (human jitter) for HAB adjustment.

    Args:
        audio: Input audio waveform
        sample_rate: Audio sample rate

    Returns:
        Micro-pitch variance score (0.0 to 1.0)
    """
    if audio is None or len(audio) < 1024 or sample_rate <= 0:
        return 0.0

    try:
        waveform = np.asarray(audio, dtype=np.float32)
        n_fft = min(2048, max(1024, int(2 ** np.ceil(np.log2(min(len(waveform), 2048))))))
        pitches, magnitudes = librosa.piptrack(
            y=waveform,
            sr=sample_rate,
            fmin=75,
            fmax=400,
            hop_length=512,
            n_fft=n_fft,
        )

        if pitches.size == 0 or magnitudes.size == 0:
            return 0.0

        frame_indices = np.argmax(magnitudes, axis=0)
        pitch_values = pitches[frame_indices, np.arange(pitches.shape[1])]
        voiced = pitch_values[pitch_values > 0]
        if voiced.size < 3:
            return 0.0

        pitch_diffs = np.diff(voiced)
        pitch_variance = float(np.std(pitch_diffs) / (np.mean(voiced) + 1e-8))

        return float(np.clip(pitch_variance / 0.02, 0.0, 1.0))
    except Exception:
        return 0.0


def apply_duration_penalty(
    probability: float,
    duration_seconds: float,
    penalty_threshold: float = 30.0,
    penalty_coefficient: float = 0.0,
) -> float:
    """
    Apply a long-duration penalty to combat stochastic drift.

    Args:
        probability: Raw AI probability score
        duration_seconds: Total audio duration
        penalty_threshold: Duration threshold in seconds (default: 30.0)
        penalty_coefficient: Maximum probability reduction once duration reaches
            twice the threshold. Set to 0.0 to disable.

    Returns:
        Adjusted probability score
    """
    probability = float(np.clip(probability, 0.0, 1.0))
    penalty_coefficient = float(np.clip(penalty_coefficient, 0.0, 0.5))
    if duration_seconds <= penalty_threshold or penalty_coefficient <= 0.0:
        return probability

    excess_duration = duration_seconds - penalty_threshold
    penalty_ratio = min(excess_duration / penalty_threshold, 1.0)
    penalty_amount = penalty_ratio * penalty_coefficient

    return max(0.0, probability - penalty_amount)


def segmented_weighted_inference(
    audio: np.ndarray,
    sample_rate: int,
    model,
    device,
    feature_extractor,
    window_duration: float = 3.0,
    stride_duration: float = 1.0,
    pooling_method: str = "median",
    apply_hab: bool = True,
    hab_pitch_variance_threshold: float = 0.3,
    hab_penalty: float = 0.15,
    duration_penalty_coefficient: float = 0.0,
) -> dict:
    """
    Perform segmented weighted inference with fixed sliding windows.

    Args:
        audio: Input audio waveform
        sample_rate: Audio sample rate
        model: PyTorch model for inference
        device: Torch device (CPU/GPU)
        feature_extractor: Function to extract features from audio
        window_duration: Duration of each window in seconds
        stride_duration: Step between window starts in seconds
        pooling_method: "mean" or "median" for probability pooling
        apply_hab: Whether to apply Human Attribute Buffer pitch jitter correction
        hab_pitch_variance_threshold: Pitch variance needed to trigger HAB
        hab_penalty: Relative probability reduction for HAB-positive windows
        duration_penalty_coefficient: Long-duration probability penalty after 30s

    Returns:
        Dictionary with inference results
    """
    windows = create_sliding_windows(audio, sample_rate, window_duration, stride_duration)

    empty_result = {
        "window_probabilities": [],
        "raw_window_probabilities": [],
        "window_weights": [],
        "pooled_probability": 0.5,
        "hab_adjustments": [],
        "pitch_variances": [],
        "duration_penalty": 0.0,
        "window_count": 0,
    }
    if not windows:
        return empty_result

    window_probabilities: list[float] = []
    raw_window_probabilities: list[float] = []
    window_weights: list[float] = []
    hab_adjustments: list[float] = []
    pitch_variances: list[float] = []

    hab_penalty = float(np.clip(hab_penalty, 0.0, 0.9))
    hab_pitch_variance_threshold = float(np.clip(hab_pitch_variance_threshold, 0.0, 1.0))

    with torch.no_grad():
        for window_audio, _start_sample, _end_sample in windows:
            features = feature_extractor(window_audio, sample_rate=sample_rate, max_pad_len=400)
            if features is None:
                continue

            tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            logit = model(tensor)
            raw_prob = float(torch.sigmoid(logit).item())
            prob = raw_prob

            hab_adjustment = 0.0
            micro_pitch_variance = 0.0
            if apply_hab:
                micro_pitch_variance = calculate_micro_pitch_variance(window_audio, sample_rate)
                if micro_pitch_variance >= hab_pitch_variance_threshold:
                    adjusted_prob = float(np.clip(prob * (1.0 - hab_penalty), 0.0, 1.0))
                    hab_adjustment = prob - adjusted_prob
                    prob = adjusted_prob

            weight = calculate_window_quality_weight(window_audio)

            raw_window_probabilities.append(raw_prob)
            window_probabilities.append(prob)
            window_weights.append(weight)
            hab_adjustments.append(hab_adjustment)
            pitch_variances.append(micro_pitch_variance)

    if not window_probabilities:
        return empty_result

    probs_np = np.asarray(window_probabilities, dtype=np.float32)
    weights_np = np.asarray(window_weights, dtype=np.float32)

    if pooling_method == "median":
        pooled_probability = float(weighted_median(probs_np, weights_np))
    else:
        pooled_probability = float(np.average(probs_np, weights=weights_np))

    duration_penalty = 0.0
    if duration_penalty_coefficient > 0.0:
        duration_seconds = len(audio) / max(sample_rate, 1)
        original_probability = pooled_probability
        pooled_probability = apply_duration_penalty(
            pooled_probability,
            duration_seconds,
            penalty_threshold=30.0,
            penalty_coefficient=duration_penalty_coefficient,
        )
        duration_penalty = original_probability - pooled_probability

    return {
        "window_probabilities": window_probabilities,
        "raw_window_probabilities": raw_window_probabilities,
        "window_weights": window_weights,
        "pooled_probability": pooled_probability,
        "hab_adjustments": hab_adjustments,
        "pitch_variances": pitch_variances,
        "duration_penalty": duration_penalty,
        "window_count": len(window_probabilities),
    }


def calculate_window_quality_weight(window_audio: np.ndarray) -> float:
    """
    Calculate quality weight for a window based on audio characteristics.

    Args:
        window_audio: Window audio data

    Returns:
        Quality weight (0.0 to 1.0)
    """
    if window_audio is None or len(window_audio) == 0:
        return 0.2

    rms = np.sqrt(np.mean(np.square(window_audio)))
    rms_score = float(np.clip(rms * 6.0, 0.2, 1.0))

    vad_ratio = calculate_voice_activity_ratio(window_audio)
    vad_score = float(np.clip(vad_ratio * 1.4, 0.2, 1.0))

    weight = 0.55 * rms_score + 0.45 * vad_score

    return float(np.clip(weight, 0.2, 1.0))


def calculate_voice_activity_ratio(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> float:
    """
    Calculate voice activity ratio for audio quality assessment.

    Args:
        audio: Input audio
        frame_length: Frame length for RMS calculation
        hop_length: Hop length for RMS calculation

    Returns:
        Voice activity ratio (0.0 to 1.0)
    """
    if audio is None or len(audio) == 0:
        return 0.0

    try:
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
        if rms.size == 0:
            return 0.0

        threshold = max(0.0015, float(np.mean(rms) * 0.45))
        return float(np.mean(rms > threshold))
    except Exception:
        return 0.0


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted median of probability values.

    Args:
        values: Array of probability values
        weights: Array of corresponding weights

    Returns:
        Weighted median value
    """
    if len(values) == 0:
        return 0.5

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    if total_weight <= 0:
        return float(np.median(sorted_values))

    median_weight = total_weight * 0.5

    median_index = np.searchsorted(cumulative_weights, median_weight)
    if median_index >= len(sorted_values):
        median_index = len(sorted_values) - 1

    return float(sorted_values[median_index])


def extract_segment_features(audio: np.ndarray, sample_rate: int, max_pad_len: int = 400) -> Optional[np.ndarray]:
    """
    Extract features from audio segment for model inference.

    Args:
        audio: Input audio segment
        sample_rate: Audio sample rate
        max_pad_len: Maximum padding length

    Returns:
        Extracted features or None if extraction fails
    """
    try:
        from feature_extraction import extract_dual_channel_from_waveform

        return extract_dual_channel_from_waveform(audio, sample_rate=sample_rate, max_pad_len=max_pad_len)
    except ImportError:
        if audio is None or len(audio) == 0:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        if mfcc.shape[1] < max_pad_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc
    except Exception:
        return None
