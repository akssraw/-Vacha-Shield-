from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from deepfake_detector import predict_deepfake_from_waveform


@dataclass
class CallMonitor:
    model: Any
    device: torch.device | str
    sample_rate: int = 16000
    vad_level: int = 3
    frame_ms: int = 30
    threshold: float = 0.55
    sensitivity: float = 0.74
    model_weight: float = 0.76
    artifact_weight: float = 0.24
    smoothing_window: int = 5
    alert_confirmations: int = 2
    backend: str = "sounddevice"
    _buffer: collections.deque = field(init=False, repr=False)
    _scores: collections.deque = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)
    _pa: Any = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _last_result: dict[str, Any] = field(default_factory=dict, init=False)
    _samples_since_inference: int = field(default=0, init=False, repr=False)
    _positive_streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        import webrtcvad

        if self.sample_rate != 16000:
            raise ValueError("CallMonitor is tuned for 16kHz mono audio.")
        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("webrtcvad supports only 10, 20, or 30 ms frames.")

        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self._buffer = collections.deque(maxlen=48000)
        self._scores = collections.deque(maxlen=max(1, self.smoothing_window))
        self._vad = webrtcvad.Vad(self.vad_level)

    def start_stream(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        if self.backend == "pyaudio":
            self._start_pyaudio()
        else:
            self._start_sounddevice()

    def stop_stream(self) -> None:
        self._stop.set()
        if self._stream is not None:
            stop = getattr(self._stream, "stop", None)
            close = getattr(self._stream, "close", None)
            if callable(stop):
                stop()
            if callable(close):
                close()
        if self._thread:
            self._thread.join(timeout=2)
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def process_frames(self, frames: bytes | np.ndarray) -> bool:
        pcm = self._to_int16_mono(frames)
        usable = (pcm.size // self.frame_samples) * self.frame_samples
        if usable <= 0:
            return False

        if usable == self.frame_samples:
            if not self._vad.is_speech(pcm[:usable].tobytes(), self.sample_rate):
                return False
            speech = pcm[:usable].astype(np.float32) / 32768.0
            with self._lock:
                self._buffer.extend(speech)
                self._samples_since_inference += int(speech.size)
            return True

        speech_frames = []
        for frame in pcm[:usable].reshape(-1, self.frame_samples):
            frame_bytes = frame.tobytes()
            if self._vad.is_speech(frame_bytes, self.sample_rate):
                speech_frames.append(frame)

        if not speech_frames:
            return False

        speech = np.concatenate(speech_frames).astype(np.float32) / 32768.0
        with self._lock:
            self._buffer.extend(speech)
            self._samples_since_inference += int(speech.size)
        return True

    def get_verdict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_result)

    def _start_sounddevice(self) -> None:
        import sounddevice as sd

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                self._last_result = {**self.get_verdict(), "stream_warning": str(status)}
            self.process_frames(indata)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=callback,
        )
        self._stream.start()

    def _start_pyaudio(self) -> None:
        import pyaudio

        self._pa = pyaudio.PyAudio()

        def callback(in_data, _frame_count, _time_info, _status):
            self.process_frames(in_data)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_samples,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.05)
            with self._lock:
                ready = self._samples_since_inference >= 16000 and len(self._buffer) >= self.sample_rate
                if not ready:
                    continue
                self._samples_since_inference = 0
                audio = np.fromiter(self._buffer, dtype=np.float32, count=len(self._buffer))

            result = predict_deepfake_from_waveform(
                audio=audio,
                sample_rate=self.sample_rate,
                model=self.model,
                device=self.device,
                threshold=self.threshold,
                chunk_seconds=3.0,
                hop_seconds=1.0,
                sensitivity=self.sensitivity,
                model_weight=self.model_weight,
                artifact_weight=self.artifact_weight,
                pooling_method="median",
            )
            self._publish_smoothed_result(result)

    def _publish_smoothed_result(self, result: dict[str, Any]) -> None:
        raw_score = float(result.get("synthetic_probability", 0.0))
        self._scores.append(raw_score)
        smoothed = self._confidence_smoothing()
        self._positive_streak = self._positive_streak + 1 if smoothed >= self.threshold else 0

        verdict = dict(result)
        verdict["raw_synthetic_probability"] = round(raw_score, 4)
        verdict["synthetic_probability"] = round(smoothed, 4)
        verdict["human_probability"] = round(1.0 - smoothed, 4)
        verdict["alert"] = self._positive_streak >= self.alert_confirmations
        verdict["smoothing_window"] = len(self._scores)
        verdict["positive_streak"] = self._positive_streak

        with self._lock:
            self._last_result = verdict

    def _confidence_smoothing(self) -> float:
        scores = np.asarray(self._scores, dtype=np.float32)
        if scores.size == 0:
            return 0.0
        weights = np.linspace(0.65, 1.0, scores.size, dtype=np.float32)
        smoothed = float(np.average(scores, weights=weights))
        if scores.size >= 3 and float(np.std(scores)) > 0.18:
            smoothed *= 0.88
        return float(np.clip(smoothed, 0.0, 1.0))

    @staticmethod
    def _to_int16_mono(frames: bytes | np.ndarray) -> np.ndarray:
        if isinstance(frames, (bytes, bytearray, memoryview)):
            return np.frombuffer(frames, dtype=np.int16)

        audio = np.asarray(frames)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.dtype == np.int16:
            return np.ascontiguousarray(audio)
        return np.clip(audio.astype(np.float32) * 32768.0, -32768, 32767).astype(np.int16)
