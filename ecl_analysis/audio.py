"""Audio management and analysis helpers."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .dependencies import (
    LIBROSA_AVAILABLE,
    PYGAME_AVAILABLE,
    librosa,
    pygame,
)


class AudioManager:
    """Handle all audio feedback in the application."""

    def __init__(self, enabled: bool = True, volume: float = 0.7):
        self.enabled = enabled and PYGAME_AVAILABLE
        self.volume = max(0.0, min(1.0, volume))
        self._initialized = False

        if self.enabled and pygame is not None:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._initialized = True
            except (ImportError, OSError, RuntimeError) as exc:
                logging.warning("Failed to initialize audio: %s", exc)
                self.enabled = False

    def play_analysis_start(self):
        if not self._can_play():
            return
        try:
            self._play_tone_sequence([440, 554, 659], duration=0.15)
        except (RuntimeError, OSError) as exc:
            logging.warning("Failed to play analysis start sound: %s", exc)

    def play_analysis_complete(self):
        if not self._can_play():
            return
        try:
            self._play_tone_sequence([523, 659, 783], duration=0.3)
        except (RuntimeError, OSError) as exc:
            logging.warning("Failed to play analysis complete sound: %s", exc)

    def play_run_detected(self):
        if not self._can_play():
            return
        try:
            self._play_tone_sequence([880, 1109], duration=0.1)
        except (RuntimeError, OSError) as exc:
            logging.warning("Failed to play run detected sound: %s", exc)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled and PYGAME_AVAILABLE and self._initialized

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))

    def _can_play(self) -> bool:
        return self.enabled and self._initialized and PYGAME_AVAILABLE and pygame is not None

    def _play_tone_sequence(self, frequencies: List[float], duration: float = 0.2):
        if not self._can_play():
            return
        assert pygame is not None

        try:
            sample_rate = 22050
            samples_per_tone = int(sample_rate * duration)

            for freq in frequencies:
                t = np.linspace(0, duration, samples_per_tone, False)
                wave = np.sin(2 * np.pi * freq * t)
                envelope = np.exp(-t * 3)
                wave = wave * envelope * self.volume
                wave = (wave * 32767).astype(np.int16)

                stereo_wave = np.zeros((samples_per_tone, 2), dtype=np.int16)
                stereo_wave[:, 0] = wave
                stereo_wave[:, 1] = wave

                sound = pygame.sndarray.make_sound(stereo_wave)
                sound.play()
                pygame.time.wait(int(duration * 1000))
        except (RuntimeError, OSError, AttributeError) as exc:
            logging.warning("Failed to generate tone sequence: %s", exc)


class AudioAnalyzer:
    """Analyze video audio to detect completion beeps and calculate frame ranges."""

    def __init__(self):
        self.available = LIBROSA_AVAILABLE
        self.sample_rate = 44100
        self.hop_length = 512
        self.n_fft = 2048

    def extract_audio_from_video(self, video_path: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if not self.available or librosa is None:
            logging.warning("librosa not available - cannot extract audio")
            return None, None

        try:
            audio_data, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            logging.info("Extracted audio: %.2f seconds at %s Hz", len(audio_data) / sr, sr)
            return audio_data, sr
        except Exception as exc:
            logging.error("Failed to extract audio from video: %s", exc)
            return None, None

    def detect_beeps(
        self,
        audio_data: np.ndarray,
        sample_rate: float,
        target_frequency: float = 7000.0,
        frequency_tolerance: float = 50.0,
        threshold_percentile: float = 95.0,
        min_duration: float = 0.1,
    ) -> List[float]:
        if not self.available or librosa is None or audio_data is None:
            return []

        try:
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)

            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft)
            freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

            min_frequency = target_frequency - frequency_tolerance
            max_frequency = target_frequency + frequency_tolerance
            freq_mask = (freqs >= min_frequency) & (freqs <= max_frequency)

            if not np.any(freq_mask):
                logging.warning(
                    "No STFT bins found within %.1f±%.1fHz; increase tolerance to detect beeps",
                    target_frequency,
                    frequency_tolerance,
                )
                return []

            target_magnitude = magnitude[freq_mask, :]
            logging.info(
                "Targeting %.1fHz ±%.1fHz (%.1f-%.1fHz)",
                target_frequency,
                frequency_tolerance,
                min_frequency,
                max_frequency,
            )
            logging.info("Frequency resolution: %.1fHz, %s bins selected", freq_resolution, np.sum(freq_mask))

            energy_per_frame = np.max(target_magnitude, axis=0)
            threshold = np.percentile(energy_per_frame, threshold_percentile)

            mean_energy = np.mean(energy_per_frame)
            max_energy = np.max(energy_per_frame)
            logging.info("Energy stats: mean=%.2f, max=%.2f, threshold=%.2f", mean_energy, max_energy, threshold)

            above_threshold = energy_per_frame > threshold
            times = librosa.frames_to_time(
                np.arange(len(energy_per_frame)), sr=sample_rate, hop_length=self.hop_length
            )

            beep_times: List[float] = []
            in_beep = False
            beep_start = 0.0

            for time_idx, (time_value, is_above) in enumerate(zip(times, above_threshold)):
                if is_above and not in_beep:
                    in_beep = True
                    beep_start = time_value
                elif not is_above and in_beep:
                    in_beep = False
                    beep_duration = time_value - beep_start
                    if beep_duration >= min_duration:
                        beep_times.append(beep_start + beep_duration / 2)

            if in_beep:
                beep_duration = times[-1] - beep_start
                if beep_duration >= min_duration:
                    beep_times.append(beep_start + beep_duration / 2)

            logging.info("Detected %d beeps at: %s", len(beep_times), beep_times)
            return beep_times
        except Exception as exc:
            logging.error("Failed to detect beeps: %s", exc)
            return []

    def find_completion_beeps(
        self,
        video_path: str,
        expected_run_duration: float = 0.0,
    ) -> List[Tuple[float, int]]:
        if not self.available or librosa is None:
            return []

        audio_data, sample_rate = self.extract_audio_from_video(video_path)
        if audio_data is None:
            return []

        beep_times = self.detect_beeps(audio_data, sample_rate)
        if not beep_times:
            return []

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps <= 0:
                logging.error("Invalid FPS from video")
                return []

            results: List[Tuple[float, int]] = []
            for beep_time in beep_times:
                frame_number = int(beep_time * fps)
                if 0 <= frame_number < total_frames:
                    results.append((beep_time, frame_number))

            if expected_run_duration > 0.0:
                filtered_results = [
                    (beep_time, frame_number)
                    for beep_time, frame_number in results
                    if beep_time >= expected_run_duration
                ]
                if filtered_results:
                    logging.info(
                        "Filtered %d beeps to %d based on run duration",
                        len(results),
                        len(filtered_results),
                    )
                    results = filtered_results

            return results
        except Exception as exc:
            logging.error("Failed to process video for beep detection: %s", exc)
            return []
