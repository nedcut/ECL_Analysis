"""Audio feedback manager for the application."""

import logging
from typing import List

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - audio features disabled")


class AudioManager:
    """Handle all audio feedback in the application."""

    def __init__(self, enabled: bool = True, volume: float = 0.7):
        """
        Initialize audio manager with pygame mixer.

        Args:
            enabled: Whether audio is enabled
            volume: Audio volume (0.0 to 1.0)
        """
        self.enabled = enabled and PYGAME_AVAILABLE
        self.volume = max(0.0, min(1.0, volume))
        self._initialized = False

        if self.enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._initialized = True
            except (ImportError, OSError, RuntimeError) as e:
                logging.warning(f"Failed to initialize audio: {e}")
                self.enabled = False

    def play_analysis_start(self) -> None:
        """Play sound when analysis starts."""
        if not self._can_play():
            return
        try:
            # Generate a simple ascending tone sequence
            self._play_tone_sequence([440, 554, 659], duration=0.15)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play analysis start sound: {e}")

    def play_analysis_complete(self) -> None:
        """Play sound when analysis completes."""
        if not self._can_play():
            return
        try:
            # Generate a completion chord
            self._play_tone_sequence([523, 659, 783], duration=0.3)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play analysis complete sound: {e}")

    def play_run_detected(self) -> None:
        """Play sound when a run is detected within expected duration."""
        if not self._can_play():
            return
        try:
            # Generate a quick notification beep
            self._play_tone_sequence([880, 1109], duration=0.1)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play run detected sound: {e}")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable audio.

        Args:
            enabled: Whether audio should be enabled
        """
        self.enabled = enabled and PYGAME_AVAILABLE and self._initialized

    def set_volume(self, volume: float) -> None:
        """
        Set audio volume (0.0 to 1.0).

        Args:
            volume: Audio volume level
        """
        self.volume = max(0.0, min(1.0, volume))

    def _can_play(self) -> bool:
        """
        Check if audio can be played.

        Returns:
            True if audio system is ready to play sounds
        """
        return self.enabled and self._initialized and PYGAME_AVAILABLE

    def _play_tone_sequence(self, frequencies: List[float], duration: float = 0.2) -> None:
        """
        Play a sequence of tones.

        Args:
            frequencies: List of frequencies in Hz to play sequentially
            duration: Duration of each tone in seconds
        """
        if not self._can_play():
            return

        try:
            sample_rate = 22050
            samples_per_tone = int(sample_rate * duration)

            for freq in frequencies:
                # Generate sine wave
                t = np.linspace(0, duration, samples_per_tone, False)
                wave = np.sin(2 * np.pi * freq * t)

                # Apply envelope to avoid clicks
                envelope = np.exp(-t * 3)  # Exponential decay
                wave = wave * envelope * self.volume

                # Convert to 16-bit integers
                wave = (wave * 32767).astype(np.int16)

                # Create stereo sound
                stereo_wave = np.zeros((samples_per_tone, 2), dtype=np.int16)
                stereo_wave[:, 0] = wave  # Left channel
                stereo_wave[:, 1] = wave  # Right channel

                # Play the sound
                sound = pygame.sndarray.make_sound(stereo_wave)
                sound.play()

                # Wait for tone to finish
                pygame.time.wait(int(duration * 1000))

        except (RuntimeError, OSError, AttributeError) as e:
            logging.warning(f"Failed to generate tone sequence: {e}")