import logging
from typing import List, Optional, Tuple

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - audio features disabled")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio analysis disabled")

class AudioManager:
    """Handle all audio feedback in the application."""
    
    def __init__(self, enabled: bool = True, volume: float = 0.7):
        """Initialize audio manager with pygame mixer."""
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
    
    def play_analysis_start(self):
        """Play sound when analysis starts."""
        if not self._can_play():
            return
        try:
            # Generate a simple ascending tone sequence
            self._play_tone_sequence([440, 554, 659], duration=0.15)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play analysis start sound: {e}")
    
    def play_analysis_complete(self):
        """Play sound when analysis completes."""
        if not self._can_play():
            return
        try:
            # Generate a completion chord
            self._play_tone_sequence([523, 659, 783], duration=0.3)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play analysis complete sound: {e}")
    
    def play_run_detected(self):
        """Play sound when a run is detected within expected duration."""
        if not self._can_play():
            return
        try:
            # Generate a quick notification beep
            self._play_tone_sequence([880, 1109], duration=0.1)
        except (RuntimeError, OSError) as e:
            logging.warning(f"Failed to play run detected sound: {e}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable audio."""
        self.enabled = enabled and PYGAME_AVAILABLE and self._initialized
    
    def set_volume(self, volume: float):
        """Set audio volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
    
    def _can_play(self) -> bool:
        """Check if audio can be played."""
        return self.enabled and self._initialized and PYGAME_AVAILABLE
    
    def _play_tone_sequence(self, frequencies: List[float], duration: float = 0.2):
        """Play a sequence of tones."""
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

class AudioAnalyzer:
    """Analyze video audio to detect completion beeps and calculate frame ranges."""
    
    def __init__(self):
        """Initialize audio analyzer."""
        self.available = LIBROSA_AVAILABLE
        self.sample_rate = 44100
        self.hop_length = 512
        self.n_fft = 2048  # Larger FFT window for better frequency resolution
        
    def extract_audio_from_video(self, video_path: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if failed
        """
        if not self.available:
            logging.warning("librosa not available - cannot extract audio")
            return None, None
        
        try:
            # Use librosa to load audio from video
            audio_data, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            logging.info(f"Extracted audio: {len(audio_data)/sr:.2f} seconds at {sr} Hz")
            return audio_data, sr
        except Exception as e:
            logging.error(f"Failed to extract audio from video: {e}")
            return None, None
    
    def detect_beeps(self, audio_data: np.ndarray, sample_rate: float, 
                     target_frequency: float = 7000.0, frequency_tolerance: float = 50.0,
                     threshold_percentile: float = 95.0, min_duration: float = 0.1) -> List[float]:
        """
        Detect beeps/tones in audio data targeting a specific frequency.
        
        Args:
            audio_data: Audio waveform data
            sample_rate: Sample rate of audio
            target_frequency: Target frequency to detect (Hz) - default 7000Hz
            frequency_tolerance: Tolerance around target frequency (Hz) - default ±500Hz
            threshold_percentile: Percentile threshold for detection
            min_duration: Minimum duration of beep (seconds)
            
        Returns:
            List of timestamps (in seconds) where beeps were detected
        """
        if not self.available or audio_data is None:
            return []
        
        try:
            # Compute STFT to get frequency domain representation
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft)
            freq_resolution = freqs[1] - freqs[0]
            
            # Find frequency bins around our target frequency
            min_frequency = target_frequency - frequency_tolerance
            max_frequency = target_frequency + frequency_tolerance
            freq_mask = (freqs >= min_frequency) & (freqs <= max_frequency)
            target_magnitude = magnitude[freq_mask, :]
            
            logging.info(f"Targeting {target_frequency}Hz ±{frequency_tolerance}Hz ({min_frequency:.1f}-{max_frequency:.1f}Hz)")
            logging.info(f"Frequency resolution: {freq_resolution:.1f}Hz, {np.sum(freq_mask)} bins selected")
            
            # Use maximum energy instead of sum to focus on the strongest signal in our frequency range
            # This helps when the beep is very pure tone at 7000Hz
            energy_per_frame = np.max(target_magnitude, axis=0)
            
            # Calculate threshold based on percentile
            threshold = np.percentile(energy_per_frame, threshold_percentile)
            
            # Also calculate some statistics for better insight
            mean_energy = np.mean(energy_per_frame)
            max_energy = np.max(energy_per_frame)
            
            logging.info(f"Energy stats: mean={mean_energy:.2f}, max={max_energy:.2f}, threshold={threshold:.2f}")
            
            # Find frames above threshold
            above_threshold = energy_per_frame > threshold
            
            # Convert frame indices to time stamps
            times = librosa.frames_to_time(np.arange(len(energy_per_frame)), 
                                         sr=sample_rate, hop_length=self.hop_length)
            
            # Find start and end of continuous regions above threshold
            beep_times = []
            in_beep = False
            beep_start = 0.0
            
            for i, (time, is_above) in enumerate(zip(times, above_threshold)):
                if is_above and not in_beep:
                    # Start of beep
                    in_beep = True
                    beep_start = time
                elif not is_above and in_beep:
                    # End of beep
                    in_beep = False
                    beep_duration = time - beep_start
                    if beep_duration >= min_duration:
                        # Use middle of beep as detection time
                        beep_times.append(beep_start + beep_duration / 2)
            
            # Handle case where beep continues to end of audio
            if in_beep:
                beep_duration = times[-1] - beep_start
                if beep_duration >= min_duration:
                    beep_times.append(beep_start + beep_duration / 2)
            
            logging.info(f"Detected {len(beep_times)} beeps at: {beep_times}")
            return beep_times
            
        except Exception as e:
            logging.error(f"Failed to detect beeps: {e}")
            return []
    
    def find_completion_beeps(self, video_path: str, expected_run_duration: float = 0.0) -> List[Tuple[float, int]]:
        """
        Find completion beeps in video audio and calculate corresponding frame ranges.
        
        Args:
            video_path: Path to video file
            expected_run_duration: Expected run duration in seconds (0 = no filtering)
            
        Returns:
            List of (beep_time_seconds, frame_number) tuples
        """
        if not self.available:
            return []
        
        # Extract audio from video
        audio_data, sample_rate = self.extract_audio_from_video(video_path)
        if audio_data is None:
            return []
        
        # Detect beeps
        beep_times = self.detect_beeps(audio_data, sample_rate)
        if not beep_times:
            return []
        
        # Get video properties to convert times to frames
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0:
                logging.error("Invalid FPS from video")
                return []
            
            # Convert beep times to frame numbers
            results = []
            for beep_time in beep_times:
                frame_number = int(beep_time * fps)
                if 0 <= frame_number < total_frames:
                    results.append((beep_time, frame_number))
            
            # Filter by expected run duration if provided
            if expected_run_duration > 0.0:
                filtered_results = []
                for beep_time, frame_number in results:
                    # Check if there's enough time before this beep for a run of expected duration
                    if beep_time >= expected_run_duration:
                        filtered_results.append((beep_time, frame_number))
                
                if filtered_results:
                    logging.info(f"Filtered {len(results)} beeps to {len(filtered_results)} based on run duration")
                    results = filtered_results
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to process video for beep detection: {e}")
            return []

