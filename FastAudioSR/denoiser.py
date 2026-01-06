"""
Audio denoising module using WebRTC VAD for silence detection + spectral gating.

Uses WebRTC VAD to accurately detect non-speech segments, then uses those
segments to build a noise profile for spectral gating noise reduction.
"""

import numpy as np
from typing import List, Tuple, Optional
import noisereduce as nr

# WebRTC VAD requires 16-bit PCM at 8000, 16000, 32000, or 48000 Hz
# Frame sizes must be 10, 20, or 30 ms
try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False


class AudioDenoiser:
    """
    Noise reduction using WebRTC VAD for silence detection + spectral gating.
    
    Pipeline:
    1. Use WebRTC VAD to detect non-speech frames
    2. Extract noise profile from those non-speech frames
    3. Apply spectral gating to remove noise from the full audio
    """
    
    def __init__(
        self,
        vad_aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        prop_decrease: float = 0.8,
        sample_rate: int = 16000
    ):
        """
        Initialize the denoiser.
        
        Args:
            vad_aggressiveness: WebRTC VAD aggressiveness (0-3).
                               0 = least aggressive (more false positives for speech)
                               3 = most aggressive (more false positives for non-speech)
                               2 is a good balance for noise profiling
            frame_duration_ms: Frame duration for VAD (10, 20, or 30 ms)
            prop_decrease: How much to reduce noise (0.0 = no reduction, 1.0 = full)
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
        """
        if not _HAS_WEBRTCVAD:
            raise ImportError("webrtcvad required. Install with: pip install webrtcvad-wheels")
        
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"WebRTC VAD requires sample rate of 8000, 16000, 32000, or 48000. Got {sample_rate}")
        
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms. Got {frame_duration_ms}")
        
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.prop_decrease = prop_decrease
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    def detect_non_speech(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect non-speech segments using WebRTC VAD.
        
        Args:
            audio: Input audio array (1D, float32, range [-1, 1])
        
        Returns:
            List of (start_sample, end_sample) tuples for non-speech regions
        """
        # Convert to 16-bit PCM for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        
        non_speech_regions = []
        current_start = None
        
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            
            if not is_speech:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    non_speech_regions.append((current_start, i))
                    current_start = None
        
        # Handle non-speech at the end
        if current_start is not None:
            non_speech_regions.append((current_start, len(audio_int16)))
        
        return non_speech_regions
    
    def extract_noise_profile(
        self,
        audio: np.ndarray,
        non_speech_regions: Optional[List[Tuple[int, int]]] = None,
        max_noise_samples: int = 48000  # 3 seconds at 16kHz
    ) -> np.ndarray:
        """
        Extract noise profile from non-speech regions.
        
        Args:
            audio: Input audio array
            non_speech_regions: List of (start, end) tuples. Auto-detected if None.
            max_noise_samples: Maximum samples to use for noise profile
        
        Returns:
            Noise profile array
        """
        if non_speech_regions is None:
            non_speech_regions = self.detect_non_speech(audio)
        
        if not non_speech_regions:
            # No non-speech detected - use the quietest segment
            chunk_size = int(self.sample_rate * 0.5)
            min_rms = float('inf')
            quietest = audio[:chunk_size] if len(audio) > chunk_size else audio
            
            for i in range(0, len(audio) - chunk_size, chunk_size // 2):
                chunk = audio[i:i + chunk_size]
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < min_rms:
                    min_rms = rms
                    quietest = chunk
            
            return quietest
        
        # Concatenate noise samples from non-speech regions
        noise_samples = []
        total = 0
        
        for start, end in non_speech_regions:
            if total >= max_noise_samples:
                break
            segment = audio[start:end]
            noise_samples.append(segment)
            total += len(segment)
        
        return np.concatenate(noise_samples)[:max_noise_samples]
    
    def denoise(
        self,
        audio: np.ndarray,
        noise_profile: Optional[np.ndarray] = None,
        prop_decrease: Optional[float] = None
    ) -> np.ndarray:
        """
        Remove noise using spectral gating with noise profile from non-speech.
        
        Args:
            audio: Input audio (1D, float32)
            noise_profile: Noise sample. Auto-detected from non-speech if None.
            prop_decrease: Override for noise reduction strength (0.0-1.0)
        
        Returns:
            Denoised audio
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio.shape}")
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        prop = prop_decrease if prop_decrease is not None else self.prop_decrease
        
        # Auto-detect if not provided
        if noise_profile is None:
            non_speech = self.detect_non_speech(audio)
            noise_profile = self.extract_noise_profile(audio, non_speech)
            print(f"  Detected {len(non_speech)} non-speech regions ({sum(e-s for s,e in non_speech)/self.sample_rate:.2f}s)")
        
        # Apply spectral gating
        denoised = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            y_noise=noise_profile,
            prop_decrease=prop,
            n_jobs=1
        )
        
        return denoised.astype(np.float32)


def denoise_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    vad_aggressiveness: int = 2,
    prop_decrease: float = 0.8
) -> np.ndarray:
    """
    Convenience function for one-call denoising.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate (8000, 16000, 32000, or 48000)
        vad_aggressiveness: WebRTC VAD aggressiveness (0-3)
        prop_decrease: Noise reduction strength (0.0-1.0)
    
    Returns:
        Denoised audio
    """
    denoiser = AudioDenoiser(
        vad_aggressiveness=vad_aggressiveness,
        prop_decrease=prop_decrease,
        sample_rate=sample_rate
    )
    return denoiser.denoise(audio)
