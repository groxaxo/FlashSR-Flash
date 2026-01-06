"""
Optimized ONNX inference module with best practices for audio super-resolution.

This module implements:
1. scipy.signal.resample_poly with padtype='line' for better boundary behavior
2. Proper ONNX Runtime threading configuration to avoid oversubscription
3. Optional ThreadPool-based chunking with overlap for longer audio files
4. No dependency on librosa for core functionality
"""

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple


class OptimizedFASRONNX:
    """
    Optimized ONNX-based Fast Audio Super-Resolution with best practices.
    
    This implementation uses scipy for resampling and proper ONNX Runtime
    threading configuration to avoid performance issues.
    
    The model upsamples from 16kHz to 48kHz (3x upsampling factor).
    """
    
    # Audio configuration constants
    INPUT_SAMPLE_RATE = 16000   # Input sample rate (Hz)
    OUTPUT_SAMPLE_RATE = 48000  # Output sample rate (Hz)
    UPSAMPLING_FACTOR = OUTPUT_SAMPLE_RATE // INPUT_SAMPLE_RATE  # 3x
    
    def __init__(
        self,
        onnx_model_path: str,
        execution_provider: str = 'CPUExecutionProvider',
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
        enable_parallel_chunking: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the optimized ONNX inference session.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            execution_provider: ONNX execution provider. Supported:
                                 - 'CPUExecutionProvider' (default, recommended)
                                 - 'OpenVINOExecutionProvider' (Intel hardware acceleration)
            intra_op_num_threads: Number of threads for intra-op parallelism (default: 1)
                                 Set to 1 when using ThreadPool chunking to avoid oversubscription
            inter_op_num_threads: Number of threads for inter-op parallelism (default: 1)
            enable_parallel_chunking: Whether to use ThreadPool for parallel chunk processing
            max_workers: Number of worker threads for parallel chunking (if enabled)
        """
        # Configure ONNX Runtime session options
        session_options = ort.SessionOptions()
        
        # Important: When using ThreadPool chunking, set intra_op_num_threads low
        # to avoid thread oversubscription (multiple chunks in parallel + ORT internal threading)
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = inter_op_num_threads
        
        # Initialize ONNX Runtime session
        providers = [execution_provider]
        self.session = ort.InferenceSession(
            onnx_model_path,
            session_options,
            providers=providers
        )
        
        # Chunking configuration
        self.enable_parallel_chunking = enable_parallel_chunking
        self.max_workers = max_workers or 1
        
    def resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
        padtype: str = 'line'
    ) -> np.ndarray:
        """
        Resample audio using scipy.signal.resample_poly with proper boundary handling.
        
        This uses scipy's polyphase FIR resampler which is:
        - BSD licensed (no LGPL dependencies)
        - Fast and high quality
        - Good for audio sample rate conversion
        
        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            padtype: Padding type for boundary handling ('line', 'mean', etc.)
                    'line' is recommended for chunked audio pipelines to reduce edge artifacts
        
        Returns:
            Resampled audio array
        """
        from math import gcd
        
        # Calculate up/down sampling factors
        g = gcd(target_sr, orig_sr)
        up = target_sr // g
        down = orig_sr // g
        
        # Use scipy.signal.resample_poly with proper boundary handling
        # padtype='line' reduces edge droop/click risk for chunked audio
        resampled = resample_poly(
            audio,
            up,
            down,
            padtype=padtype
        )
        
        return resampled
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Process a single audio chunk through the ONNX model.
        
        Args:
            audio_chunk: Input audio chunk (1D array at 16kHz)
            normalize: Whether to normalize output to [-0.999, 0.999]
        
        Returns:
            Upsampled audio chunk at 48kHz
        """
        # Prepare input for ONNX model (add batch and channel dimensions)
        input_data = audio_chunk[np.newaxis, np.newaxis, :].astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {'x': input_data})
        upsampled = outputs[0].squeeze()
        
        # Normalize if requested
        if normalize:
            max_val = np.abs(upsampled).max()
            if max_val > 0:
                upsampled = upsampled / max_val * 0.999
        
        return upsampled
    
    def process_with_overlap(
        self,
        audio: np.ndarray,
        chunk_size: int = 80000,  # 5 seconds at 16kHz
        overlap: int = 1600,      # 100ms at 16kHz
        crossfade: bool = True
    ) -> np.ndarray:
        """
        Process audio in chunks with overlap to avoid seam artifacts.
        
        Even with perfect resampling, splitting audio into independent chunks
        can create audible seams if the model needs context. This method:
        - Adds overlap between chunks
        - Optionally crossfades the overlap regions
        - Improves perceived quality
        
        Args:
            audio: Input audio at 16kHz (1D array)
            chunk_size: Size of each chunk in samples
            overlap: Overlap size in samples (recommended 50-200ms)
            crossfade: Whether to crossfade overlap regions
        
        Returns:
            Upsampled audio at 48kHz
        """
        if len(audio) <= chunk_size:
            # Audio is short enough to process in one chunk
            return self.process_chunk(audio)
        
        # Process chunks with overlap
        chunks = []
        hop_size = chunk_size - overlap
        
        for i in range(0, len(audio), hop_size):
            # Extract chunk - each chunk overlaps with the previous one
            start = i
            end = min(len(audio), i + chunk_size)
            chunk = audio[start:end]
            
            # Process chunk
            upsampled_chunk = self.process_chunk(chunk)
            chunks.append((i, upsampled_chunk))
            
            if end >= len(audio):
                break
        
        # Combine chunks
        if not crossfade or len(chunks) == 1:
            # Simple concatenation (trim overlap)
            result = []
            for i, (pos, chunk) in enumerate(chunks):
                if i == 0:
                    # First chunk: keep everything
                    result.append(chunk)
                else:
                    # Subsequent chunks: trim the overlap region at the start
                    # Calculate overlap in output space
                    overlap_out = overlap * self.UPSAMPLING_FACTOR
                    result.append(chunk[overlap_out:])
            return np.concatenate(result)
        else:
            # Crossfade overlap regions
            return self._crossfade_chunks(chunks, overlap * self.UPSAMPLING_FACTOR)
    
    def _crossfade_chunks(
        self,
        chunks: list,
        overlap: int
    ) -> np.ndarray:
        """
        Crossfade overlapping chunks for smooth transitions.
        
        Args:
            chunks: List of (start_position, audio_chunk) tuples
            overlap: Overlap size in output samples (at 48kHz)
        
        Returns:
            Combined audio with crossfaded transitions
        """
        if len(chunks) == 0:
            return np.array([])
        
        if len(chunks) == 1:
            return chunks[0][1]
        
        result = chunks[0][1]
        
        for i in range(1, len(chunks)):
            current = chunks[i][1]
            
            # Create crossfade ramp
            fade_samples = min(overlap, len(result), len(current))
            ramp = np.linspace(0, 1, fade_samples)
            
            # Extract overlap regions
            prev_overlap = result[-fade_samples:]
            curr_overlap = current[:fade_samples]
            
            # Apply crossfade
            faded = prev_overlap * (1 - ramp) + curr_overlap * ramp
            
            # Combine
            result = np.concatenate([
                result[:-fade_samples],
                faded,
                current[fade_samples:]
            ])
        
        return result
    
    def process_parallel(
        self,
        audio: np.ndarray,
        chunk_size: int = 80000
    ) -> np.ndarray:
        """
        Process audio in parallel chunks using ThreadPool.
        
        Note: Only use this when intra_op_num_threads is set to 1 or a small value
        to avoid thread oversubscription. The ONNX Runtime will parallelize
        internally, and adding ThreadPool on top can hurt performance if not
        configured properly.
        
        Args:
            audio: Input audio at 16kHz
            chunk_size: Size of each chunk in samples
        
        Returns:
            Upsampled audio at 48kHz
        """
        if not self.enable_parallel_chunking:
            raise ValueError(
                "Parallel chunking is not enabled. "
                "Set enable_parallel_chunking=True when initializing."
            )
        
        # Split audio into chunks
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        
        # Concatenate results
        return np.concatenate(results)
    
    def upsample(
        self,
        audio: np.ndarray,
        use_overlap: bool = True,
        chunk_size: int = 80000,
        overlap: int = 1600
    ) -> np.ndarray:
        """
        Main method to upsample audio from 16kHz to 48kHz.
        
        Args:
            audio: Input audio at 16kHz (1D array, float32, range [-1, 1])
            use_overlap: Whether to use overlapping chunks (recommended)
            chunk_size: Size of chunks in samples (default: 5s at 16kHz)
            overlap: Overlap between chunks in samples (default: 100ms at 16kHz)
        
        Returns:
            Upsampled audio at 48kHz
        """
        # Validate input
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio.shape}")
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Process based on configuration
        if use_overlap:
            return self.process_with_overlap(
                audio,
                chunk_size=chunk_size,
                overlap=overlap
            )
        elif self.enable_parallel_chunking:
            return self.process_parallel(audio, chunk_size=chunk_size)
        else:
            return self.process_chunk(audio)
