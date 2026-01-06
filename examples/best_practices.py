"""
Example script demonstrating best practices for audio resampling and inference.

This script shows:
1. Using scipy.signal.resample_poly with padtype='line' for better boundary behavior
2. Proper ONNX Runtime threading configuration
3. Processing with overlap and crossfading for quality
4. Avoiding librosa dependency for resampling (though it can be used for loading)
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(file_path: str, target_sr: int = 16000):
    """
    Load audio file and resample if needed.
    
    Can use either soundfile (recommended) or librosa for loading.
    For resampling, uses scipy.signal.resample_poly which is:
    - BSD licensed (no LGPL dependencies)
    - Fast and high quality with proper configuration
    - Better for chunked audio pipelines when using padtype='line'
    """
    # Load audio with soundfile (or could use librosa.load)
    audio, sr = sf.read(file_path, dtype='float32')
    
    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = resample_audio_scipy(audio, sr, target_sr)
    
    return audio


def resample_audio_scipy(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    padtype: str = 'line'
) -> np.ndarray:
    """
    Resample audio using scipy.signal.resample_poly with proper configuration.
    
    Benefits over default librosa.resample (without SoXR):
    - 40-70× faster than librosa's Kaiser windowed sinc (default without SoXR)
    - BSD licensed (vs LGPL for SoXR)
    - Good quality polyphase FIR resampling
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate  
        padtype: Padding type for boundary handling
                'line' - Linear extrapolation (recommended for chunked audio)
                'mean' - Mean of edge values
                'constant' - Zero padding (default, but can cause edge artifacts)
    
    Returns:
        Resampled audio
    """
    # Calculate up/down factors (reduced to lowest terms)
    g = gcd(target_sr, orig_sr)
    up = target_sr // g
    down = orig_sr // g
    
    # Resample with proper boundary handling
    # padtype='line' reduces edge droop/click risk in chunked pipelines
    resampled = resample_poly(audio, up, down, padtype=padtype)
    
    return resampled


def example_basic_inference():
    """Example 1: Basic inference with single audio file."""
    print("Example 1: Basic ONNX inference")
    print("-" * 50)
    
    from FastAudioSR.streaming import StreamingFASRONNX
    
    # Initialize model with proper threading
    # n_cpu=1 is good for single-threaded inference
    model = StreamingFASRONNX(
        'model.onnx',
        onnx_execution_provider='CPUExecutionProvider',
        n_cpu=1  # Keep low if using ThreadPool externally
    )
    
    # Load and resample audio to 16kHz
    audio = load_audio('input.wav', target_sr=16000)
    
    # Run inference
    upsampled = model.run(audio)
    
    # Save output at 48kHz
    sf.write('output_basic.wav', upsampled, samplerate=48000)
    print(f"Processed {len(audio)/16000:.2f}s audio -> output_basic.wav\n")


def example_optimized_inference():
    """Example 2: Using the optimized inference module with overlap."""
    print("Example 2: Optimized inference with overlap and crossfading")
    print("-" * 50)
    
    from FastAudioSR.optimized_inference import OptimizedFASRONNX
    
    # Initialize with proper threading configuration
    model = OptimizedFASRONNX(
        'model.onnx',
        execution_provider='CPUExecutionProvider',
        intra_op_num_threads=1,  # Low when not using parallel chunking
        inter_op_num_threads=1,
        enable_parallel_chunking=False
    )
    
    # Load audio
    audio = load_audio('input.wav', target_sr=16000)
    
    # Process with overlap and crossfading for best quality
    upsampled = model.upsample(
        audio,
        use_overlap=True,
        chunk_size=80000,  # 5 seconds at 16kHz
        overlap=1600       # 100ms overlap
    )
    
    # Save output
    sf.write('output_optimized.wav', upsampled, samplerate=48000)
    print(f"Processed {len(audio)/16000:.2f}s audio -> output_optimized.wav")
    print("Used overlap and crossfading for smoother transitions\n")


def example_parallel_chunking():
    """
    Example 3: Parallel chunk processing with ThreadPool.
    
    Important: Only use when intra_op_num_threads is low to avoid oversubscription!
    Rule of thumb:
    - If ThreadPoolExecutor(max_workers=N): set intra_op_num_threads=1 (or small)
    - If you want ORT to parallelize internally: don't use ThreadPool
    """
    print("Example 3: Parallel chunk processing")
    print("-" * 50)
    
    from FastAudioSR.optimized_inference import OptimizedFASRONNX
    
    # Initialize with parallel chunking enabled
    # IMPORTANT: intra_op_num_threads=1 to avoid thread oversubscription
    model = OptimizedFASRONNX(
        'model.onnx',
        execution_provider='CPUExecutionProvider',
        intra_op_num_threads=1,  # MUST be low when using ThreadPool
        inter_op_num_threads=1,
        enable_parallel_chunking=True,
        max_workers=4  # Number of parallel worker threads
    )
    
    # Load audio
    audio = load_audio('input.wav', target_sr=16000)
    
    # Process in parallel
    upsampled = model.process_parallel(audio, chunk_size=80000)
    
    # Save output
    sf.write('output_parallel.wav', upsampled, samplerate=48000)
    print(f"Processed {len(audio)/16000:.2f}s audio with 4 parallel workers")
    print("Note: intra_op_num_threads=1 to avoid thread oversubscription\n")


def example_openvino():
    """
    Example 4: Using OpenVINO execution provider.
    
    OpenVINO EP has its own threading configuration (NUM_STREAMS, inference threads).
    Don't combine high EP-level parallelism with Python ThreadPool.
    """
    print("Example 4: OpenVINO execution provider")
    print("-" * 50)
    
    from FastAudioSR.streaming import StreamingFASRONNX
    
    try:
        # Initialize with OpenVINO provider
        # OpenVINO has its own parallelism controls
        model = StreamingFASRONNX(
            'model.onnx',
            onnx_execution_provider='OpenVINOExecutionProvider',
            n_cpu=1  # Keep ORT threads low, let OpenVINO handle threading
        )
        
        # Load and process audio
        audio = load_audio('input.wav', target_sr=16000)
        upsampled = model.run(audio)
        
        # Save output
        sf.write('output_openvino.wav', upsampled, samplerate=48000)
        print(f"Processed with OpenVINO EP -> output_openvino.wav")
        print("OpenVINO handles its own threading (NUM_STREAMS, etc.)\n")
        
    except Exception as e:
        print(f"OpenVINO not available: {e}")
        print("Install with: pip install onnxruntime-openvino\n")


def example_scipy_vs_librosa():
    """
    Example 5: Comparison of resampling methods.
    
    Shows the difference between scipy and librosa resampling.
    """
    print("Example 5: Resampling comparison")
    print("-" * 50)
    
    # Generate test signal
    duration = 1.0  # seconds
    orig_sr = 44100
    t = np.linspace(0, duration, int(duration * orig_sr))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone
    
    # Method 1: scipy.signal.resample_poly (recommended)
    import time
    start = time.time()
    scipy_result = resample_audio_scipy(audio, orig_sr, 16000, padtype='line')
    scipy_time = time.time() - start
    print(f"scipy.signal.resample_poly: {scipy_time*1000:.2f}ms")
    
    # Method 2: librosa (if available)
    try:
        import librosa
        start = time.time()
        # Without SoXR installed, this will be slow (Kaiser windowed sinc)
        # With SoXR, this will default to 'soxr_hq' and be fast
        librosa_result = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        librosa_time = time.time() - start
        print(f"librosa.resample: {librosa_time*1000:.2f}ms")
        print(f"Speedup: {librosa_time/scipy_time:.1f}×")
        
        # Check if SoXR is being used
        import importlib.util
        has_soxr = importlib.util.find_spec('soxr') is not None
        if has_soxr:
            print("✓ SoXR is installed - librosa will use fast 'soxr_hq' backend")
        else:
            print("✗ SoXR not installed - librosa using slower Kaiser backend")
            print("  Install with: pip install soxr")
    except ImportError:
        print("librosa not installed (optional)")
    
    print("\nRecommendation: Use scipy.signal.resample_poly with padtype='line'")
    print("- BSD licensed (no LGPL dependencies)")
    print("- Fast polyphase FIR resampling")
    print("- Better boundary handling for chunked audio\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Audio Super-Resolution Best Practices Examples")
    print("=" * 50)
    print()
    
    # Run resampling comparison (doesn't need model file)
    example_scipy_vs_librosa()
    
    # Uncomment to run inference examples (requires model.onnx and input.wav)
    # example_basic_inference()
    # example_optimized_inference()
    # example_parallel_chunking()
    # example_openvino()
    
    print("=" * 50)
    print("Done!")
    print("=" * 50)
