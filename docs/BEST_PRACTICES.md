# Audio Processing Best Practices

This document outlines best practices for using FlashSR efficiently and correctly, particularly regarding resampling, threading, and ONNX Runtime configuration.

## Table of Contents
- [Audio Resampling](#audio-resampling)
- [ONNX Runtime Threading](#onnx-runtime-threading)
- [Chunk Processing with Overlap](#chunk-processing-with-overlap)
- [Execution Providers](#execution-providers)

## Audio Resampling

### Recommendation: Use `scipy.signal.resample_poly`

For resampling audio (e.g., from 44.1kHz to 16kHz), we recommend using `scipy.signal.resample_poly` instead of `librosa.resample` when SoXR is not available.

#### Why scipy?

1. **Performance**: 40-70× faster than librosa's default Kaiser windowed sinc backend
2. **License**: BSD licensed (no LGPL dependencies like SoXR)
3. **Quality**: Polyphase FIR resampling is the standard approach for audio SRC
4. **Boundary handling**: Supports `padtype='line'` which reduces edge artifacts

#### Example Usage

```python
from scipy.signal import resample_poly
from math import gcd

def resample_audio(audio, orig_sr, target_sr, padtype='line'):
    """Resample audio with proper boundary handling."""
    g = gcd(target_sr, orig_sr)
    up = target_sr // g
    down = orig_sr // g
    
    # padtype='line' is recommended for chunked audio pipelines
    return resample_poly(audio, up, down, padtype=padtype)
```

#### About librosa.resample

Modern `librosa` (0.10+) defaults to `res_type='soxr_hq'` **if SoXR is installed**. If you want to use librosa:

```bash
# Install SoXR for fast resampling
pip install soxr
```

Without SoXR, librosa falls back to slower methods. The 40-70× speedups mentioned indicate SoXR was not available.

### Boundary Handling with `padtype`

When using `scipy.signal.resample_poly`, the `padtype` parameter controls how the signal is extended beyond boundaries:

- `'line'` (recommended): Linear extrapolation - good for chunked audio
- `'mean'`: Mean of edge values
- `'constant'` (default): Zero padding - can cause edge artifacts

For chunked audio processing, `padtype='line'` reduces click/droop risk at chunk boundaries.

## ONNX Runtime Threading

### Thread Safety

`InferenceSession.Run()` is **thread-safe** - you can call it from multiple threads without extra locking.

### Avoiding Thread Oversubscription

⚠️ **Important**: ONNX Runtime uses internal threading (`intra_op_num_threads`). If you also use Python ThreadPool for chunk parallelism, you can easily oversubscribe CPU cores and **lose performance**.

#### Rule of Thumb

Choose **ONE** parallelization strategy:

**Option 1: Python ThreadPool + Low ORT Threads**
```python
# Use ThreadPool for chunk parallelism
model = OptimizedFASRONNX(
    'model.onnx',
    intra_op_num_threads=1,  # Keep LOW to avoid oversubscription
    enable_parallel_chunking=True,
    max_workers=4
)
```

**Option 2: High ORT Threads + No ThreadPool**
```python
# Let ONNX Runtime parallelize internally
model = StreamingFASRONNX(
    'model.onnx',
    n_cpu=4  # ORT handles parallelism
)
# Process sequentially, no ThreadPool
```

### Configuration Examples

#### CPU Execution Provider
```python
from FastAudioSR.streaming import StreamingFASRONNX

model = StreamingFASRONNX(
    'model.onnx',
    onnx_execution_provider='CPUExecutionProvider',
    n_cpu=1  # Single-threaded for simplest case
)
```

#### OpenVINO Execution Provider
```python
# OpenVINO has its own threading (NUM_STREAMS, inference threads)
# Keep ORT threads low, let OpenVINO handle parallelism
model = StreamingFASRONNX(
    'model.onnx',
    onnx_execution_provider='OpenVINOExecutionProvider',
    n_cpu=1  # Let OpenVINO manage threading
)
```

## Chunk Processing with Overlap

Even with perfect resampling, splitting audio into independent chunks can create audible seams if the model needs context.

### Best Practice: Use Overlap + Crossfading

```python
from FastAudioSR.optimized_inference import OptimizedFASRONNX

model = OptimizedFASRONNX('model.onnx')

# Process with 100ms overlap and crossfading
upsampled = model.upsample(
    audio,
    use_overlap=True,
    chunk_size=80000,  # 5 seconds at 16kHz
    overlap=1600       # 100ms at 16kHz
)
```

This approach:
1. Adds small overlap (50-200ms) between chunks
2. Processes each chunk independently
3. Crossfades the overlap regions
4. Often improves perceived quality more than any resampler tweak

### Recommended Overlap Sizes

- **Minimum**: 50ms (800 samples at 16kHz)
- **Typical**: 100ms (1600 samples at 16kHz)
- **Maximum**: 200ms (3200 samples at 16kHz)

Larger overlaps provide better context but increase computation.

## Execution Providers

### CPU (Default)
```bash
pip install onnxruntime
```

Best for: General use, development, CPU-only systems

### OpenVINO (Intel Hardware)
```bash
pip install onnxruntime-openvino
```

Best for: Intel CPUs, GPUs, VPUs

**Note**: OpenVINO EP bundles OpenVINO runtime (e.g., 2025.1.0). It has its own threading configuration:
- `NUM_STREAMS`: Number of parallel inference streams
- Inference threads per stream
- Don't combine high OpenVINO parallelism with Python ThreadPool

### CUDA (NVIDIA GPUs)
```bash
pip install onnxruntime-gpu
```

Best for: NVIDIA GPUs with CUDA support

## Complete Example

See `examples/best_practices.py` for complete working examples demonstrating:

1. scipy.signal.resample_poly with padtype='line'
2. Proper ONNX Runtime threading configuration
3. Chunk processing with overlap and crossfading
4. Different execution providers
5. Performance comparison between resampling methods

## Summary of Recommendations

1. ✅ Use `scipy.signal.resample_poly(..., padtype='line')` for resampling
2. ✅ Set `intra_op_num_threads=1` when using ThreadPool for chunk parallelism
3. ✅ Use overlap (50-200ms) and crossfading for chunk processing
4. ✅ Choose ONE parallelization strategy (ThreadPool OR ORT internal threading)
5. ✅ For OpenVINO EP, let it handle threading, don't add ThreadPool on top
6. ✅ Make librosa optional (only needed for audio loading, not resampling)

## References

- [scipy.signal.resample_poly documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [ONNX Runtime threading documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [librosa.resample documentation](https://librosa.org/doc/main/generated/librosa.resample.html)
