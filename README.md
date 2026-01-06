# FlashSR

This is a tiny audio super-resolution model based on [hierspeech++](https://github.com/sh-lee-prml/HierSpeechpp) that upscales 16khz audio into much clearer 48khz audio efficiently!

FlashSR is released under an apache-2.0 license.

Model link: https://huggingface.co/YatharthS/FlashSR

## Features
 
- **Ultra-Fast Upscaling**: 3x super-resolution (16kHz -> 48kHz).
- **Smart Noise Reduction**: Integrated [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) detects silence to build accurate noise profiles, coupled with spectral gating for clean output.
- **GPU Acceleration**: Optional CUDA support for even faster processing using `onnxruntime-gpu`.
- **Edge Optimized**: Lightweight ONNX model (~500KB) suitable for deployment on low-power devices.
- **Streaming Support**: Low-latency streaming capabilities for real-time applications.
 
## Performance & Best Practices
 
For optimal performance and quality, see [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md).
 
- **GPU Inference**: Use `denoise_and_upscale.py --gpu` for CUDA acceleration (requires `onnxruntime-gpu`).
- **Noise Reduction**: Use the built-in denoising to clean up audio before upscaling.
- **CPU Optimization**: The default ONNX model runs efficiently on CPU.

## Usage
 
### 1. Installation
 
```bash
pip install -r requirements.txt
```
 
For GPU support:
```bash
pip install onnxruntime-gpu
```
 
### 2. Denoise & Upscale (Recommended)
 
We provide a complete script that automatically removes background noise (using WebRTC VAD) and upscales the audio.
 
```bash
# Basic usage (CPU)
python examples/denoise_and_upscale.py input.mp3 -o output.wav
 
# With GPU acceleration
python examples/denoise_and_upscale.py input.mp3 -o output.wav --gpu
 
# Adjust settings
python examples/denoise_and_upscale.py input.mp3 \
    --denoise-strength 0.9 \
    --normalize 0.95 \
    --vad-aggressiveness 3
```
 
### 3. Python API
 
To use the upsampler in your own code:
 
```python
import onnxruntime as ort
import numpy as np
import soundfile as sf
 
# 1. Load model
session = ort.InferenceSession('models/model_lite.onnx', providers=['CPUExecutionProvider'])
 
# 2. Load and prepare audio (must be 16kHz)
audio, sr = sf.read('input.wav')
# ... ensure audio is 16kHz and mono ...
input_tensor = audio[np.newaxis, np.newaxis, :].astype(np.float32)
 
# 3. Run inference
output = session.run(None, {'x': input_tensor})[0].squeeze()
 
# 4. Save output (48kHz)
sf.write('output_upscaled.wav', output, 48000)
```

# Streaming Input

The onnx model can be used in streaming mode for even lower latency. With a reasonable modern desktop/laptop CPU,
the upsampling can usually be done in real-time on a single core.

```python
from FastAudioSR.streaming import StreamingFASRONNX
import numpy as np
import soundfile as sf

# Initialize with downloaded onnx model
model = StreamingFASRONNX('model.onnx')

# Set input chunk size, which defines latency (4000 samples, 250 ms of 16khz audio in this case)
chunk_size = 4000
upsampled_output = []

# Make generater to consume the upsampled chunks as they are ready
gen = model.get_output(n_samples=chunk_size*3)  # 12000 samples at 48 khz, still 250 ms

# Simulate streaming in 16khz audio in 250 ms chunks
for i in range(0, len(dat), chunk_size):
    audio_chunk = dat[i:i+chunk_size]
    model.process_input(audio_chunk)
    upsampled_output.append(next(gen))

# Combine and save chunks, simulating real-time playback of upsampled chunks
sf.write('output.wav', np.concatenate(upsampled_output), samplerate=48000)
```

## Final notes
Thanks very much to the authors of hierspeech++. Thanks for checking out this repository as well.

Stars would be well appreciated, thank you.

Email: yatharthsharma3501@gmail.com
