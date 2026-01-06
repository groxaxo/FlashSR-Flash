#!/usr/bin/env python3
"""
Denoise and upscale audio using FlashSR.

Usage:
    conda activate flashsr-cpu
    python examples/denoise_and_upscale.py input.mp3 --output output_clean.wav
"""

import argparse
import os
import sys
import time
from math import gcd

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FastAudioSR.denoiser import AudioDenoiser  # Direct import, avoids __init__.py torch dependency


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to target sample rate."""
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g)
    return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Denoise and upscale audio using FlashSR')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-o', '--output', default='output_clean.wav')
    parser.add_argument('-m', '--model', default=None, help='Path to ONNX model')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU (CUDA) for inference')
    parser.add_argument('-d', '--denoise-strength', type=float, default=0.8)
    parser.add_argument('-a', '--vad-aggressiveness', type=int, default=2,
                        help='WebRTC VAD aggressiveness 0-3 (default: 2)')
    parser.add_argument('-n', '--normalize', type=float, default=0.95)
    parser.add_argument('--skip-denoise', action='store_true')
    args = parser.parse_args()
    
    # Provider selection
    providers = ['CPUExecutionProvider']
    if args.gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Requesting GPU execution...")
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'models', 'model_lite.onnx')
        if not os.path.exists(model_path):
            model_path = os.path.join(base, 'models', 'model.onnx')
    
    if not os.path.exists(model_path):
        sys.exit(f"Error: Model not found at {model_path}")
    
    print(f"Input: {args.input}\nModel: {model_path}\nOutput: {args.output}\n")
    
    # Load
    t_start_total = time.time()
    audio = load_audio(args.input)
    duration = len(audio) / 16000
    print(f"Loaded {duration:.2f}s audio")
    
    # Denoise using WebRTC VAD
    if not args.skip_denoise:
        denoiser = AudioDenoiser(
            vad_aggressiveness=args.vad_aggressiveness,
            prop_decrease=args.denoise_strength,
            sample_rate=16000
        )
        t0 = time.time()
        audio = denoiser.denoise(audio)
        print(f"Denoised in {time.time()-t0:.2f}s")
    
    # Upscale
    try:
        opts = ort.SessionOptions()
        # Optimize threading for CPU if not using GPU
        if not args.gpu:
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        
        # Check actual provider used
        active_providers = session.get_providers()
        print(f"Using providers: {active_providers}")
        if args.gpu and 'CUDAExecutionProvider' not in active_providers:
            print("WARNING: GPU requested but CUDAExecutionProvider not available. Falling back to CPU.")
        
        t0 = time.time()
        output = session.run(None, {'x': audio[np.newaxis, np.newaxis, :]})[0].squeeze()
        print(f"Upscaled in {time.time()-t0:.2f}s ({duration/(time.time()-t0):.1f}x RT)")
        
    except Exception as e:
        sys.exit(f"Error during inference: {e}")
    
    # Normalize
    if args.normalize > 0:
        mx = np.abs(output).max()
        if mx > 0:
            output = output * (args.normalize / mx)
    
    # Save
    sf.write(args.output, output, samplerate=48000)
    print(f"Saved {args.output}")
    print(f"Total time: {time.time() - t_start_total:.2f}s")


if __name__ == '__main__':
    main()
