import os
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import onnxruntime as ort

def load_audio(audio_path, target_sr=16000):
    """Load and resample audio using soundfile and scipy."""
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # Mono
    
    if sr != target_sr:
        from math import gcd
        g = gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down)
    
    return data.astype(np.float32)

def benchmark_onnx(model_path, audio_path):
    # Load audio
    y = load_audio(audio_path, target_sr=16000)
    duration = len(y) / 16000
    lowres_wav = y[np.newaxis, np.newaxis, :] # Add batch and channel dims
    
    # Configure session for CPU
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1 # Avoid oversubscription
    opts.inter_op_num_threads = 1
    
    ort_session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    start_time = time.time()
    # The optimized model uses 'x' as input name
    onnx_output = ort_session.run(None, {"x": lowres_wav})[0]
    end_time = time.time()
    
    proc_time = end_time - start_time
    rtf = duration / proc_time
    return duration, proc_time, rtf, onnx_output.squeeze()

def main():
    music_dir = "/home/op/Music"
    test_files = ["Brenda30secs.wav", "brenda.mp3", "colombiana.mp3"]
    
    onnx_model_path = "models/model_lite.onnx"
    if not os.path.exists(onnx_model_path):
        onnx_model_path = "models/model.onnx"
        
    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX model not found at {onnx_model_path}")
        return

    print(f"Using ONNX model: {onnx_model_path}")
    print(f"{'File':<25} | {'Type':<6} | {'Duration (s)':<12} | {'Proc Time (s)':<12} | {'RTF':<8}")
    print("-" * 75)
    
    for filename in test_files:
        audio_path = os.path.join(music_dir, filename)
        if not os.path.exists(audio_path):
            continue
            
        # ONNX Benchmark
        try:
            dur, p_time, rtf, out = benchmark_onnx(onnx_model_path, audio_path)
            print(f"{filename[:24]:<25} | {'ONNX':<6} | {dur:<12.2f} | {p_time:<12.4f} | {rtf:<8.2f}")
            
            # Save a sample output
            out_name = f"output_{filename.split('.')[0]}.wav"
            sf.write(out_name, out, samplerate=48000)
        except Exception as e:
            print(f"Error benchmark ONNX {filename}: {e}")

if __name__ == "__main__":
    main()
