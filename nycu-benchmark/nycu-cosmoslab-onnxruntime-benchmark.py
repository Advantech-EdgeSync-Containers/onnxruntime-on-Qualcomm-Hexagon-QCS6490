# Copyright (c) NYCU COSMOS Lab.
# ONNX Runtime Performance Benchmarking: CPU vs. QNN (HTP) - 100 Iterations

import argparse
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

def load_labels(filename):
    """Load labels from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path, height, width, input_dtype):
    """Resize and convert image to the exact type required by the model."""
    img = Image.open(image_path).convert('RGB').resize((width, height))
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    if "float" in input_dtype:
        return img.astype(np.float32) / 255.0
    elif "uint16" in input_dtype:
        return img.astype(np.uint16)
    else:
        return img.astype(np.uint8)

def benchmark_session(session, input_name, img, iterations=1000, label="Inference"):
    """
    Warm up once, then measure the average latency over N iterations.
    """
    # 1. Warm-up (Executed once, not timed)
    # This triggers graph optimization and hardware power-up (burst mode)
    session.run(None, {input_name: img})
    
    # 2. Benchmark Loop
    print(f"[{label}] Running {iterations} iterations...")
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        outputs = session.run(None, {input_name: img})
        
    end_time = time.perf_counter()
    
    # Calculate average
    total_duration_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_duration_ms / iterations
    
    print(f"[{label}] Total Time: {total_duration_ms:.2f} ms")
    print(f"[{label}] Average Latency: {avg_latency_ms:.4f} ms")
    
    return outputs, avg_latency_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="grace_hopper.bmp")
    parser.add_argument("-m", "--model_file", default="model.onnx")
    parser.add_argument("-l", "--label_file", default="labels.txt")
    parser.add_argument("-n", "--iterations", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.model_file):
        raise RuntimeError(f"Model file not found: {args.model_file}")

    # 1. Initialize CPU Session
    print("--- Initializing CPU Session ---")
    cpu_session = ort.InferenceSession(args.model_file, providers=['CPUExecutionProvider'])

    # 2. Initialize QNN (NPU) Session
    print("--- Initializing QNN Session (HTP/DSP) ---")
    qnn_options = {
        "backend_type": 'htp',
        "htp_performance_mode": "burst",
        "device_id": "0",
        "htp_graph_finalization_optimization_mode": "3",
        "soc_model": "35",
        "htp_arch": "68",
        "vtcm_mv": "2",
    }
    
    try:
        qnn_session = ort.InferenceSession(
            args.model_file, 
            providers=['QNNExecutionProvider'], 
            provider_options=[qnn_options]
        )
    except Exception as e:
        print(f"Error: QNN Session failed. {e}")
        qnn_session = None

    # Get Metadata & Preprocess
    input_info = cpu_session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    input_dtype = input_info.type
    
    img = preprocess_image(args.image, input_shape[2], input_shape[3], input_dtype)

    # --- Performance Comparison ---
    print("\n" + "="*40)
    print(f" PERFORMANCE COMPARISON ({args.iterations} Iterations)")
    print("="*40)
    
    # CPU Benchmark
    _, cpu_avg = benchmark_session(cpu_session, input_name, img, iterations=args.iterations, label="CPU Only")

    # QNN Benchmark
    if qnn_session:
        outputs, qnn_avg = benchmark_session(qnn_session, input_name, img, iterations=args.iterations, label="QNN (NPU)")
        
        speedup = cpu_avg / qnn_avg
        print(f"\n Result: QNN is {speedup:.2f}x faster than CPU (Average)")
    else:
        print("\n[Result] QNN benchmarking skipped.")

    # --- Verification (using last QNN output) ---
    if qnn_session:
        logits = outputs[0][0]
        probs = logits.astype(np.float32)
        probs /= np.sum(probs) # Simple normalization
        
        top_k_idx = probs.argsort()[-5:][::-1]
        labels = load_labels(args.label_file)

        print("\n===== Top-5 Prediction (QNN) =====")
        for idx in top_k_idx:
            print(f"{probs[idx]:.6f}: {labels[idx]}")