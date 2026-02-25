# ONNX Runtime on Qualcomm Hexagon – QCS6490

**Version:** 1.0

**Release Date:** Feb 2026

**Copyright:** © 2026 Advantech Corporation & NYCU COSMOS Lab. All rights reserved.


This document describes how to validate the Qualcomm NPU-enabled ONNX Runtime container on the QCS6490 platform.

## 1. Hardware Specifications

| Component       | Specification      |
|-----------------|--------------------|
| Target Hardware | [ADVANTECH AOM-2721](https://www.advantech.com/en-us/products/risc_evaluation_kit/aom-dk2721/mod_0e561ece-295c-4039-a545-68f8ded469a8) |
| SoC             | Qualcomm QCS6490   |
| GPU             | Adreno™ 643        |
| DSP             | Hexagon™ 770       |
| Memory          | 8GB LPDDR5         |


## 2. Software Components

| Component          | Version | Description                                                        |
| ------------------ | ------- | ------------------------------------------------------------------ |
| Python             | 3.10    | Runtime environment                                                |
| ONNX Runtime (QNN) | 1.24.1  | Custom build with QNN Execution Provider (Built with QAIRT 2.43.0) |
| QAIRT (QNN SDK)    | 2.43.0  | Qualcomm AI Runtime backend                                        |

**Note**: The custom build of `onnxruntime-qnn` currently only works within this container environment.

## 3. Run Container
Clone the project:
- On the PC
```
git clone https://github.com/Advantech-EdgeSync-Containers/onnxruntime-on-Qualcomm-Hexagon-QCS6490.git
scp -r ./onnxruntime-on-Qualcomm-Hexagon-QCS6490-main\ <username>@<aom2721-ip>:/home/<username>/
```
- On AOM-2721
```
chmod +x -R onnxruntime-on-Qualcomm-Hexagon-QCS6490-main
cd onnxruntime-on-Qualcomm-Hexagon-QCS6490-main
```

Start container:
```
./run-container.sh
```
This script launches the container and opens an interactive shell.

## 4. Exit container
Inside the container, type:
```
exit
```

Expected output:
```
Exited container. Cleaning up...
[+] Running 2/2
 ✔ Container qualcomm-onnxruntime-qnn-ready-container        Removed                                                                                 10.4s 
 ✔ Network qualcomm-onnxruntime-qnn-ready-container_default  Removed  
```

## 5. Test ONNX Runtime with NPU capability
Run the benchmark script:
```
cd nycu-benchmark
python nycu-cosmoslab-onnxruntime-benchmark.py
```

Benchmark Result (100 Iterations)

Model: [EfficientNet-B0](https://aihub.qualcomm.com/models/efficientnet_b0)

Quantiaztion: w8a16

**Model is download from Qualcomm AI-Hub
```
--- Initializing CPU Session ---
--- Initializing QNN Session (HTP/DSP) ---
/prj/qct/webtech_scratch20/mlg_user_admin/qaisw_source_repo/rel/qairt-2.43.0/release/SNPE_SRC/avante-tools/prebuilt/dsp/hexagon-sdk-5.5.5/ipc/fastrpc/rpcmem/src/rpcmem_android.c:38:dummy call to rpcmem_init, rpcmem APIs will be used from libxdsprpc
Starting stage: Graph Preparation Initializing
Completed stage: Graph Preparation Initializing (268 us)
Starting stage: Graph Optimizations
Completed stage: Graph Optimizations (603465 us)
Starting stage: Post Graph Optimization
Completed stage: Post Graph Optimization (18554 us)
Starting stage: Graph Sequencing for Target
Completed stage: Graph Sequencing for Target (100218 us)
Starting stage: VTCM Allocation
Completed stage: VTCM Allocation (25607 us)
Starting stage: Parallelization Optimization
Completed stage: Parallelization Optimization (7322 us)
Starting stage: Finalizing Graph Sequence

====== DDR bandwidth summary ======
spill_bytes=0
fill_bytes=0
write_total_bytes=65536
read_total_bytes=11130880

Completed stage: Finalizing Graph Sequence (9903 us)
Starting stage: Completion
Completed stage: Completion (551 us)

========================================
 PERFORMANCE COMPARISON (100 Iterations)
========================================
[CPU Only] Running 100 iterations...
[CPU Only] Total Time: 13687.09 ms
[CPU Only] Average Latency: 136.8709 ms
[QNN (NPU)] Running 100 iterations...
[QNN (NPU)] Total Time: 537.07 ms
[QNN (NPU)] Average Latency: 5.3707 ms

 Result: QNN is 25.48x faster than CPU (Average)

```
The result confirms that inference is successfully offloaded to the Hexagon 770  through the QNN Execution Provider, achieving approximately 25× acceleration compared to CPU execution.

## 6. Development Workflow

The container uses a bind mount configuration:
```
volumes:
  - ./:/workspace/
```
The host project directory (e.g., onnxruntime-on-Qualcomm-Hexagon-QCS6490-main) is directly synchronized with /workspace inside the container.

You can create or modify Python files directly in the host project folder, and they will be immediately available inside the container without rebuilding the image.

### Example

Create a new Python file on the host:
```
touch test.py
```

Edit `test.py` with the following content:

```
import onnxruntime as ort
print(ort.get_available_providers())
```

Expected output:
```
['QNNExecutionProvider', 'CPUExecutionProvider']
```

Run it inside the container:
```
python test.py
```

If `QNNExecutionProvider` appears in the output, it confirms that the QNN Execution Provider is successfully enabled and the container can access the Hexagon 770.


This workflow enables rapid development and testing while keeping the runtime environment isolated within the container.
