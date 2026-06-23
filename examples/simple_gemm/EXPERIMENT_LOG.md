# Experiment Log: FP8 GEMM Scaling Analysis

**Date**: 2026-06-13  
**GPU**: NVIDIA GB10 (Blackwell Architecture, SM100)  
**CUDA Version**: 13.0  
**CUTLASS Version**: 3.5.1  

## Goal
Evaluate how matrix size impacts the performance (TOPS and GFLOPS) of CUTLASS FP32 and FP8 General Matrix Multiplication (GEMM) kernels to determine if larger sizes allow the GPU to reach higher peak compute utilization.

---

## Experimental Results

The benchmarks were run on square matrices of dimension $N \times N \times N$.
- Host-to-device memory allocation and copies are excluded from the "Pure Kernel" times.
- Naive CPU GEMM was run at $1024 \times 1024 \times 1024$ as a baseline (Execution time: **1.17s**, throughput: **1.84 GFLOPS**).

### Performance Scaling Table

| Matrix Size ($N$) | FP32 E2E (TFLOPS) | FP32 Pure Kernel (TFLOPS) | FP8 E2E (TOPS) | FP8 Pure Kernel (TOPS) |
| :---: | :---: | :---: | :---: | :---: |
| **1024** | 1.68 | 12.33 | 0.26 | 28.06 |
| **2048** | 3.25 | 12.43 | 0.52 | 25.81 |
| **4096** | 2.28 | 7.14 | 0.81 | **31.55** |
| **8192** | 3.06 | 5.46 | 1.61 | **34.01** |

---

## Key Conclusions

1. **Sustained Workload Scales FP8 TOPS**:
   - The user's hypothesis is **confirmed**. Increasing the matrix size from $1024^3$ to $8192^3$ increased the pure FP8 kernel performance from **28.06 TOPS** to **34.01 TOPS** (a **21.2% improvement**).
   - At smaller sizes (like 1024 and 2048), the kernel execution time is extremely short ($\approx 77\mu s$), which means overheads (such as GPU grid launch latencies, warp scheduling ramps, and tail-effect drop-offs) drag down the average throughput. At $8192^3$, the execution time is longer ($\approx 32.3ms$), allowing the GPU to run at a steady-state peak frequency and keep the Tensor Cores fully saturated.

2. **Divergent Scaling: SIMT FP32 vs. Tensor Core FP8**:
   - **FP8 (Tensor Cores)**: Scales up as size increases because it uses dense Tensor Core instructions (`OpClassTensorOp`), which are highly compute-bound. The 4x reduction in memory footprint compared to FP32 mitigates memory bandwidth bottlenecks.
   - **FP32 (SIMT)**: Performance **degrades** from **12.43 TFLOPS** at 2048 to **5.46 TFLOPS** at 8192. This is because the baseline FP32 kernel is configured to use SIMT CUDA cores (`OpClassSimt`), which are not hardware-accelerated for matrix multiplications. For larger sizes, the memory accesses exceed the GPU's L2 cache capacity, causing performance to become severely memory-bandwidth limited.

3. **E2E Overhead vs. Pure Kernel Performance**:
   - For both FP32 and FP8, there is a large gap between End-to-End (E2E) performance and Pure Kernel performance.
   - For FP8 at $8192^3$, E2E is **1.61 TOPS** compared to **34.01 TOPS** for the pure kernel. This is because E2E measurements include host-to-device conversions, transpositions (B matrix is transposed from RowMajor to ColumnMajor on the CPU), memory allocations (`cudaMalloc`), and device transfers (`cudaMemcpy`).
   - In production environments, this memory copy/allocation overhead is typically hidden by reusing pre-allocated device buffers and pipelining memory copies with execution streams.
