#include "gemm.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

// Define the GEMM configuration using RowMajor layout
using RowMajor = cutlass::layout::RowMajor;

// Instantiation of CUTLASS GEMM
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, RowMajor,               // ElementA, LayoutA
    float, RowMajor,               // ElementB, LayoutB
    float, RowMajor,               // ElementC, LayoutC
    float,                         // ElementAccumulator
    cutlass::arch::OpClassSimt,    // Operator class (SIMT for widest compatibility)
    cutlass::arch::Sm80            // Target architecture (SM80 works on Ampere, Hopper, and Blackwell)
>;

std::vector<float> gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K) {
    
    if (A.size() != static_cast<size_t>(M * K)) {
        throw std::invalid_argument("Size of A must be M * K");
    }
    if (B.size() != static_cast<size_t>(K * N)) {
        throw std::invalid_argument("Size of B must be K * N");
    }
    
    std::vector<float> C(M * N, 0.0f);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    return C;
}

std::vector<float> gemm_cutlass(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K) {
    
    if (A.size() != static_cast<size_t>(M * K)) {
        throw std::invalid_argument("Size of A must be M * K");
    }
    if (B.size() != static_cast<size_t>(K * N)) {
        throw std::invalid_argument("Size of B must be K * N");
    }
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    cudaError_t err;
    err = cudaMalloc(&d_A, A.size() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for A: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_B, B.size() * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error("Failed to allocate device memory for B: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_C, M * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("Failed to allocate device memory for C: " + std::string(cudaGetErrorString(err)));
    }
    
    // Copy data to device
    err = cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy A to device: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy B to device: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemset(d_C, 0, M * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to memset C on device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Run CUTLASS GEMM
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},           // Problem size
        {d_A, K},            // TensorRef A
        {d_B, N},            // TensorRef B
        {d_C, N},            // TensorRef C
        {d_C, N},            // TensorRef D
        {1.0f, 0.0f}         // alpha = 1.0, beta = 0.0
    );
    
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("CUTLASS GEMM failed with status code: " + std::to_string(static_cast<int>(status)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("cudaDeviceSynchronize failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Copy result back to host
    std::vector<float> C(M * N);
    err = cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy C from device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

double benchmark_cutlass_kernel(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    int num_runs) {

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    cudaError_t err;
    err = cudaMalloc(&d_A, A.size() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for A: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_B, B.size() * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error("Failed to allocate device memory for B: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_C, M * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("Failed to allocate device memory for C: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy A to device: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy B to device: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemset(d_C, 0, M * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to memset C on device: " + std::string(cudaGetErrorString(err)));
    }
    
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1.0f, 0.0f}
    );
    
    // Warmup kernel execution
    for (int i = 0; i < 10; ++i) {
        gemm_op(args);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cudaEventRecord(start_event);
    for (int i = 0; i < num_runs; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            throw std::runtime_error("CUTLASS GEMM kernel execution failed inside benchmark loop");
        }
    }
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    float kernel_milliseconds = 0;
    cudaEventElapsedTime(&kernel_milliseconds, start_event, stop_event);
    double avg_kernel_time = (kernel_milliseconds / 1000.0) / num_runs;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return avg_kernel_time;
}
