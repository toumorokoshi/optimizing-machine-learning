#pragma once

#include <vector>

// Performs matrix multiplication: C = A * B
// A is of size M x K
// B is of size K x N
// C is of size M x N (returned)
std::vector<float> gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K);

// Performs matrix multiplication on GPU using CUTLASS: C = A * B
// A is of size M x K
// B is of size K x N
// C is of size M x N (returned)
std::vector<float> gemm_cutlass(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K);

// Benchmarks the pure CUTLASS GEMM kernel execution time (excluding host-device transfers and mallocs)
// Returns the average execution time in seconds.
double benchmark_cutlass_kernel(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    int num_runs);
