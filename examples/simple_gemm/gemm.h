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
