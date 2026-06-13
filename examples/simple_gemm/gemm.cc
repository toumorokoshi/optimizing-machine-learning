#include "gemm.h"
#include <stdexcept>

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
