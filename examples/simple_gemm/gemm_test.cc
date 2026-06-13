#include "gemm.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

int main() {
    // ----------------------------------------------------
    // Test 1: Verification with a small matrix
    // ----------------------------------------------------
    // 2x3 Matrix A
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // 3x2 Matrix B
    std::vector<float> B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    // Expected Output C (2x2)
    std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};

    auto result_naive = gemm_naive(A, B, 2, 2, 3);
    auto result_cutlass = gemm_cutlass(A, B, 2, 2, 3);

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(result_naive[i] - expected[i]) > 1e-5f) {
            std::cerr << "Naive GEMM assertion failed at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << result_naive[i] << std::endl;
            return 1;
        }
        if (std::abs(result_cutlass[i] - expected[i]) > 1e-5f) {
            std::cerr << "CUTLASS GEMM assertion failed at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << result_cutlass[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Test 1: Small matrix verification passed!" << std::endl;

    // ----------------------------------------------------
    // Test 2: Verification with a larger random matrix
    // ----------------------------------------------------
    int check_M = 128;
    int check_N = 128;
    int check_K = 128;
    
    std::vector<float> check_A(check_M * check_K);
    std::vector<float> check_B(check_K * check_N);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : check_A) val = dis(gen);
    for (auto& val : check_B) val = dis(gen);
    
    auto check_naive = gemm_naive(check_A, check_B, check_M, check_N, check_K);
    auto check_cutlass = gemm_cutlass(check_A, check_B, check_M, check_N, check_K);
    
    double max_diff = 0.0;
    for (size_t i = 0; i < check_naive.size(); ++i) {
        double diff = std::abs(check_naive[i] - check_cutlass[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "Test 2: Large random matrix verification. Max absolute difference: " << max_diff << std::endl;
    if (max_diff > 1e-4) {
        std::cerr << "Test 2 Failed! CUTLASS output does not match naive CPU output." << std::endl;
        return 1;
    }
    std::cout << "Test 2 Passed!" << std::endl;

    // ----------------------------------------------------
    // Benchmark: Performance measurement
    // ----------------------------------------------------
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    std::vector<float> bench_A(M * K);
    std::vector<float> bench_B(K * N);
    for (auto& val : bench_A) val = dis(gen);
    for (auto& val : bench_B) val = dis(gen);
    
    // Total floating point operations for GEMM: 2 * M * N * K
    double num_flops = 2.0 * double(M) * double(N) * double(K);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "GEMM Benchmark (Size: " << M << "x" << N << "x" << K << ")" << std::endl;
    std::cout << "Total Operations: " << num_flops / 1e9 << " GFLOPs" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Benchmark Naive CPU GEMM (1 iteration is enough since it's slow)
    std::cout << "Running Naive CPU GEMM..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_res = gemm_naive(bench_A, bench_B, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_end - cpu_start;
    double cpu_time = cpu_diff.count();
    double cpu_gflops = (num_flops / 1e9) / cpu_time;
    std::cout << "Naive CPU GEMM: " << std::fixed << std::setprecision(4) 
              << cpu_time << " seconds, " << cpu_gflops << " GFLOPS" << std::endl;

    // 2. Benchmark End-to-End CUTLASS GEMM (includes allocation and memory copy)
    std::cout << "\nRunning End-to-End CUTLASS GEMM (Warmup)..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        auto dummy = gemm_cutlass(bench_A, bench_B, M, N, K);
    }
    
    std::cout << "Running End-to-End CUTLASS GEMM Benchmark..." << std::endl;
    int num_runs = 50;
    auto e2e_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        auto dummy = gemm_cutlass(bench_A, bench_B, M, N, K);
    }
    auto e2e_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e2e_diff = e2e_end - e2e_start;
    double e2e_avg_time = e2e_diff.count() / num_runs;
    double e2e_gflops = (num_flops / 1e9) / e2e_avg_time;
    std::cout << "End-to-End CUTLASS (alloc + copy + run): " 
              << std::fixed << std::setprecision(4) 
              << e2e_avg_time << " seconds, " << e2e_gflops << " GFLOPS" << std::endl;

    // 3. Benchmark Pure CUTLASS GEMM Kernel (excluding memory transfer & allocation)
    std::cout << "\nRunning Pure CUTLASS GEMM Kernel Benchmark..." << std::endl;
    double avg_kernel_time = benchmark_cutlass_kernel(bench_A, bench_B, M, N, K, num_runs);
    double kernel_gflops = (num_flops / 1e9) / avg_kernel_time;
    
    std::cout << "Pure CUTLASS GEMM Kernel: " 
              << std::fixed << std::setprecision(6) 
              << avg_kernel_time << " seconds, " << kernel_gflops << " GFLOPS" << std::endl;

    std::cout << "\nAll benchmarks completed successfully!" << std::endl;
    return 0;
}
