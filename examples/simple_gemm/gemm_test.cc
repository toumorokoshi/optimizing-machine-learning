#include "gemm.h"
#include <iostream>
#include <cmath>

int main() {
    // 2x3 Matrix A
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // 3x2 Matrix B
    // [ 7.0,  8.0  ]
    // [ 9.0,  10.0 ]
    // [ 11.0, 12.0 ]
    std::vector<float> B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    // Expected Output C (2x2)
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};

    auto result = gemm_naive(A, B, 2, 2, 3);

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-5f) {
            std::cerr << "Assertion failed at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << result[i] << std::endl;
            return 1;
        }
    }

    std::cout << "All GEMM tests passed successfully!" << std::endl;
    return 0;
}
