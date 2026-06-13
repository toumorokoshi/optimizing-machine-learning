# Code Execution via Bazel

This section demonstrates a simple C++ baseline implementation of GEMM (General Matrix Multiply) and shows how we test it using Bazel and include it dynamically in this book.

## Baseline GEMM Implementation

Below is the simple matrix multiplication code located in [gemm.cc](file:///home/yusuke/workspace/optimizing-machine-learning/examples/simple_gemm/gemm.cc).

```cpp
{{#include ../examples/simple_gemm/gemm.cc}}
```

## Unit Verification

To ensure that the logic is correct, we write a unit test at [gemm_test.cc](file:///home/yusuke/workspace/optimizing-machine-learning/examples/simple_gemm/gemm_test.cc):

```cpp
{{#include ../examples/simple_gemm/gemm_test.cc}}
```

## Running the Code via Bazel

You can verify and run this target locally. This will download any needed dependencies (like compiler toolchains) and build/run the test:

```bash
# Run the GEMM unit test
bazel test //examples/simple_gemm:gemm_test
```
