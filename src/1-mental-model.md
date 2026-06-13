# The Overall Mental Model

When it comes to optimizing machine learning models, there are two broad categories of improvment:

1. Improving the execution of the model more efficiently, without modifying the model itself.
    - fusion of the kernels, to leverage cache locality.
    - leveraging compute-specific functionality such as asynchronous memory loading (e.g. Tensor Memory Accelerators with NVidia GPUs).
2. Modifying the model to be more performant generally, regardless of the compute platform.
    - Quantization: lowering the data precision of a given layer.
2. Modifying the model to be more performant, to have affinity with that specific hardware.
    - modifying the size of the model parameters and operations to align with block / thread sizes that are efficient for the target compute.
    - sparse matrix multiplication for systems that can detect and perform sparse matrix multiplication.

As a machine learning workload optimizer, the optimizer would have to consider which of these techniques will result in the most significant improvements in resource utilization and latency, prioritize those, and perform them appropriately.