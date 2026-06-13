# Glossary

This glossary defines key terms and concepts related to machine learning model optimization and hardware execution.

## Inference

**Inference** is the phase where a trained machine learning model is deployed to generate predictions, classify inputs, or produce outputs based on new, unseen data. 

Unlike the training phase, which requires computing gradients and updating model weights via backpropagation, inference consists only of the forward pass. Consequently, the optimization goals for inference focus on:
- **Latency**: Minimizing the time taken to process a single request (crucial for real-time applications).
- **Throughput**: Maximizing the number of requests processed per second.
- **Resource Utilization**: Minimizing memory footprint (e.g., model weight storage, KV cache) and maximizing utilization of hardware compute units (e.g., Tensor Cores on GPUs).
