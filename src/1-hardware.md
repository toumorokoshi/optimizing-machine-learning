# Understanding the Hardware

Before looking at optimization of the actual individual models, it's valuable to understanding the fundamental limitations of the underlying hardware that Machine learning workloads run on.

With respect to hardware configuration, the following values are the leading factors impacting inference latency:

## TOPs

Short for trillions of operations per second. It is a standard unit of measurement of the speed at which an AI accelerator is able to perform an individual arithmetic operation.

However, a benchmark of a TOPs for a given accelerator does not always provide an apples to apples direct comparison across AI accelerators - there are multiple additional dimensions that should be considered in that benchmark including:

- precision: what is the data type that the TOPs has been benchmarked with? fp32, fp16, bf16, int8, etc.
- sparse or dense: for matrix multiplcation specifically, many companies use the latency of performing an NxM matrix multiplication on a sparse matrix, where many of the values may be zero. Many machine learning models operate against dense matrices, so a sparse benchmark may not be a realistic throughput target as compared to a dense measurement.

## DRAM Memory Bandwidth

DRAM memory bandwidth will inform the speed by which the model can pre-fill KV caches and perform operations that do not support L2 cache centric algorithms such as FlashAttention.

Transformers are often limited by memory bandwidth.

For the purpose of this investigation, the comparable platforms often use one of:

## Precision

The TOPs of a particular compute make an assumption about the data type of the number. This is relevant as larger data types can have increased precision and therefore lead to improved model performance. However, higher precision also comes with increased data bandwidth needs depending on if the model weights themselves are stored at a higher precision (which they often are), as well as whether data types are cast to a lower precision before being written / read from DRAM and L2 cache.

## Register / L2 Cache Size

Overcoming throughput limitations of DRAM access heavily relies on the sizes of various temporary caches or registers available close to the streaming multiprocessor. Larger L2 cache sizes and significantly increase the throughput of flashattention by increasing the size of the query, key, and value tiles that are stored within them and therefore do not require DRAM access.