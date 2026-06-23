# Quantization

Quantization is the process of reducing the precision of a given model layer.


## With dynamic scaling

Not all quantization to the same format is equal - there is also the idea of *dynamic scaling* - having a single scaled coefficient for a given block of matrices.

If this exists, INT8 quantization can exceed FP8 quantization (see https://arxiv.org/pdf/2510.25602v1).