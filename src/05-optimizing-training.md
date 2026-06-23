# Optimizing Training

TODO: proof-read and review.

## Differences between training and inference

Training has a few factors that differ from inference:

- the kernels executed and optimization may change due to the need to perform backpropagation. The gradients for each layer must be stored

## New techniques to training:

- hiding dataloading latency: can we load data for each batch in training while performing inference, backpropagation, and synchronization across cluster?