# Adapter

Adapter is in charge of building a global graph and its inputs based on the fused op sequences returned by Capturer and user input.


## Features

### Parallel

#### TP

Tensor Parallel. We have to identify ops that support Tensor Parallel, changing their dim and adding communication op into right place.

#### DP

Data Parallel.

#### EP

Expert Parallel for MoE models. The raw op sequences usually represented FusedEP impl in modeling.py, we hope a DeepEP impl instead. So, we have to identify the MoE parts in raw op sequences, adding `Dispatch` and `Combine` op, changing Experts computation in `Gemm`. Additionally, placing shared experts in independent rank need to be considered.

### MTP

### Quant

### Prefix Cache

### Chunked Prefill
