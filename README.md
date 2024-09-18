# Mixtral offloading

This project is a fork of the original [Mixtral offloading](https://github.com/dvmazur/mixtral-offloading) project. Our main contributions involve testing the performance of various caching strategies (including layer-wise independent caching strategies to handle varying distributions of expert selections across layers), as well as upper-bounding the performance of speculative decoding by hard-coding expert activations for a selected set of prompts.

Specifically in [dvmazur/mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) they used i) LRU caching of experts, and ii) speculative pre-loading by predicting the active experts ahead of time, to accelerate token generation. In our project, we delve deeper into these two ideas and conduct a comprehensive analysis. Our investigation revealed the following:
- Performance (measured by throughput) is largely unaffected by the caching strategy, and techniques such as LRU and LFU caching offer only marginal improvements over totally random cache eviction policies.
- Speculative pre-loading of experts offers no further performance gains for 4-bit quantized MoE inference, and is bound by CPU-GPU communication overheads.
- Reducing communication between the GPU and CPU and conducting inference on the CPU presents a favourable approach for MoE inference. Consequently, development of quantized multi-precision operation kernels for CPU inference presents the most promising, but challenging direction for further optimization.
