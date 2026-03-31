# General Response:

We thank all the reviewers for their constructive feedback and insightful questions. In response to interest in computational performance, we provide a detailed throughput analysis across various chunk sizes ($C$) and sequence lengths ($L$).

Training Throughput Comparison (k tokens/sec).
Benchmark setup: 110M parameter models trained on a 2×2 TPU Trillium setup.

For Chunk Size $C=64$ 

| Model | L=2K | L=4K | L=8K | L=16K |
| :--- | :--- | :--- | :--- | :--- |
| Attention (blocksize=256) | 342K | 211K | 119K | 65K |
| TTT | 350K | 288K | 305K | 302K |
| Lattice | 247K | 211K | 202K | 187K |
| Mamba2 | 310K | 297K | 281K | 261K |
| Gared DeltaNet | 34K | 54K | 50K | 46K |


For Chunk Size $C=16$
  
| Model | L=2K | L=4K | L=8K | L=16K |
| :--- | :--- | :--- | :--- | :--- |
| Attention (blocksize=256) | 342K | 211K | 119K | 65K |
| TTT | 310K | 233K | 218K | 189K |
| Lattice | 184K | 184K | 161K | 123K |
| Mamba2 | 318K | 268K | 231K | 183K |
| Gared DeltaNet | 99K | 45K | 44K | 41K |

For Chunk Size $C=4$ 

| Model | L=2K | L=4K | L=8K | L=16K |
| :--- | :--- | :--- | :--- | :--- |
| Attention (blocksize=256) | 342K | 211K | 119K | 65K |
| TTT | 176K | 132K | 102K | 70K |
| Lattice | 125K | 107K | 84K | 58K |
| Mamba2 | 139K | 122K | 121K | 77K |
| Gared DeltaNet | 109K | 35K | 32K | 26K |

To ensure a fair comparison of algorithmic complexity, all models (except the Transformer) were implemented in pure Jax without specialized hardware-optimized kernels. The Transformer++ uses a highly optimized Flash Attention  kernels implemented in Pallas.

- While Gated-DeltaNet is a strong baseline, its requirement to solve lower triangular systems introduces computational overhead that becomes more restrictive as chunk size increases.

- All the models can be further accelerated with hardware-specific customized kernels (e.g., Triton or Pallas) but the focus of these results is to show that Lattice’s mathematical formulation is inherently scalable.

We believe these results, combined with our state-efficiency gains, MQAR accuracy and language modeling performance, demonstrate that Lattice achieves a favorable trade-off between the expressive power of non-linear state-dependent updates and the training efficiency required for long sequence setting.
