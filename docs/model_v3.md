# PriST-RIS model contract

The paper-facing model name and all newly written metadata use **PriST-RIS**. The implementation has a single `PriSTRIS` class with four canonical configurations:

| Key | Structured spatial | Ridge anchor | Cross-attention | Low-rank temporal |
|---|---|---|---|---|
| `prist_ris_a` | yes | no | no | no |
| `prist_ris_b` | yes | yes | no | no |
| `prist_ris_c` | yes | yes | yes | no |
| `prist_ris_full` | yes | yes | yes | yes |

The default width is 80 with stage depths `[2,2,3]`, four attention heads, zero dropout, and temporal rank 2. Width candidates are 64/80/96 and temporal ranks are 2/3 only.

The backbone maps `[B,4,64,32]` to `[B,H,64,256]`; Quasi pads its two absent observation channels so source and target models remain structurally compatible. Each progressive stage doubles only the last (RIS) dimension. Factorized blocks contain independent depthwise `1x3` and `3x1` paths followed by pointwise channel mixing.

The Ridge prediction is encoded and added to dense features. The learned anchor head produces an explicit residual over that prior. Cross-attention obtains keys and values only from observed tensors. The temporal module produces complex spatial bases and query coefficients, then performs explicit complex multiplication. Queries 0 and 1 reuse the matching observed contexts; later queries use only pooled observed context and time encoding. The compact temporal correction has one depthwise and one pointwise layer.
