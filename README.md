# Stiletto.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Stiletto.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Stiletto.jl/dev/)
[![Build Status](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Stiletto.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Stiletto.jl)

Traces plain Julia array code into fused [cuDNN graphs](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html).

```julia
using Stiletto, CUDA

M, N, K = 64, 32, 48
A = CUDA.randn(Float32, K, M)
B = CUDA.randn(Float32, K, N)

function matmul_epilogue(a::AbstractMatrix, b::AbstractMatrix)
    tanh.(transpose(a) * b / √K)
end

C = @jit matmul_epilogue(A, B)   # 64×32 CuArray
```
