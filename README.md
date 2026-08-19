# Stiletto.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Stiletto.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Stiletto.jl/dev/)
[![Build Status](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Stiletto.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Stiletto.jl)

Traces plain Julia array code into fused [cuDNN graphs](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html).

```julia
using Stiletto, CUDA

M, N, K = 640, 320, 480
A = CUDA.randn(Float32, K, M)
B = CUDA.randn(Float32, K, N)

function matmul_epilogue(a::AbstractMatrix, b::AbstractMatrix)
    sum(tanh.(transpose(a) * b / √K), dims=1)
end

C = @jit matmul_epilogue(A, B)   # 1×320 CuArray, 1 allocation

function matmul_epilogue!(c::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    c .= matmul_epilogue(a, b)
end

@jit matmul_epilogue!(C, A, B)   # 0 allocations
```

Stiletto will try to run any graph supported by cuDNN, but not all graphs will run
due limitations of cuDNN fusion and engine selection.

## Installation

```julia
using Pkg
Registry.add(url="https://registry.jool.space")
Pkg.add("Stiletto")
```
