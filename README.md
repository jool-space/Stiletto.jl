# Stiletto.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Stiletto.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Stiletto.jl/dev/)
[![Build Status](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Stiletto.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Stiletto.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Stiletto.jl)

Traces plain Julia array code into fused [cuDNN graphs](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html).

```julia
using CUDACore, Stiletto

A, B = CuArray(rand(Float32, 64, 48)), CuArray(rand(Float32, 48, 32))

f = compile(A, B) do a, b
    max.(a * b .- 1f0, 0f0)  # one fused kernel
end
f(A, B)                      # 64×32 CuArray

@jit f(A, B)
```

Values are symbolic while the traced function runs; cuDNN tensors are declared
only once the whole graph is known, so rank policy — matmul wants rank-3
operands, pointwise broadcast wants uniform ranks — is decided from all use
sites at once. `*` (2-D, and 3-D with trailing broadcastable batch),
broadcasting over the mapped pointwise functions, `sum`/`maximum`/`minimum`
(reduced dims kept as singletons), `adjoint`/`transpose`/`permutedims` of
inputs, closed-over arrays, and scalar literals all trace.
