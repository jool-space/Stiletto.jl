```@meta
CurrentModule = Stiletto
```

# Stiletto

Stiletto traces plain Julia array code — `*`, `mul!`, broadcasting,
reductions, `transpose`, `reshape`, `view`, `y .= ...` — into fused cuDNN
graphs. Values are symbolic while the traced function runs; cuDNN tensors
are declared only once the whole graph is known, so decisions that span the
entire computation are made from all use sites at once:

- **Rank unification** — cuDNN wants matmul operands at rank ≥ 3 and
  pointwise broadcasts at uniform rank; every tensor is declared at the
  maximum rank any operation requires, lifted by trailing singletons (free
  for column-major storage).
- **Composite values** — an argument may back several graph tensors (a
  block-scaled array declares elements, swizzled scales, and a dequantize
  node) through the `declare`/`bind!`/`argkey` extension seams.
- **In-place and presentation semantics** — `mul!` and `y .= ...` become
  graph outputs bound to the caller's buffers; `transpose`, `permutedims`,
  `reshape`, and `view` are free stride re-presentations, not operations.

Engine support is never faked: when cuDNN has no engine for a graph,
compilation throws `cuDNN.UnsupportedGraphError` rather than falling back
to a different computation.

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

One graph, one kernel launch: the gemm engine takes `tanh` and `/√K` as its
pointwise epilogue and the `sum` fuses as a terminal reduction, so the only
allocation is the output — and the in-place form writes the caller's buffer
directly. Traced functions compose like the ordinary Julia functions they
are.

## Compiling and executing

`compile` traces once and returns a callable; `jit` compiles per argument
signature on first use and executes the cached plan, keyed on shapes,
strides, dtypes — and on the native code of the traced call, so redefining
a function (or anything it calls) retraces instead of replaying a stale
graph.

```@docs
compile
@compile
jit
@jit
TracedArray
```

The compiled callable can be split into its two halves for callers that
schedule execution themselves:

```@docs
Stiletto.graph
Stiletto.bindings
Stiletto.workspace
```

## Traced semantics

Traced code keeps Base's meaning. Output element types follow Julia's
promotion of the traced computation (explicit `io_dtype` overrides that as
a precision policy — see [`compile`](@ref)). Base's mutating idioms trace
as written: a trailing `mul!(c, a, b)`, the returned-destination form
`(mul!(c, a, b); c)`, and `y .= ...; y` all hand back the caller's buffer.

Arguments are runtime inputs — arrays rebind pointers per call, scalars
become by-value tensors. Captured values are trace-time constants: isbits
captures (flags, dimensions, `size(x, 1)`) specialize the graph per value
and may be branched on; captured arrays are baked by identity. Pass data as
arguments; capture configuration.

## Operator library

Named operations built on the tracer, each in two tiers: `TracedArray`
methods become nodes of the surrounding trace, and plain-array methods
jit-compile a standalone graph — which is also how operands no BLAS method
claims (block-scaled composites, narrow dtypes) get eager execution.

### Matrix multiplication

```@docs
Stiletto.mul!
Stiletto.batched_mul
```

### Attention

```@docs
Stiletto.attention
Stiletto.attention_backward
```

### Normalization

```@docs
Stiletto.rmsnorm
Stiletto.batchnorm
```

### Convolution and pooling

```@docs
Stiletto.conv
Stiletto.maxpool
```

### Quantization

```@docs
Stiletto.quantize!
```

## Extensions

- **NNlib** — activation functions map to cuDNN pointwise modes;
  `NNlib.batched_mul`/`batched_mul!` and `batched_transpose`/`batched_adjoint`
  on traced values route onto the traced matmul, whose semantics are a
  strict superset.
- **SpecialFunctions** — `erf` traces as a pointwise mode.
- **Microscaling** — `BlockscaledArray` arguments trace like any array,
  declaring element and swizzled-scale tensors joined by a dequantize node
  and binding their storage components at execution; `quantize!` fuses
  block-scale quantization onto computed values, writing into a composite's
  storage.

## Index

```@index
```
