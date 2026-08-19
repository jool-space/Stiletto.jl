module Stiletto

# Traces plain Julia array code — `*`, broadcasting, `sum`, `adjoint`,
# `mul!`, `.=` — into a cuDNN graph. Values are symbolic while the traced
# function runs; cuDNN tensors are declared only once the whole graph is
# known, so rank policy (matmul needs rank-3 operands, pointwise broadcast
# needs uniform rank) is decided from all use sites at once: every tensor is
# declared at the maximum rank any operation requires, lifted by trailing
# singleton dimensions, which is free for column-major storage.

using CUDACore: CUDACore, CuArray
using CompilerCaching: CompilerCaching, CacheView
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using Statistics: Statistics
using cuDNN: cuDNN, Graph, Tensor, tensor!, scalar!, output!, matmul!, pointwise!,
    reduction!, norm_fwd!, sdpa_fwd!, sdpa_bwd!, conv_fprop!, resample_fwd!,
    execute!, build!, UnsupportedGraphError,
    cudnnPointwiseMode_t,
    CUDNN_POINTWISE_ADD, CUDNN_POINTWISE_MUL, CUDNN_POINTWISE_SUB, CUDNN_POINTWISE_DIV,
    CUDNN_POINTWISE_MAX, CUDNN_POINTWISE_MIN, CUDNN_POINTWISE_POW, CUDNN_POINTWISE_ATAN2,
    CUDNN_POINTWISE_ABS, CUDNN_POINTWISE_CEIL, CUDNN_POINTWISE_FLOOR,
    CUDNN_POINTWISE_COS, CUDNN_POINTWISE_SIN, CUDNN_POINTWISE_TAN,
    CUDNN_POINTWISE_EXP, CUDNN_POINTWISE_LOG, CUDNN_POINTWISE_SQRT,
    CUDNN_POINTWISE_RSQRT, CUDNN_POINTWISE_NEG, CUDNN_POINTWISE_RECIPROCAL,
    CUDNN_POINTWISE_ERF, CUDNN_POINTWISE_IDENTITY, CUDNN_POINTWISE_TANH_FWD,
    CUDNN_POINTWISE_CMP_EQ, CUDNN_POINTWISE_CMP_NEQ, CUDNN_POINTWISE_CMP_GT,
    CUDNN_POINTWISE_CMP_GE, CUDNN_POINTWISE_CMP_LT, CUDNN_POINTWISE_CMP_LE,
    CUDNN_POINTWISE_LOGICAL_AND, CUDNN_POINTWISE_LOGICAL_OR, CUDNN_POINTWISE_LOGICAL_NOT,
    CUDNN_POINTWISE_BINARY_SELECT,
    CUDNN_DATA_BOOLEAN

export compile, jit, @compile, @jit, TracedArray
public rmsnorm, rmsnorm!, layernorm, layernorm!, batchnorm, batchnorm!
public attention, attention!, attention_backward, attention_backward!
public conv, conv!, maxpool, maxpool!, meanpool, meanpool!
public mul!, batched_mul, batched_mul!, ⊠, quantize!
public graph, bindings, workspace

# Language surface: what traced programs are written in
include("language/types.jl")
include("language/broadcast.jl")
include("language/operations.jl")

# Compiler: traces become executed cuDNN graphs
include("compiler/emission.jl")
include("compiler/compile.jl")
include("compiler/execute.jl")
include("compiler/cache.jl")

# Operator library: named ops built on the tracer, in dispatch tiers
include("library/mul.jl")
include("library/batched_mul.jl")
include("library/norms.jl")
include("library/attention.jl")
include("library/conv.jl")
include("library/pool.jl")
include("library/quantize.jl")

end
