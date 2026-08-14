module NNlibExt

using Stiletto: Stiletto, TracedArray
using NNlib: NNlib
using cuDNN: CUDNN_POINTWISE_RELU_FWD, CUDNN_POINTWISE_SIGMOID_FWD,
    CUDNN_POINTWISE_TANH_FWD, CUDNN_POINTWISE_ELU_FWD, CUDNN_POINTWISE_SOFTPLUS_FWD,
    CUDNN_POINTWISE_SWISH_FWD, CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, CUDNN_POINTWISE_GELU_FWD

Stiletto.unary_mode(::typeof(NNlib.relu)) = CUDNN_POINTWISE_RELU_FWD
Stiletto.unary_mode(::typeof(NNlib.sigmoid)) = CUDNN_POINTWISE_SIGMOID_FWD
Stiletto.unary_mode(::typeof(NNlib.sigmoid_fast)) = CUDNN_POINTWISE_SIGMOID_FWD
Stiletto.unary_mode(::typeof(NNlib.tanh_fast)) = CUDNN_POINTWISE_TANH_FWD
Stiletto.unary_mode(::typeof(NNlib.elu)) = CUDNN_POINTWISE_ELU_FWD
Stiletto.unary_mode(::typeof(NNlib.softplus)) = CUDNN_POINTWISE_SOFTPLUS_FWD
Stiletto.unary_mode(::typeof(NNlib.swish)) = CUDNN_POINTWISE_SWISH_FWD
Stiletto.unary_mode(::typeof(NNlib.gelu_tanh)) = CUDNN_POINTWISE_GELU_APPROX_TANH_FWD
Stiletto.unary_mode(::typeof(NNlib.gelu_erf)) = CUDNN_POINTWISE_GELU_FWD

# NNlib's batched verbs on traced values route onto the traced matmul, whose
# semantics are a strict superset (any number of batch dims, broadcasting).
# Same-rank signatures mirror NNlib's own methods so these dominate them
# without ambiguity.
NNlib.batched_mul(a::TracedArray{<:Any,N}, b::TracedArray{<:Any,N}) where {N} =
    Stiletto.batched_mul(a, b)
NNlib.batched_mul(a::TracedArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N} =
    Stiletto.batched_mul(a, b)
NNlib.batched_mul(a::AbstractArray{<:Any,N}, b::TracedArray{<:Any,N}) where {N} =
    Stiletto.batched_mul(a, b)
NNlib.batched_mul!(c::TracedArray{<:Any,N}, a::AbstractArray{<:Any,N},
                   b::AbstractArray{<:Any,N}) where {N} =
    Stiletto.batched_mul!(c, a, b)

# batched transpose/adjoint are stride re-presentations of trace inputs
NNlib.batched_transpose(t::TracedArray) =
    permutedims(t, (2, 1, ntuple(i -> i + 2, ndims(t) - 2)...))
NNlib.batched_adjoint(t::TracedArray{<:Real}) = NNlib.batched_transpose(t)

end
