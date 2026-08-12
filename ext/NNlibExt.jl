module NNlibExt

using Stiletto: Stiletto
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

end
