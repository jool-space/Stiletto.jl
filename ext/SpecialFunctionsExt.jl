module SpecialFunctionsExt

using Stiletto: Stiletto
using SpecialFunctions: SpecialFunctions
using cuDNN: CUDNN_POINTWISE_ERF

Stiletto.unary_mode(::typeof(SpecialFunctions.erf)) = CUDNN_POINTWISE_ERF

end
