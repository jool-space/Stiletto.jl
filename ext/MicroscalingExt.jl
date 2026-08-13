module MicroscalingExt

# Block-scaled (MXFP8/NVFP4) arguments in traced code. Declaration goes
# through Microscaling's own cuDNN hook — `tensor!(g, ::BlockscaledArray)`
# adds an element tensor, a swizzled scale tensor, and a dequantize node,
# returning the virtual dequantized tensor the trace's consumers use — lifted
# to Stiletto's uniform graph rank. Binding then splits the argument into its
# storage components.

using Stiletto: Stiletto
using Microscaling: BlockscaledArray, elements, scales
using LinearAlgebra: Adjoint, Transpose
using cuDNN: cuDNN, Graph, Tensor

const PermutedBlockscaled{N} =
    PermutedDimsArray{<:Any,N,<:Any,<:Any,<:BlockscaledArray{<:Any,N}}

Stiletto.declare(g::Graph, i, node, ex::Union{BlockscaledArray,PermutedBlockscaled}, R) =
    cuDNN.tensor!(g, ex; name=Stiletto.nodename(i, node), rank=R)

# a captured (or passed) `W'` wraps the composite eagerly — same presentation
# as a permutation; real elements, so adjoint is transpose
const WrappedBlockscaled = Union{Adjoint{<:Real,<:BlockscaledArray{<:Real,2}},
                                 Transpose{<:Real,<:BlockscaledArray{<:Real,2}}}
Stiletto.declare(g::Graph, i, node, ex::WrappedBlockscaled, R) =
    Stiletto.declare(g, i, node, PermutedDimsArray(parent(ex), (2, 1)), R)

# the leaf's tensor is the virtual dequantize output; the buffers bind to the
# element and scale tensors feeding it
function Stiletto.bind!(g::Graph, bindings, t::Tensor, a::BlockscaledArray)
    for op in g.ops
        op isa cuDNN.BlockScaleDequantizeOp && op.y === t || continue
        bindings[op.x] = elements(a)
        bindings[op.scale] = scales(a)
        return bindings
    end
    throw(ArgumentError("tensor $(t.name) is not fed by a block-scale dequantize"))
end

# strides of the composite are meaningless; the signature is the components'
Stiletto.argkey(a::BlockscaledArray) = (typeof(a), size(elements(a)), size(scales(a)))

end
