module MicroscalingExt

# Block-scaled (MXFP8/NVFP4) arguments in traced code. Declaration goes
# through Microscaling's own cuDNN hook — `tensor!(g, ::BlockscaledArray)`
# adds an element tensor, a swizzled scale tensor, and a dequantize node,
# returning the virtual dequantized tensor the trace's consumers use — lifted
# to Stiletto's uniform graph rank. Binding then splits the argument into its
# storage components.

using Stiletto: Stiletto, TracedArray
using Microscaling: BlockscaledArray, elements, scales, block_size,
    element_type, scale_type
using LinearAlgebra: Adjoint, Transpose
using cuDNN: cuDNN, Graph, Tensor, block_scale_quantize!

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

# the composite binds by role: an input's tensor is the virtual dequantize
# output (bind the components feeding it); a quantize destination's tensors
# are the op's element and scale outputs (bind the matching component)
function Stiletto.bind!(g::Graph, bindings, t::Tensor, a::BlockscaledArray)
    for op in g.ops
        if op isa cuDNN.BlockScaleDequantizeOp && op.y === t
            bindings[op.x] = elements(a)
            bindings[op.scale] = scales(a)
            return bindings
        elseif op isa cuDNN.BlockScaleQuantizeOp && op.y === t
            bindings[t] = elements(a)
            return bindings
        elseif op isa cuDNN.BlockScaleQuantizeOp && op.scale === t
            bindings[t] = scales(a)
            return bindings
        end
    end
    throw(ArgumentError("tensor $(t.name) is not a block-scale operand"))
end

# strides of the composite are meaningless; the signature is the components'
Stiletto.argkey(a::BlockscaledArray) = (typeof(a), size(elements(a)), size(scales(a)))

# ## Quantized outputs
#
# `quantize!(dest, x)` fuses block-scale quantization onto a computed value,
# writing into the composite's storage. Two results (elements, scales), so
# the trace records the op node (primary: elements) plus an Aux projection
# (scales), both destined for the same composite argument.

struct Quantize
    x::Int
    block_size::Int
    block_dim::Int
    dtype::DataType
    scale_dtype::DataType
end
Stiletto.noderefs(n::Quantize) = (n.x,)

function Stiletto.quantize!(dest::TracedArray, x::TracedArray)
    tr = Stiletto.sametrace(dest, x)
    leaf = tr.nodes[dest.id]
    leaf isa Stiletto.Leaf || throw(ArgumentError(
        "quantize! destination must be an argument of the trace"))
    ex = Stiletto.input_example(leaf)
    ex isa BlockscaledArray || throw(ArgumentError(
        "quantize! writes a block-scaled composite; the destination is a $(typeof(ex))"))
    size(dest) == size(x) || throw(DimensionMismatch(
        "cannot quantize a $(size(x)) value into a $(size(dest)) destination"))
    blk = block_size(ex)
    bd = findfirst(b -> b !== 1, blk)
    bd !== nothing && blk[bd] isa Int &&
        all(i -> i == bd || blk[i] === 1, eachindex(blk)) || throw(ArgumentError(
        "cuDNN quantizes one dimension in integer blocks; got $(blk) blocks"))
    any(==(dest.id), values(tr.destinations)) &&
        throw(ArgumentError("destination is written more than once"))
    q = Stiletto.traced(tr, element_type(ex), x.dims,
                        Quantize(x.id, blk[bd], bd, element_type(ex), scale_type(ex)))
    s = Stiletto.traced(tr, scale_type(ex), size(scales(ex)), Stiletto.Aux(q.id, :scale))
    tr.destinations[q.id] = dest.id   # elements and scales share the composite
    tr.destinations[s.id] = dest.id
    return q
end

function Stiletto.emit!(g::Graph, tf, i, node::Quantize, R)
    y, scale = block_scale_quantize!(g, tf(node.x); block_size=node.block_size,
                                     block_dim=node.block_dim, dtype=node.dtype,
                                     scale_dtype=node.scale_dtype)
    tf.aux[(i, :scale)] = scale
    return y
end

function Stiletto.quantize!(dest::BlockscaledArray, x::AbstractArray)
    Stiletto.jit((d, x) -> (Stiletto.quantize!(d, x); nothing), dest, x)
    return dest
end

end
