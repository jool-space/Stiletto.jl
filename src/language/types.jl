# ## Trace IR
#
# Nodes reference earlier nodes by index, so the vector is topologically
# ordered by construction. Only what emission needs is stored; logical shapes
# live on the TracedArray values and are recomputed by the cuDNN layer.

struct Leaf                     # positional argument of the traced function
    argindex::Int
    example::Any
end
struct Captured                 # array closed over inside the traced function
    array::Any
end
struct Constant                 # scalar literal, bound by value at execution
    value::Float32
end
struct Presented                # stride re-presentation of an input (transpose)
    src::Int
    perm::Vector{Int}
end
struct Reshaped                 # contiguous re-presentation of an input
    src::Int
    dims::Vector{Int}
end
struct Matmul
    a::Int
    b::Int
end
struct Pointwise
    mode::cudnnPointwiseMode_t
    args::Vector{Int}
end
struct Reduction
    mode::Symbol
    x::Int
    dims::Vector{Int}
end
struct Norm
    mode::Symbol
    x::Int
    scale::Int
    bias::Union{Nothing,Int}
    mean::Union{Nothing,Int}          # batch norm inference statistics
    inv_variance::Union{Nothing,Int}
    epsilon::Union{Nothing,Int}       # Constant node id (per-sample norms)
end
struct Cast                 # dtype conversion: IDENTITY with a typed output
    x::Int
    T::DataType
end
struct Sdpa                 # scaled dot-product attention (unified SDPA op)
    q::Int
    k::Int
    v::Int
    scale::Int              # Constant node, bound by value at execution
    causal::Bool            # trace-time: bakes a mask subgraph into the graph
end
struct Conv                 # cross-correlation; groups inferred from channels
    x::Int
    w::Int
    pre_padding::Vector{Int}
    post_padding::Vector{Int}
    stride::Vector{Int}
    dilation::Vector{Int}
end
struct Resample             # pooling: windowed reduction over spatial dims
    x::Int
    mode::Symbol
    window::Vector{Int}
    pre_padding::Vector{Int}
    post_padding::Vector{Int}
    stride::Vector{Int}
end

mutable struct Trace
    nodes::Vector{Any}
    destinations::Dict{Int,Int}   # computed node id => input node whose buffer it writes
end
Trace() = Trace(Any[], Dict{Int,Int}())

input_example(node::Leaf) = node.example
input_example(node::Captured) = node.array

# node ids read by a node, for the write-after-read guard
noderefs(n::Matmul) = (n.a, n.b)
noderefs(n::Pointwise) = Tuple(n.args)
noderefs(n::Reduction) = (n.x,)
noderefs(n::Norm) =
    Tuple(id for id in (n.x, n.scale, n.bias, n.mean, n.inv_variance, n.epsilon)
          if id !== nothing)
noderefs(n::Cast) = (n.x,)
noderefs(n::Sdpa) = (n.q, n.k, n.v, n.scale)
noderefs(n::Conv) = (n.x, n.w)
noderefs(n::Resample) = (n.x,)
noderefs(n::Presented) = (n.src,)
noderefs(n::Reshaped) = (n.src,)
noderefs(n) = ()

# Subtyping AbstractArray admits TracedArrays into generic ::AbstractArray
# code; interface promises a symbolic value cannot keep (element access,
# allocation) throw informative errors instead.
struct TracedArray{T,N} <: AbstractArray{T,N}
    trace::Trace
    id::Int
    dims::NTuple{N,Int}
end

function traced(tr::Trace, ::Type{T}, dims::NTuple{N,Int}, node) where {T,N}
    push!(tr.nodes, node)
    return TracedArray{T,N}(tr, length(tr.nodes), dims)
end

Base.size(t::TracedArray) = t.dims
Base.getindex(t::TracedArray, i...) = throw(ArgumentError(
    "TracedArrays are symbolic: element access cannot be traced"))
Base.setindex!(t::TracedArray, v, i...) = throw(ArgumentError(
    "TracedArrays are symbolic: element assignment cannot be traced; use y .= ..."))
Base.similar(t::TracedArray, ::Type{T}, dims::Dims) where {T} = throw(ArgumentError(
    "TracedArrays are symbolic: outputs are the values the traced function returns"))
Base.show(io::IO, t::TracedArray{T,N}) where {T,N} =
    print(io, join(t.dims, "×"), " TracedArray{", T, ",", N, "} (node ", t.id, ")")
Base.show(io::IO, ::MIME"text/plain", t::TracedArray) = show(io, t)

function sametrace(ts::TracedArray...)
    tr = first(ts).trace
    all(t -> t.trace === tr, ts) ||
        throw(ArgumentError("cannot mix values from different traces"))
    return tr
end

capture(tr::Trace, a::AbstractArray) = traced(tr, eltype(a), size(a), Captured(a))
