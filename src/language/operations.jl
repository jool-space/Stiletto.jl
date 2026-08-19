# ## Matmul
#
# Julia dimension order with the batch trailing, matching cuDNN.matmul!:
# (M, K, batch...) × (K, N, batch...) → (M, N, batch...), with batch extents
# broadcasting pairwise — matching or 1, missing dimensions counting as 1.
# `*` and `mul!` keep Base's 2-D semantics; batching is spelled
# `batched_mul`/`⊠`, which exposes the general rank-N form.

function matmul(a::TracedArray, b::TracedArray)
    tr = sametrace(a, b)
    ndims(a) >= 2 && ndims(b) >= 2 ||
        throw(ArgumentError("matmul operands need at least 2 dimensions"))
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "matmul inner dimensions do not match: $(size(a)) × $(size(b))"))
    batch = ntuple(max(ndims(a), ndims(b)) - 2) do i
        da, db = size(a, i + 2), size(b, i + 2)
        da == db || da == 1 || db == 1 || throw(DimensionMismatch(
            "matmul batch dimensions are not broadcastable: $(size(a)) × $(size(b))"))
        max(da, db)
    end
    T = promote_type(eltype(a), eltype(b))
    return traced(tr, T, (size(a, 1), size(b, 2), batch...), Matmul(a.id, b.id))
end

const TracedVector = TracedArray{<:Any,1}
const TracedMatrix = TracedArray{<:Any,2}
Base.:*(a::TracedMatrix, b::TracedMatrix) = matmul(a, b)
Base.:*(a::TracedMatrix, b::AbstractMatrix) = matmul(a, capture(a.trace, b))
Base.:*(a::AbstractMatrix, b::TracedMatrix) = matmul(capture(b.trace, a), b)
# disambiguators against LinearAlgebra's wrapper-specialized *
const LazyWrapped = Union{Adjoint{<:Any,<:AbstractMatrix},Transpose{<:Any,<:AbstractMatrix}}
Base.:*(a::LazyWrapped, b::TracedMatrix) = matmul(capture(b.trace, a), b)
Base.:*(a::TracedMatrix, b::LazyWrapped) = matmul(a, capture(a.trace, b))
Base.:*(a::Adjoint{<:Any,<:AbstractVector}, b::TracedMatrix) = matmul(capture(b.trace, a), b)
Base.:*(a::Transpose{<:Any,<:AbstractVector}, b::TracedMatrix) = matmul(capture(b.trace, a), b)

# ## Presentation
#
# Permutations are free stride changes on inputs. A computed (virtual)
# tensor's layout belongs to the fused kernel mid-graph, but at the output
# boundary the op that produces it can write a permuted-stride output — the
# epilogue write is the permute, no extra op.

function presented(t::TracedArray, perm)
    tr = t.trace
    permv = collect(Int, perm)
    isperm(permv) && length(permv) == ndims(t) ||
        throw(ArgumentError("invalid permutation $perm for $(ndims(t))-dimensional value"))
    node = tr.nodes[t.id]
    dims = ntuple(i -> t.dims[permv[i]], ndims(t))
    if node isa Leaf || node isa Captured
        return traced(tr, eltype(t), dims, Presented(t.id, permv))
    elseif node isa Presented
        return traced(tr, eltype(t), dims, Presented(node.src, node.perm[permv]))
    elseif node isa Union{Reshaped,Constant}
        throw(ArgumentError(
            "cannot permute a reshaped input or constant; permute the original input"))
    end
    return traced(tr, eltype(t), dims, Permuted(t.id, collect(Int, t.dims), permv))
end

Base.permutedims(t::TracedArray, perm) = presented(t, perm)
Base.transpose(t::TracedArray{<:Any,2}) = presented(t, (2, 1))
Base.adjoint(t::TracedArray{T,2}) where {T<:Real} = transpose(t)

# reshape is likewise free on inputs — the declaration changes, the memory
# doesn't — as long as the underlying storage is contiguous
Base.reshape(t::TracedArray, dims::Tuple{Vararg{Union{Int,Colon}}}) = reshape_traced(t, dims)
Base.reshape(t::TracedArray, dims::Tuple{Vararg{Int}}) = reshape_traced(t, dims)
Base.reshape(t::TracedArray, dims::Union{Int,Colon}...) = reshape_traced(t, dims)
Base.reshape(t::TracedArray{<:Any,1}, dims::Tuple{Colon}) = t
Base.reshape(t::TracedArray{<:Any,1}, dims::Colon) = t

function reshape_traced(t::TracedArray, dims)
    rdims = resolve_reshape_dims(length(t), dims)
    rdims == collect(t.dims) && return t
    tr = t.trace
    node = tr.nodes[t.id]
    if node isa Reshaped
        return traced(tr, eltype(t), Tuple(rdims), Reshaped(node.src, rdims))
    elseif node isa Union{Leaf,Captured}
        iscontiguous(input_example(node)) || throw(ArgumentError(
            "reshape needs contiguous storage; this input is a strided view"))
        return traced(tr, eltype(t), Tuple(rdims), Reshaped(t.id, rdims))
    end
    throw(ArgumentError(
        "only inputs of the trace can be reshaped; computed values have a fixed layout"))
end

function resolve_reshape_dims(len, dims)
    count(d -> d isa Colon, dims) <= 1 ||
        throw(DimensionMismatch("reshape accepts at most one Colon"))
    known = prod(Int[d for d in dims if d isa Int]; init=1)
    resolved = Int[d isa Colon ? len ÷ max(known, 1) : d for d in dims]
    prod(resolved; init=1) == len || throw(DimensionMismatch(
        "cannot reshape $len elements to dimensions $(dims)"))
    return resolved
end

iscontiguous(a::AbstractArray) =
    isempty(a) || strides(a) == Tuple(cumprod([1; collect(size(a))[1:end-1]]))

# views of inputs are free stride-and-offset re-presentations: the
# declaration carries the view's strides, binding offsets the parent's
# pointer. Computed values stay unsliceable (their layout belongs to the
# fused kernel), and integer indices are refused rather than dropping
# dimensions the graph would have to reshape away.

function Base.view(t::TracedArray, idxs...)
    tr = t.trace
    length(idxs) == ndims(t) || throw(ArgumentError(
        "view of a traced value takes one index per dimension"))
    ranges = map(enumerate(idxs)) do (d, i)
        # unit steps only: a step on the packed dimension leaves the tensor
        # with no stride-1 dimension, which engines accept and then fault on
        r = i isa Colon ? (1:size(t, d)) :
            i isa AbstractUnitRange ? i :
            throw(ArgumentError(
                "traced views take unit ranges or Colon; integer indices drop " *
                "dimensions and stepped ranges leave no packed dimension"))
        checkindex(Bool, 1:size(t, d), r) || throw(BoundsError(t, idxs))
        convert(UnitRange{Int}, r)
    end
    node = tr.nodes[t.id]
    if node isa Sliced   # views compose: reindex the stored ranges
        return traced(tr, eltype(t), Tuple(length.(ranges)),
                      Sliced(node.src, [node.ranges[d][ranges[d]] for d in 1:ndims(t)]))
    end
    node isa Union{Leaf,Captured} || throw(ArgumentError(
        "only inputs of the trace can be viewed; computed values have a fixed layout"))
    return traced(tr, eltype(t), Tuple(length.(ranges)), Sliced(t.id, collect(ranges)))
end

# ## In-place verbs
#
# Mutation is recorded as a destination: the computed value's tensor becomes a
# graph output declared with the destination buffer's layout and bound to the
# caller's array. The pre-write contents remain readable only by nodes created
# before the write — the graph has no order, so later reads are refused.

function assign!(dest::TracedArray, v::TracedArray)
    tr = sametrace(dest, v)
    tr.nodes[dest.id] isa Union{Leaf,Captured} || throw(ArgumentError(
        "in-place destination must be an input of the trace"))
    tr.nodes[v.id] isa Union{Leaf,Captured,Presented,Constant} && throw(ArgumentError(
        "in-place assignment needs a computed value"))
    dest.dims == v.dims || throw(DimensionMismatch(
        "cannot assign a $(v.dims) value into a $(dest.dims) destination"))
    any(==(dest.id), values(tr.destinations)) &&
        throw(ArgumentError("destination is written more than once"))
    haskey(tr.destinations, v.id) &&
        throw(ArgumentError("value is already assigned to a destination"))
    tr.destinations[v.id] = dest.id
    return v
end

# how a value is stored is the destination's to decide — Base's semantics
# for any assignment. The default is the plain buffer write; extensions
# override on the destination input's example type (a block-scaled
# destination stores by quantization, so `c .= x` IS the quantize).
store!(dest::TracedArray, v::TracedArray) = store!(dest_example(dest), dest, v)
store!(::Any, dest::TracedArray, v::TracedArray) = assign!(dest, v)

function dest_example(dest::TracedArray)
    node = dest.trace.nodes[dest.id]
    return node isa Union{Leaf,Captured} ? input_example(node) : nothing
end

# the 5-arg forms fuse α·A·B + β·C into one graph, the scaling and
# accumulation a pointwise epilogue on the matmul. β = 0 must not read the
# destination (Base leaves its contents unspecified there); a β ≠ 0 read is
# created before the write, which the destination model permits, and the
# epilogue reads the pre-write value. α and β are trace-time constants.
function scaled_assign!(c::TracedArray, v::TracedArray, α::Number, β::Number)
    isone(α) || (v = α .* v)
    iszero(β) || (v = isone(β) ? v .+ c : v .+ β .* c)
    return store!(c, v)
end

# `LinearAlgebra.mul!` on a traced destination is the library verb
# `Stiletto.mul!`; one method per arity — a traced destination claims any
# matrix operands, traced or captured
LinearAlgebra.mul!(c::TracedMatrix, a::AbstractMatrix, b::AbstractMatrix) =
    mul!(c, a, b)
LinearAlgebra.mul!(c::TracedMatrix, a::AbstractMatrix, b::AbstractMatrix,
                   α::Number, β::Number) = mul!(c, a, b, α, β)
LinearAlgebra.mul!(y::TracedVector, a::AbstractMatrix, x::AbstractVector) =
    mul!(y, a, x)
LinearAlgebra.mul!(y::TracedVector, a::AbstractMatrix, x::AbstractVector,
                   α::Number, β::Number) = mul!(y, a, x, α, β)

# `y .= v` lowers to an identity broadcast; unwrap it so already-computed
# values (including permuted outputs) assign directly instead of copying
function Base.Broadcast.materialize!(dest::TracedArray, bc::Broadcast.Broadcasted)
    bc.f === identity && length(bc.args) == 1 && bc.args[1] isa TracedArray &&
        return Base.Broadcast.materialize!(dest, bc.args[1])
    return store!(dest, walkbc(dest.trace, bc))
end
# y .= t for an already-computed value assigns it directly, no copy node
function Base.Broadcast.materialize!(dest::TracedArray, v::TracedArray)
    computed = !(v.trace.nodes[v.id] isa Union{Leaf,Captured,Presented,Constant})
    return store!(dest, computed ? v : pointwise(identity, v))
end

# ## Reductions
#
# Reduced dimensions are kept as singletons, cuDNN-style; `sum(t)` therefore
# returns an all-singleton TracedArray rather than a scalar.

function reduce_traced(mode::Symbol, t::TracedArray; dims)
    ds = dims === Colon() ? collect(1:ndims(t)) :
         collect(Int, dims isa Integer ? (dims,) : dims)
    all(d -> 1 <= d <= ndims(t), ds) ||
        throw(ArgumentError("reduction dims $dims out of range for $(ndims(t)) dimensions"))
    outdims = ntuple(i -> i in ds ? 1 : t.dims[i], ndims(t))
    return traced(t.trace, eltype(t), outdims, Reduction(mode, t.id, ds))
end

Base.sum(t::TracedArray; dims=:) = reduce_traced(:add, t; dims)
Base.prod(t::TracedArray; dims=:) = reduce_traced(:mul, t; dims)
Base.maximum(t::TracedArray; dims=:) = reduce_traced(:max, t; dims)
Base.minimum(t::TracedArray; dims=:) = reduce_traced(:min, t; dims)
Statistics.mean(t::TracedArray; dims=:) = reduce_traced(:avg, t; dims)
Statistics.mean(f, t::TracedArray; dims=:) = reduce_traced(:avg, pointwise(f, t); dims)
