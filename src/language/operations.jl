# ## Matmul
#
# Julia dimension order with the batch trailing, matching cuDNN.matmul!:
# (M, K[, B]) × (K, N[, B]) → (M, N[, B]), batch extents broadcastable.

function matmul(a::TracedArray, b::TracedArray)
    tr = sametrace(a, b)
    2 <= ndims(a) <= 3 && 2 <= ndims(b) <= 3 ||
        throw(ArgumentError("traced matmul takes 2- or 3-dimensional operands"))
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "matmul inner dimensions do not match: $(size(a)) × $(size(b))"))
    batch = max(size(a, 3), size(b, 3))
    size(a, 3) in (1, batch) && size(b, 3) in (1, batch) || throw(DimensionMismatch(
        "matmul batch dimensions are not broadcastable: $(size(a)) × $(size(b))"))
    T = promote_type(eltype(a), eltype(b))
    dims = max(ndims(a), ndims(b)) == 2 ? (size(a, 1), size(b, 2)) :
                                          (size(a, 1), size(b, 2), batch)
    return traced(tr, T, dims, Matmul(a.id, b.id))
end

const TracedMatOrBatch = Union{TracedArray{<:Any,2},TracedArray{<:Any,3}}
Base.:*(a::TracedMatOrBatch, b::TracedMatOrBatch) = matmul(a, b)
Base.:*(a::TracedMatOrBatch, b::AbstractMatrix) = matmul(a, capture(a.trace, b))
Base.:*(a::AbstractMatrix, b::TracedMatOrBatch) = matmul(capture(b.trace, a), b)
Base.:*(a::TracedMatOrBatch, b::AbstractArray{<:Any,3}) = matmul(a, capture(a.trace, b))
Base.:*(a::AbstractArray{<:Any,3}, b::TracedMatOrBatch) = matmul(capture(b.trace, a), b)
# disambiguators against LinearAlgebra's wrapper-specialized *
const LazyWrapped = Union{Adjoint{<:Any,<:AbstractMatrix},Transpose{<:Any,<:AbstractMatrix}}
Base.:*(a::LazyWrapped, b::TracedMatOrBatch) = matmul(capture(b.trace, a), b)
Base.:*(a::TracedMatOrBatch, b::LazyWrapped) = matmul(a, capture(a.trace, b))
Base.:*(a::Adjoint{<:Any,<:AbstractVector}, b::TracedMatOrBatch) = matmul(capture(b.trace, a), b)
Base.:*(a::Transpose{<:Any,<:AbstractVector}, b::TracedMatOrBatch) = matmul(capture(b.trace, a), b)

# ## Presentation
#
# Permutations are free stride changes, but only on inputs: a computed
# (virtual) tensor's layout belongs to the fused kernel.

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
    end
    throw(ArgumentError(
        "only inputs of the trace can be permuted; computed values have a fixed layout"))
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

LinearAlgebra.mul!(c::TracedArray, a::TracedArray, b::TracedArray) = assign!(c, matmul(a, b))
LinearAlgebra.mul!(c::TracedArray, a::AbstractArray, b::TracedArray) =
    assign!(c, matmul(capture(c.trace, a), b))
LinearAlgebra.mul!(c::TracedArray, a::TracedArray, b::AbstractArray) =
    assign!(c, matmul(a, capture(c.trace, b)))

Base.Broadcast.materialize!(dest::TracedArray, bc::Broadcast.Broadcasted) =
    assign!(dest, walkbc(dest.trace, bc))
# y .= t for an already-computed value assigns it directly, no copy node
function Base.Broadcast.materialize!(dest::TracedArray, v::TracedArray)
    computed = !(v.trace.nodes[v.id] isa Union{Leaf,Captured,Presented,Constant})
    return assign!(dest, computed ? v : pointwise(identity, v))
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
