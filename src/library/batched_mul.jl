# ## Batched matmul
#
# The vendored NNlib verb grown torch-style: any number of trailing batch
# dimensions, broadcasting pairwise. cuDNN's matmul op consumes the batch
# dims directly — no flattening into one batch axis and no materialized
# expansion; a broadcast operand is read through its singleton extent.

"""
    Stiletto.batched_mul(a, b)
    Stiletto.batched_mul!(c, a, b)
    a ⊠ b

Batched matrix multiplication `(M, K, batch...) × (K, N, batch...) →
(M, N, batch...)`. Trailing batch extents broadcast pairwise: each pair must
match or be 1 — missing dimensions count as 1 — and the result takes the
larger. Unlike `NNlib.batched_mul`, any number of batch dimensions is
accepted. `⊠` (`\\boxtimes`) is the infix spelling.
"""
batched_mul(a::TracedArray, b::TracedArray) = matmul(a, b)
batched_mul(a::TracedArray, b::AbstractArray) = matmul(a, capture(a.trace, b))
batched_mul(a::AbstractArray, b::TracedArray) = matmul(capture(b.trace, a), b)

function batched_mul!(c::TracedArray, a, b)
    tr = c.trace
    return assign!(c, matmul(a isa TracedArray ? a : capture(tr, a),
                             b isa TracedArray ? b : capture(tr, b)))
end

# ## Eager tier

function batched_mul(a::AbstractArray, b::AbstractArray)
    dims = (size(a, 1), size(b, 2),
            ntuple(i -> max(size(a, i + 2), size(b, i + 2)),
                   max(ndims(a), ndims(b)) - 2)...)
    c = similar(a, promote_type(eltype(a), eltype(b)), dims)
    return batched_mul!(c, a, b)
end

function batched_mul!(c::AbstractArray, a, b)
    jit((c, a, b) -> (batched_mul!(c, a, b); nothing), c, a, b)
    return c
end

const ⊠ = batched_mul
