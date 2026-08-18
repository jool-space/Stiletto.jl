# ## Batched matmul
#
# The vendored NNlib verb grown torch-style: any number of trailing batch
# dimensions, broadcasting pairwise. cuDNN's matmul op consumes the batch
# dims directly — no flattening into one batch axis and no materialized
# expansion; a broadcast operand is read through its singleton extent.

"""
    Stiletto.batched_mul(a, b)
    Stiletto.batched_mul!(c, a, b[, α, β])
    a ⊠ b

Batched matrix multiplication `(M, K, batch...) × (K, N, batch...) →
(M, N, batch...)`. Trailing batch extents broadcast pairwise: each pair must
match or be 1 — missing dimensions count as 1 — and the result takes the
larger. Unlike `NNlib.batched_mul`, any number of batch dimensions is
accepted. `⊠` (`\\boxtimes`) is the infix spelling.

The 5-arg form computes `α·A·B + β·C` with the scaling and accumulation
fused as a matmul epilogue; `β = 0` leaves the destination unread. `α` and
`β` are trace-time constants — each pair compiles its own plan.
"""
batched_mul(a::TracedArray, b::TracedArray) = matmul(a, b)
batched_mul(a::TracedArray, b::AbstractArray) = matmul(a, capture(a.trace, b))
batched_mul(a::AbstractArray, b::TracedArray) = matmul(capture(b.trace, a), b)

function batched_mul!(c::TracedArray, a, b, α::Number=true, β::Number=false)
    tr = c.trace
    return scaled_assign!(c, matmul(a isa TracedArray ? a : capture(tr, a),
                                    b isa TracedArray ? b : capture(tr, b)), α, β)
end

# ## Eager tier

function batched_mul(a::AbstractArray, b::AbstractArray)
    dims = (size(a, 1), size(b, 2),
            ntuple(i -> max(size(a, i + 2), size(b, i + 2)),
                   max(ndims(a), ndims(b)) - 2)...)
    c = similar(a, promote_type(eltype(a), eltype(b)), dims)
    return batched_mul!(c, a, b)
end

# α/β ride the closure: isbits captures key the plan cache by value, so each
# coefficient pair is its own specialized graph
function batched_mul!(c::AbstractArray, a, b, α::Number=true, β::Number=false)
    jit((c, a, b) -> (batched_mul!(c, a, b, α, β); nothing), c, a, b)
    return c
end

const ⊠ = batched_mul
