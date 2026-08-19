# ## mul!
#
# `LinearAlgebra.mul!` under Stiletto's own name, in the same two dispatch
# tiers as `batched_mul!`. Matrix-matrix and matrix-vector, matching Base —
# batching is spelled `batched_mul!`. Owning the name makes the eager tier
# possible:
# extending `LinearAlgebra.mul!` on plain device arrays would pirate the
# BLAS methods, so the cuDNN spelling is opt-in by qualification.

"""
    Stiletto.mul!(C, A, B[, α, β])
    Stiletto.mul!(y, A, x[, α, β])

`LinearAlgebra.mul!` executed as a cuDNN graph: `C = α·A·B + β·C` with the
scaling and accumulation fused as a matmul epilogue; `β = 0` leaves the
destination unread. Matrix-matrix and matrix-vector, matching Base;
batching is spelled [`batched_mul!`](@ref Stiletto.batched_mul).

On traced values this is what `LinearAlgebra.mul!` dispatches to, so both
spellings trace identically. On plain arrays it jit-compiles the graph —
the eager `mul!` for operands no BLAS method claims, such as block-scaled
composites. `α` and `β` are trace-time constants — each coefficient pair
compiles its own plan.
"""
function mul!(c::TracedMatrix, a::AbstractMatrix, b::AbstractMatrix,
              α::Number=true, β::Number=false)
    tr = c.trace
    return scaled_assign!(c, matmul(a isa TracedArray ? a : capture(tr, a),
                                    b isa TracedArray ? b : capture(tr, b)), α, β)
end

# gemv is trace-level bookkeeping over the same Matmul node: rank lifting
# declares the (K,) operand and the (M,) product as K×1 and M×1, so no
# reshape node is needed — which also keeps computed vectors acceptable
function mul!(y::TracedVector, a::AbstractMatrix, x::AbstractVector,
              α::Number=true, β::Number=false)
    tr = y.trace
    at = a isa TracedArray ? a : capture(tr, a)
    xt = x isa TracedArray ? x : capture(tr, x)
    size(at, 2) == length(xt) || throw(DimensionMismatch(
        "gemv inner dimensions do not match: $(size(at)) × ($(length(xt)),)"))
    v = traced(tr, promote_type(eltype(at), eltype(xt)), (size(at, 1),),
               Matmul(at.id, xt.id))
    return scaled_assign!(y, v, α, β)
end

# ## Eager tier

# α/β ride the closure: isbits captures key the plan cache by value, so each
# coefficient pair is its own specialized graph
function mul!(c::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix,
              α::Number=true, β::Number=false)
    jit((c, a, b) -> (mul!(c, a, b, α, β); nothing), c, a, b)
    return c
end

function mul!(y::AbstractVector, a::AbstractMatrix, x::AbstractVector,
              α::Number=true, β::Number=false)
    jit((y, a, x) -> (mul!(y, a, x, α, β); nothing), y, a, x)
    return y
end
