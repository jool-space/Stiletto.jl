# ## Attention
#
# Scaled dot-product attention as a named function over cuDNN's unified SDPA
# op, in the usual tiers. SDPA is a mega-op with its own operation-graph
# mode: it fuses internally (flash-style, no s×s materialization) but not
# with surrounding ops, so attention typically stands alone in its trace —
# a graph mixing it with other nodes builds, and engine support decides.

"""
    Stiletto.attention(q, k, v; scale=1/√size(q, 1), causal=false)
    Stiletto.attention!(o, q, k, v; scale=..., causal=false)

Scaled dot-product attention `softmax(scale ⋅ qᵀk) ⋅ v` over rank-4 operands
laid out `(head_dim, heads, seq_len, batch)`. Grouped-query attention gives
`k`/`v` fewer heads (a divisor of `q`'s). `scale` binds by value at
execution; `causal` bakes a causal mask into the graph.
"""
function attention(q::TracedArray, k, v;
                   scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false)
    tr = q.trace
    k = k isa TracedArray ? k : capture(tr, k)
    v = v isa TracedArray ? v : capture(tr, v)
    sametrace(q, k, v)
    ndims(q) == ndims(k) == ndims(v) == 4 || throw(ArgumentError(
        "attention operands are rank-4 (head_dim, heads, seq_len, batch)"))
    d, hq, sq, b = size(q)
    size(k, 1) == size(v, 1) == d || throw(DimensionMismatch(
        "attention head dimensions do not match: q $(size(q)), k $(size(k)), v $(size(v))"))
    size(k, 2) == size(v, 2) && hq % size(k, 2) == 0 || throw(DimensionMismatch(
        "attention q heads ($hq) must be a multiple of matching k/v heads " *
        "($(size(k, 2)), $(size(v, 2)))"))
    size(k, 3) == size(v, 3) || throw(DimensionMismatch(
        "attention k/v sequence lengths do not match: k $(size(k)), v $(size(v))"))
    size(k, 4) == size(v, 4) == b || throw(DimensionMismatch(
        "attention batch sizes do not match: q $(size(q)), k $(size(k)), v $(size(v))"))
    s = traced(tr, Float32, (), Constant(Float32(scale)))
    return traced(tr, eltype(q), q.dims, Sdpa(q.id, k.id, v.id, s.id, causal))
end

attention!(o::TracedArray, q::TracedArray, k, v; kwargs...) =
    assign!(o, attention(q, k, v; kwargs...))

# ## Eager tier

attention(q::AbstractArray, k, v; kwargs...) = attention!(similar(q), q, k, v; kwargs...)

function attention!(o::AbstractArray, q::AbstractArray, k, v;
                    scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false)
    s = Float32(scale)
    jit((o, q, k, v) -> (attention!(o, q, k, v; scale=s, causal); nothing), o, q, k, v)
    return o
end
