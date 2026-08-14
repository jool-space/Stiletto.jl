# ## Attention
#
# Scaled dot-product attention as a named function over cuDNN's unified SDPA
# op, in the usual tiers. SDPA is a mega-op with its own operation-graph
# mode: it fuses internally (flash-style, no s×s materialization) but not
# with surrounding ops, so attention typically stands alone in its trace —
# a graph mixing it with other nodes builds, and engine support decides.

"""
    Stiletto.attention(q, k, v; scale=1/√size(q, 1), causal=false,
                       seq_len_q=nothing, seq_len_kv=nothing)
    Stiletto.attention!(o, q, k, v; ...)

Scaled dot-product attention `softmax(scale ⋅ qᵀk) ⋅ v` over rank-4 operands
laid out `(head_dim, heads, seq_len, batch)`. Grouped-query attention gives
`k`/`v` fewer heads (a divisor of `q`'s). `scale` binds by value at
execution; `causal` bakes a causal mask into the graph.

`seq_len_q`/`seq_len_kv` (passed together) are `Int32` vectors of per-batch
valid lengths — runtime inputs masking each sequence's tail, so one graph
built at maximum length serves ragged batches by rebinding lengths per call.
Output rows past `seq_len_q` are undefined.
"""
function attention(q::TracedArray, k, v;
                   scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false,
                   seq_len_q=nothing, seq_len_kv=nothing)
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
    (seq_len_q === nothing) == (seq_len_kv === nothing) || throw(ArgumentError(
        "seq_len_q and seq_len_kv must be passed together"))
    lens = map((seq_len_q, seq_len_kv)) do sl
        sl === nothing && return nothing
        t = sl isa TracedArray ? sl : capture(tr, sl)
        sametrace(q, t)
        ndims(t) == 1 && length(t) == b || throw(DimensionMismatch(
            "sequence lengths are per-batch vectors of length $b, got $(size(t))"))
        eltype(t) == Int32 || throw(ArgumentError(
            "sequence lengths must be Int32; convert explicitly"))
        return reshape(t, (1, 1, 1, b)).id
    end
    s = traced(tr, Float32, (), Constant(Float32(scale)))
    return traced(tr, eltype(q), q.dims, Sdpa(q.id, k.id, v.id, s.id, causal, lens...))
end

attention!(o::TracedArray, q::TracedArray, k, v; kwargs...) =
    assign!(o, attention(q, k, v; kwargs...))

# ## Eager tier

attention(q::AbstractArray, k, v; kwargs...) = attention!(similar(q), q, k, v; kwargs...)

function attention!(o::AbstractArray, q::AbstractArray, k, v;
                    scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false,
                    seq_len_q=nothing, seq_len_kv=nothing)
    (seq_len_q === nothing) == (seq_len_kv === nothing) || throw(ArgumentError(
        "seq_len_q and seq_len_kv must be passed together"))
    s = Float32(scale)
    if seq_len_q === nothing
        jit((o, q, k, v) -> (attention!(o, q, k, v; scale=s, causal); nothing), o, q, k, v)
    else  # lengths are runtime inputs: one cached plan serves all bindings
        jit((o, q, k, v, lq, lkv) ->
                (attention!(o, q, k, v; scale=s, causal, seq_len_q=lq, seq_len_kv=lkv);
                 nothing),
            o, q, k, v, seq_len_q, seq_len_kv)
    end
    return o
end
