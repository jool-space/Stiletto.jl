# ## Attention
#
# Scaled dot-product attention as a named function over cuDNN's unified SDPA
# op, in the usual tiers. SDPA is a mega-op with its own operation-graph
# mode: it fuses internally (flash-style, no s×s materialization) but not
# with surrounding ops, so attention typically stands alone in its trace —
# a graph mixing it with other nodes builds, and engine support decides.
#
# The backward is cuDNN's composite sdpa_bwd! (the spelling the fMHA
# backward engines pattern-match; the unified SDPA_BWD op has no engines
# anywhere). It consumes the forward's output and softmax LSE stats, so
# training runs the forward with `stats=true`.

"""
    Stiletto.attention(q, k, v; stats=false, scale=1/√size(q, 1), causal=false,
                       seq_len_q=nothing, seq_len_kv=nothing)
    Stiletto.attention!(o, q, k, v; ...)

Scaled dot-product attention `softmax(scale ⋅ qᵀk) ⋅ v` over rank-4 operands
laid out `(head_dim, heads, seq_len, batch)`. Grouped-query attention gives
`k`/`v` fewer heads (a divisor of `q`'s). `scale` binds by value at
execution; `causal` bakes a causal mask into the graph.

`stats=true` additionally returns the softmax log-sum-exp as a
`(1, seq_len, heads, batch)` Float32 array — the saved state
[`attention_backward`](@ref Stiletto.attention_backward) consumes.

`seq_len_q`/`seq_len_kv` (passed together) are `Int32` vectors of per-batch
valid lengths — runtime inputs masking each sequence's tail, so one graph
built at maximum length serves ragged batches by rebinding lengths per call.
Output rows past `seq_len_q` are undefined.
"""
function attention(q::TracedArray, k, v; stats::Bool=false,
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
    lens = sdpa_lens(tr, q, b, seq_len_q, seq_len_kv)
    s = traced(tr, Float32, (), Constant(Float32(scale)))
    o = traced(tr, eltype(q), q.dims, Sdpa(q.id, k.id, v.id, s.id, causal, stats, lens...))
    stats || return o
    st = traced(tr, Float32, (1, sq, hq, b), Aux(o.id, :stats))
    return o, st
end

# per-batch valid lengths: Int32 vectors, reshaped in-trace to the (1,1,1,b)
# tensors the SDPA ops want
function sdpa_lens(tr, anchor, b, seq_len_q, seq_len_kv)
    (seq_len_q === nothing) == (seq_len_kv === nothing) || throw(ArgumentError(
        "seq_len_q and seq_len_kv must be passed together"))
    return map((seq_len_q, seq_len_kv)) do sl
        sl === nothing && return nothing
        t = sl isa TracedArray ? sl : capture(tr, sl)
        sametrace(anchor, t)
        ndims(t) == 1 && length(t) == b || throw(DimensionMismatch(
            "sequence lengths are per-batch vectors of length $b, got $(size(t))"))
        eltype(t) == Int32 || throw(ArgumentError(
            "sequence lengths must be Int32; convert explicitly"))
        return reshape(t, (1, 1, 1, b)).id
    end
end

attention!(o::TracedArray, q::TracedArray, k, v; kwargs...) =
    assign!(o, attention(q, k, v; kwargs...))

"""
    Stiletto.attention_backward(dO, q, k, v, o, stats; scale=1/√size(q, 1),
                                causal=false, seq_len_q=nothing,
                                seq_len_kv=nothing) -> (dQ, dK, dV)
    Stiletto.attention_backward!(dQ, dK, dV, dO, q, k, v, o, stats; ...)

Gradients of [`attention`](@ref Stiletto.attention) with respect to `q`, `k`,
and `v`, from the forward pass's output `o` and saved softmax `stats`
(`attention(...; stats=true)`). `scale`/`causal`/sequence lengths must match
the forward call. The operands cannot feed other operations in the same
trace — the composite pattern owns its graph.
"""
function attention_backward(dO::TracedArray, q, k, v, o, stats;
                            scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false,
                            seq_len_q=nothing, seq_len_kv=nothing)
    tr = dO.trace
    cap(x) = x isa TracedArray ? x : capture(tr, x)
    qv, kv, vv, ov, st = cap(q), cap(k), cap(v), cap(o), cap(stats)
    sametrace(dO, qv, kv, vv, ov, st)
    d, hq, sq, b = size(qv)
    size(dO) == size(qv) == size(ov) || throw(DimensionMismatch(
        "dO and o must have q's dimensions: q $(size(qv)), o $(size(ov)), dO $(size(dO))"))
    size(st) == (1, sq, hq, b) || throw(DimensionMismatch(
        "stats dimensions must be $((1, sq, hq, b)), got $(size(st))"))
    lens = sdpa_lens(tr, dO, b, seq_len_q, seq_len_kv)
    s = traced(tr, Float32, (), Constant(Float32(scale)))
    dQ = traced(tr, eltype(dO), qv.dims,
                SdpaBwd(dO.id, qv.id, kv.id, vv.id, ov.id, st.id, s.id, causal, lens...))
    dK = traced(tr, eltype(dO), kv.dims, Aux(dQ.id, :dk))
    dV = traced(tr, eltype(dO), vv.dims, Aux(dQ.id, :dv))
    return dQ, dK, dV
end

function attention_backward!(dQ::TracedArray, dK::TracedArray, dV::TracedArray,
                             dO::TracedArray, q, k, v, o, stats; kwargs...)
    q1, k1, v1 = attention_backward(dO, q, k, v, o, stats; kwargs...)
    return assign!(dQ, q1), assign!(dK, k1), assign!(dV, v1)
end

# ## Eager tier

function attention(q::AbstractArray, k, v; stats::Bool=false,
                   scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false,
                   seq_len_q=nothing, seq_len_kv=nothing)
    stats || return attention!(similar(q), q, k, v; scale, causal, seq_len_q, seq_len_kv)
    (seq_len_q === nothing) == (seq_len_kv === nothing) || throw(ArgumentError(
        "seq_len_q and seq_len_kv must be passed together"))
    _, hq, sq, b = size(q)
    o, st = similar(q), similar(q, Float32, (1, sq, hq, b))
    s = Float32(scale)
    if seq_len_q === nothing
        jit((o, st, q, k, v) ->
                (attention_fwd_stats!(o, st, q, k, v; scale=s, causal); nothing),
            o, st, q, k, v)
    else
        jit((o, st, q, k, v, lq, lkv) ->
                (attention_fwd_stats!(o, st, q, k, v; scale=s, causal,
                                      seq_len_q=lq, seq_len_kv=lkv); nothing),
            o, st, q, k, v, seq_len_q, seq_len_kv)
    end
    return o, st
end

# traced helper for the stats-producing eager form
function attention_fwd_stats!(o::TracedArray, st::TracedArray, q::TracedArray, k, v;
                              kwargs...)
    y, stats = attention(q, k, v; stats=true, kwargs...)
    return assign!(o, y), assign!(st, stats)
end

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

function attention_backward(dO::AbstractArray, q, k, v, o, stats; kwargs...)
    return attention_backward!(similar(q), similar(k), similar(v),
                               dO, q, k, v, o, stats; kwargs...)
end

function attention_backward!(dQ::AbstractArray, dK::AbstractArray, dV::AbstractArray,
                             dO::AbstractArray, q, k, v, o, stats;
                             scale::Real=inv(sqrt(size(q, 1))), causal::Bool=false,
                             seq_len_q=nothing, seq_len_kv=nothing)
    s = Float32(scale)
    if seq_len_q === nothing
        jit((dQ, dK, dV, dO, q, k, v, o, st) ->
                (attention_backward!(dQ, dK, dV, dO, q, k, v, o, st; scale=s, causal);
                 nothing),
            dQ, dK, dV, dO, q, k, v, o, stats)
    else
        jit((dQ, dK, dV, dO, q, k, v, o, st, lq, lkv) ->
                (attention_backward!(dQ, dK, dV, dO, q, k, v, o, st; scale=s, causal,
                                     seq_len_q=lq, seq_len_kv=lkv);
                 nothing),
            dQ, dK, dV, dO, q, k, v, o, stats, seq_len_q, seq_len_kv)
    end
    return dQ, dK, dV
end
