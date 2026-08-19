# ## Emission
#
# One rank for the whole graph: the maximum any input has or any operation
# requires, floored at 3 — cuDNN engines expect tensors of at least rank 3.
# Tensors are declared lazily by depth-first walk from the outputs, so unused
# nodes never become graph tensors.

function graph_rank(tr::Trace)
    R = 3
    for node in tr.nodes
        node isa Union{Leaf,Captured} && (R = max(R, ndims(input_example(node))))
        node isa Reshaped && (R = max(R, length(node.dims)))
    end
    return R
end

lift_dims(dims, R) = Int64[collect(Int64, dims); ones(Int64, R - length(dims))]
function lift_strides(a, R)
    s = collect(Int64, strides(a))
    return [s; fill(Int64(max(length(a), 1)), R - length(s))]
end

# a node with a recorded destination writes straight into that input's buffer:
# its tensor is a non-virtual output declared with the buffer's layout; a
# pre-built destination (permuted output) takes precedence
function dest_tensor(tf, i)
    t = get(tf.dests, i, nothing)
    t === nothing || return t
    leafid = get(tf.trace.destinations, i, nothing)
    leafid === nothing && return nothing
    ex = input_example(tf.trace.nodes[leafid])
    return tensor!(tf.g; dims=lift_dims(size(ex), tf.rank),
                   strides=lift_strides(ex, tf.rank), dtype=eltype(ex),
                   output=true, name="dest$i")
end

function emit!(g::Graph, tf, i, node::Union{Leaf,Captured}, R)
    ex = input_example(node)
    ex isa Number && return scalar!(g, Float32; rank=R, name=nodename(i, node))
    return declare(g, i, node, ex, R)
end
emit!(g::Graph, tf, i, node::Constant, R) = scalar!(g, Float32; rank=R, name=nodename(i, node))
emit!(g::Graph, tf, i, node::Matmul, R) =
    matmul!(g, tf(node.a), tf(node.b); c=dest_tensor(tf, i), name=nodename(i, node))
function emit!(g::Graph, tf, i, node::Pointwise, R)
    args = map(tf, node.args)
    if boolean_output(node.mode)
        dims = [maximum(t.dims[d] for t in args) for d in 1:R]
        y = tensor!(g; dims, dtype=CUDNN_DATA_BOOLEAN, virtual=true, name=nodename(i, node))
        return pointwise!(g, node.mode, args...; y)
    end
    return pointwise!(g, node.mode, args...; y=dest_tensor(tf, i), name=nodename(i, node))
end
emit!(g::Graph, tf, i, node::Reduction, R) =
    reduction!(g, node.mode, tf(node.x); y=dest_tensor(tf, i), dims=node.dims,
               name=nodename(i, node))
maybe(tf, id) = id === nothing ? nothing : tf(id)
emit!(g::Graph, tf, i, node::Norm, R) =
    norm_fwd!(g, tf(node.x), tf(node.scale), maybe(tf, node.bias);
              y=dest_tensor(tf, i), mode=node.mode, phase=:inference,
              mean=maybe(tf, node.mean), inv_variance=maybe(tf, node.inv_variance),
              epsilon=maybe(tf, node.epsilon), name=nodename(i, node))

function emit!(g::Graph, tf, i, node::Sdpa, R)
    q = tf(node.q)
    stats = nothing
    if node.stats
        # the (1, s, h, b)-packed buffer the backward requires, declared on
        # the forward's (1, h, s, b) dims through strides at no cost
        _, hq, sq, b = Int.(q.dims)
        stats = tensor!(g; dims=lift_dims((1, hq, sq, b), R),
                        strides=[Int64[1, sq, 1, sq * hq];
                                 fill(Int64(sq * hq * b), R - 4)],
                        dtype=Float32, output=true, name="stats$i")
    end
    r = sdpa_fwd!(g, q, tf(node.k), tf(node.v); o=dest_tensor(tf, i), stats,
                  scale=tf(node.scale), causal=node.causal,
                  seq_len_q=maybe(tf, node.seq_len_q),
                  seq_len_kv=maybe(tf, node.seq_len_kv))
    # the causal mask's fill value is a runtime input of the mega-op, not a
    # node of the trace; bind it here
    node.causal && (tf.extra[g.ops[end].mask_subgraph.fill] = -Inf32)
    node.stats || return r
    o, st = r
    tf.aux[(i, :stats)] = st
    return o
end

# the composite backward owns its graph: operands get backward's (d, s, h, b)
# declarations presenting the canonical (d, h, s, b) storage — same memory,
# different dims order — so they cannot also be declared by other uses
function emit!(g::Graph, tf, i, node::SdpaBwd, R)
    R == 4 || throw(ArgumentError(
        "attention_backward requires a rank-4 graph; it cannot share a trace " *
        "with higher-rank values"))
    function io_tensor(id, name)
        tf.tensors[id] === nothing || throw(ArgumentError(
            "attention_backward operands cannot feed other operations in the same trace"))
        n = tf.trace.nodes[id]
        n isa Union{Leaf,Captured} || throw(ArgumentError(
            "attention_backward operands must be trace inputs, not computed or " *
            "re-presented values"))
        ex = input_example(n)
        d, h, s, b = size(ex)
        t = tensor!(tf.g; dims=(d, s, h, b), strides=(1, d * h, d, d * h * s),
                    dtype=eltype(ex), name)
        tf.tensors[id] = t
        return t
    end
    q = io_tensor(node.q, "Q")
    k = io_tensor(node.k, "K")
    v = io_tensor(node.v, "V")
    o = io_tensor(node.o, "O")
    dO = io_tensor(node.dO, "dO")
    stats = tf(node.stats)   # (1, s, h, b) dense: already backward's packing

    ex = input_example(tf.trace.nodes[node.q])
    T = eltype(ex)
    d, h, sq, b = size(ex)
    kex = input_example(tf.trace.nodes[node.k])
    _, hk, skv, _ = size(kex)
    out(name, s, nh) = tensor!(g; dims=(d, s, nh, b),
                               strides=(1, d * nh, d, d * nh * s),
                               dtype=T, output=true, name)
    tdq = out("dQ", sq, h)
    tdk = out("dK", skv, hk)
    tdv = out("dV", skv, hk)

    _, _, _, softmax_sum, dQ_accum, dKf, dVf =
        sdpa_bwd!(g, q, k, v, o, dO, stats; dQ=tdq, dK=tdk, dV=tdv,
                  scale=tf(node.scale), causal=node.causal,
                  seq_len_q=maybe(tf, node.seq_len_q),
                  seq_len_kv=maybe(tf, node.seq_len_kv))

    # fp32 spill buffers and grouped-head workspaces the pattern requires:
    # allocated once at emission, pinned for the plan's lifetime like its
    # cuDNN workspace; the ×1 scalar fills the pattern's dropout-scale slot
    tf.extra[softmax_sum] = CUDACore.zeros(Float32, 1, sq, h, b)
    tf.extra[dQ_accum] = CUDACore.zeros(Float32, d, sq, h, b)
    dKf === nothing || (tf.extra[dKf] = CUDACore.zeros(T, d, skv, h, b))
    dVf === nothing || (tf.extra[dVf] = CUDACore.zeros(T, d, skv, h, b))
    tf.extra[cuDNN.tensor(g, "One")] = 1f0
    (node.causal || node.seq_len_q !== nothing) &&
        (tf.extra[cuDNN.tensor(g, "MaskValue")] = -Inf32)

    tf.aux[(i, :dk)] = tdk
    tf.aux[(i, :dv)] = tdv
    return tdq
end

emit!(g::Graph, tf, i, node::Conv, R) =
    conv_fprop!(g, tf(node.x), tf(node.w); y=dest_tensor(tf, i),
                pre_padding=node.pre_padding, post_padding=node.post_padding,
                stride=node.stride, dilation=node.dilation, name=nodename(i, node))

emit!(g::Graph, tf, i, node::Resample, R) =
    resample_fwd!(g, tf(node.x); y=dest_tensor(tf, i), mode=node.mode,
                  window=node.window, pre_padding=node.pre_padding,
                  post_padding=node.post_padding, stride=node.stride,
                  name=nodename(i, node))

# projections force their op (memoized) and fetch the auxiliary tensor the
# op's emission recorded
function emit!(g::Graph, tf, i, node::Aux, R)
    tf(node.src)
    t = get(tf.aux, (node.src, node.which), nothing)
    t === nothing && throw(ArgumentError(
        "node $(node.src) provides no auxiliary result $(node.which)"))
    return t
end

function emit!(g::Graph, tf, i, node::Cast, R)
    x = tf(node.x)
    y = dest_tensor(tf, i)
    y === nothing &&
        (y = tensor!(g; dims=x.dims, dtype=node.T, virtual=true, name=nodename(i, node)))
    return pointwise!(g, CUDNN_POINTWISE_IDENTITY, x; y)
end

# a permuted computed value materializes at the output boundary: the source
# op's output tensor is declared non-virtual with strides that lay the buffer
# out dense in permuted axis order (or with the destination's layout permuted)
function emit!(g::Graph, tf, i, node::Permuted, R)
    tf.tensors[node.src] === nothing || throw(ArgumentError(
        "permutedims of a computed value must be its only use"))
    haskey(tf.trace.destinations, node.src) && throw(ArgumentError(
        "a value written in place cannot also be returned in permuted layout"))
    perm = [node.perm; length(node.perm)+1:R]
    sdims = lift_dims(node.dims, R)
    leafid = get(tf.trace.destinations, i, nothing)
    st = Vector{Int64}(undef, R)
    if leafid === nothing
        st[perm] = cumprod([Int64(1); sdims[perm][1:end-1]])
        t = tensor!(g; dims=sdims, strides=st, output=true, name=nodename(i, node))
    else
        ex = input_example(tf.trace.nodes[leafid])
        st[perm] = lift_strides(ex, R)
        t = tensor!(g; dims=sdims, strides=st, dtype=eltype(ex), output=true,
                    name=nodename(i, node))
    end
    tf.dests[node.src] = t
    tf(node.src) === t || throw(ArgumentError(
        "this operation cannot materialize a permuted output"))
    return t
end

function emit!(g::Graph, tf, i, node::Reshaped, R)
    ex = input_example(tf.trace.nodes[node.src])
    return tensor!(g; dims=lift_dims(node.dims, R), dtype=eltype(ex),
                   name=nodename(i, node))
end

function emit!(g::Graph, tf, i, node::Presented, R)
    ex = PermutedDimsArray(input_example(tf.trace.nodes[node.src]), Tuple(node.perm))
    return declare(g, i, node, ex, R)
end

emit!(g::Graph, tf, i, node::Sliced, R) =
    declare(g, i, node, view(input_example(tf.trace.nodes[node.src]), node.ranges...), R)

declare(g::Graph, i, node, ex, R) =
    tensor!(g; dims=lift_dims(size(ex), R), strides=lift_strides(ex, R),
            dtype=eltype(ex), alignment=array_alignment(ex), name=nodename(i, node))

# a view's pointer is offset from its parent's allocation, so the usual
# 16-byte assumption may not hold; declare what the example actually has
array_alignment(ex) = 16
array_alignment(ex::SubArray) = min(16, 1 << trailing_zeros(UInt(view_pointer(ex))))

# offset pointer computed from the parent, which every array type can produce
view_pointer(a::SubArray) =
    pointer(parent(a), LinearIndices(parent(a))[map(first, parentindices(a))...])

nodename(i, node::Leaf) = "arg$(node.argindex)"
nodename(i, node::Captured) = "capture$i"
nodename(i, node::Constant) = "const$i"
nodename(i, node::Presented) = "perm$i"
nodename(i, node::Permuted) = "permout$i"
nodename(i, node::Reshaped) = "reshape$i"
nodename(i, node::Sliced) = "view$i"
nodename(i, node) = "node$i"
