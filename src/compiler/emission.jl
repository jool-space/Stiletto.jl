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
    o = sdpa_fwd!(g, tf(node.q), tf(node.k), tf(node.v); o=dest_tensor(tf, i),
                  scale=tf(node.scale), causal=node.causal,
                  seq_len_q=maybe(tf, node.seq_len_q),
                  seq_len_kv=maybe(tf, node.seq_len_kv))
    # the causal mask's fill value is a runtime input of the mega-op, not a
    # node of the trace; bind it here
    node.causal && (tf.extra[g.ops[end].mask_subgraph.fill] = -Inf32)
    return o
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
