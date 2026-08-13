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
# its tensor is a non-virtual output declared with the buffer's layout
function dest_tensor(tf, i)
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
                  scale=tf(node.scale), causal=node.causal)
    # the causal mask's fill value is a runtime input of the mega-op, not a
    # node of the trace; bind it here
    node.causal && (tf.extra[g.ops[end].mask_subgraph.fill] = -Inf32)
    return o
end

emit!(g::Graph, tf, i, node::Conv, R) =
    conv_fprop!(g, tf(node.x), tf(node.w); y=dest_tensor(tf, i),
                pre_padding=node.pre_padding, post_padding=node.post_padding,
                stride=node.stride, dilation=node.dilation, name=nodename(i, node))

function emit!(g::Graph, tf, i, node::Cast, R)
    x = tf(node.x)
    y = dest_tensor(tf, i)
    y === nothing &&
        (y = tensor!(g; dims=x.dims, dtype=node.T, virtual=true, name=nodename(i, node)))
    return pointwise!(g, CUDNN_POINTWISE_IDENTITY, x; y)
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

declare(g::Graph, i, node, ex, R) =
    tensor!(g; dims=lift_dims(size(ex), R), strides=lift_strides(ex, R),
            dtype=eltype(ex), name=nodename(i, node))

nodename(i, node::Leaf) = "arg$(node.argindex)"
nodename(i, node::Captured) = "capture$i"
nodename(i, node::Constant) = "const$i"
nodename(i, node::Presented) = "perm$i"
nodename(i, node::Reshaped) = "reshape$i"
nodename(i, node) = "node$i"
