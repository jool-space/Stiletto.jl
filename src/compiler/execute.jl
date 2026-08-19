# ## Execution

# declarations carry the presentation (permuted strides); bindings take the
# dense storage behind it, which the layout check accepts as the same layout
bindable(a) = a
bindable(a::Union{PermutedDimsArray,Transpose,Adjoint}) = bindable(parent(a))

# strided views bind by pointer: the declaration carries the view's strides
# and the parent's storage backs it at an offset. cuDNN's dense layout
# comparison cannot express that, so route through its public extension
# point with a wrapper that owns the check.
struct ViewBinding{V}
    v::V
end
bindable(a::SubArray) = ViewBinding(a)

function cuDNN.checked_array_pointer(t::Tensor, b::ViewBinding)
    a = b.v
    cuDNN.cudnnDataType(eltype(a)) == t.dtype || throw(ArgumentError(
        "binding for $(t.name) has eltype $(eltype(a)), expected $(t.dtype)"))
    layout(dims, sts) = sort!([(Int64(d), Int64(s)) for (d, s) in zip(dims, sts) if d != 1];
                              by=last)
    layout(size(a), strides(a)) == layout(t.dims, t.strides) || throw(DimensionMismatch(
        "view binding for $(t.name) has size $(size(a)) with strides $(strides(a)), " *
        "which does not lay out the tensor's $(Tuple(t.dims)) with strides $(Tuple(t.strides))"))
    p = view_pointer(a)
    UInt(p) % t.alignment == 0 || throw(ArgumentError(
        "view binding for $(t.name) is less aligned than the graph was built for; " *
        "compile with a view of this alignment"))
    return p
end

# one input usually binds one tensor; extensions override on the array type
# when an argument backs several graph tensors (composite arrays whose
# declaration split them into storage components)
bind!(g::Graph, bindings, t::Tensor, a) = (bindings[t] = a)

"""
    Stiletto.graph(c) -> cuDNN.Graph

The built cuDNN graph behind a compiled callable, for callers that schedule
execution themselves: `execute!(Stiletto.graph(c), b)` with `b` from
[`Stiletto.bindings`](@ref) is exactly what calling `c` does.
"""
graph(c::Compiled) = c.graph

"""
    Stiletto.bindings(c, args...) -> (bindings, outputs)

Prepare the tensor bindings a call of `c` with `args` would execute with:
every graph tensor mapped to its backing (argument storage — reshaped,
viewed, or split into components as the trace declared it — captured
arrays, by-value scalars, and freshly allocated output buffers). `outputs`
are the arrays the call would return, already present in `bindings`.

Together with [`Stiletto.graph`](@ref) this splits `c(args...)` into its
two halves, so execution can be scheduled by the caller — under CUDA graph
capture, with rebinding between launches, or inside a larger execution
scheme.
"""
function bindings(c::Compiled, args...)
    length(args) == c.nargs ||
        throw(ArgumentError("compiled graph takes $(c.nargs) arrays, got $(length(args))"))
    b = copy(c.extra)
    for (i, node) in enumerate(c.trace.nodes)
        t = c.tensors[i]
        t === nothing && continue
        if node isa Leaf
            t.by_value ? (b[t] = args[node.argindex]) :
                         bind!(c.graph, b, t, bindable(args[node.argindex]))
        elseif node isa Captured
            bind!(c.graph, b, t, bindable(node.array))
        elseif node isa Constant
            b[t] = node.value
        elseif node isa Presented
            src = c.trace.nodes[node.src]
            bind!(c.graph, b, t,
                  bindable(src isa Leaf ? args[src.argindex] : src.array))
        elseif node isa Reshaped
            src = c.trace.nodes[node.src]
            arr = bindable(src isa Leaf ? args[src.argindex] : src.array)
            b[t] = reshape(arr, node.dims...)
        elseif node isa Sliced
            src = c.trace.nodes[node.src]
            arr = src isa Leaf ? args[src.argindex] : src.array
            bind!(c.graph, b, t, bindable(view(arr, node.ranges...)))
        end
    end
    destarray(leafid) = (leaf = c.trace.nodes[leafid];
                         leaf isa Leaf ? args[leaf.argindex] : leaf.array)
    for (vid, leafid) in c.trace.destinations
        bind!(c.graph, b, c.tensors[vid], bindable(destarray(leafid)))
    end
    # outputs allocate their logical shape; the layout check accepts the dense
    # buffer, and a permuted output's declared strides make the engine write it
    # dense in the returned axis order
    outs = map(zip(c.outputs, c.output_dims, c.output_eltypes)) do (id, dims, T)
        haskey(c.trace.destinations, id) && return destarray(c.trace.destinations[id])
        t = c.tensors[id]
        arr = c.allocator(T, Tuple(dims))
        b[t] = arr
        arr
    end
    return b, outs
end

function (c::Compiled)(args...)
    b, outs = bindings(c, args...)
    execute!(c.graph, b)
    return isempty(outs) ? nothing : length(outs) == 1 ? outs[1] : Tuple(outs)
end

Base.show(io::IO, c::Compiled) =
    print(io, "$Compiled($(c.nargs) arguments, \
               $(count(!isnothing, c.tensors)) tensors)")
