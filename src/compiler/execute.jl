# ## Execution

# declarations carry the presentation (permuted strides); bindings take the
# dense storage behind it, which the layout check accepts as the same layout
bindable(a) = a
bindable(a::Union{PermutedDimsArray,Transpose,Adjoint}) = bindable(parent(a))

function (c::Compiled)(args...)
    length(args) == c.nargs ||
        throw(ArgumentError("compiled graph takes $(c.nargs) arrays, got $(length(args))"))
    bindings = copy(c.extra)
    for (i, node) in enumerate(c.trace.nodes)
        t = c.tensors[i]
        t === nothing && continue
        if node isa Leaf
            bindings[t] = t.by_value ? args[node.argindex] : bindable(args[node.argindex])
        elseif node isa Captured
            bindings[t] = bindable(node.array)
        elseif node isa Constant
            bindings[t] = node.value
        elseif node isa Presented
            src = c.trace.nodes[node.src]
            bindings[t] = bindable(src isa Leaf ? args[src.argindex] : src.array)
        elseif node isa Reshaped
            src = c.trace.nodes[node.src]
            arr = bindable(src isa Leaf ? args[src.argindex] : src.array)
            bindings[t] = reshape(arr, node.dims...)
        end
    end
    destarray(leafid) = (leaf = c.trace.nodes[leafid];
                         leaf isa Leaf ? args[leaf.argindex] : leaf.array)
    for (vid, leafid) in c.trace.destinations
        bindings[c.tensors[vid]] = bindable(destarray(leafid))
    end
    # outputs allocate their logical shape; the layout check accepts the dense
    # buffer, and a permuted output's declared strides make the engine write it
    # dense in the returned axis order
    outs = map(zip(c.outputs, c.output_dims, c.output_eltypes)) do (id, dims, T)
        haskey(c.trace.destinations, id) && return destarray(c.trace.destinations[id])
        t = c.tensors[id]
        arr = c.allocator(T, Tuple(dims))
        bindings[t] = arr
        arr
    end
    execute!(c.graph, bindings)
    return isempty(outs) ? nothing : length(outs) == 1 ? outs[1] : Tuple(outs)
end

Base.show(io::IO, c::Compiled) =
    print(io, "$Compiled($(c.nargs) arguments, \
               $(count(!isnothing, c.tensors)) tensors)")

