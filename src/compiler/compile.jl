# ## Compilation

struct Compiled
    graph::Graph
    trace::Trace
    tensors::Vector{Union{Nothing,Tensor}}
    nargs::Int
    outputs::Vector{Int}
    output_dims::Vector{Vector{Int}}
    output_eltypes::Vector{DataType}
    allocator::Any
end

default_allocator(::Type{T}, dims::Dims) where {T} = CuArray{T}(undef, dims)

"""
    compile(f, args...; io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32,
            allocator=(T, dims) -> CuArray{T}(undef, dims))

Trace `f` applied to symbolic stand-ins for `args`, build the resulting cuDNN
graph, and return a callable. Calling the result with arrays of the same
shapes as `args` executes the graph and returns outputs obtained from
`allocator`; in-place assignments (`mul!`, `y .= ...`) write into the
caller's buffers and allocate nothing.
"""
function compile(f, args...; io_dtype=Float32, intermediate_dtype=Float32,
                 compute_dtype=Float32, allocator=default_allocator)
    tr = Trace()
    # scalar arguments become by-value tensors: runtime inputs rebound per
    # call, not trace-time constants
    targs = ntuple(length(args)) do i
        a = args[i]
        a isa Number ? traced(tr, Float32, (), Leaf(i, a)) :
                       traced(tr, eltype(a), size(a), Leaf(i, a))
    end
    result = f(targs...)
    outputs = result === nothing ? TracedArray[] :
              result isa TracedArray ? TracedArray[result] :
              result isa Union{Tuple,AbstractVector} && !isempty(result) &&
                  all(x -> x isa TracedArray, result) ? TracedArray[result...] :
              throw(ArgumentError(
                  "traced function must return TracedArrays or nothing, got $(typeof(result)); " *
                  "a value computed without the traced arguments cannot be part of the graph"))
    isempty(outputs) && isempty(tr.destinations) &&
        throw(ArgumentError("traced function returned no TracedArrays"))
    isempty(outputs) || sametrace(outputs...) === tr ||
        throw(ArgumentError("traced function returned values from another trace"))

    # graph execution is dataflow, not program order: reading a buffer after
    # writing it in place has no defined meaning, so refuse it
    for (vid, leafid) in tr.destinations, j in vid+1:length(tr.nodes)
        leafid in noderefs(tr.nodes[j]) && throw(ArgumentError(
            "an input written in place is read afterwards; " *
            "use the assigned value instead of the original input"))
    end

    R = graph_rank(tr)
    g = Graph(; io_dtype, intermediate_dtype, compute_dtype)
    tensors = Vector{Union{Nothing,Tensor}}(nothing, length(tr.nodes))
    tf = TensorFor(g, tr, tensors, R)
    for id in keys(tr.destinations)   # writes execute even when not returned
        tf(id)
    end
    for o in outputs
        t = tf(o.id)
        haskey(tr.destinations, o.id) && continue  # already a bound output
        t.virtual || throw(ArgumentError(
            "traced function must return computed values, not inputs"))
        output!(t)
    end
    build!(g)
    # explicitly typed outputs (casts) keep their traced eltype; the rest
    # follow io_dtype (the trace knows the Julia type, so no reverse
    # dtype-enum mapping is needed — that map is not extensible)
    eltypes = DataType[tensors[o.id].dtype === nothing ? io_dtype : eltype(o)
                       for o in outputs]
    # examples are only needed during emission; binding uses the call's
    # arguments, so don't keep the trace-time arrays alive (cached Compiled
    # objects would pin them indefinitely)
    for (i, node) in enumerate(tr.nodes)
        node isa Leaf && (tr.nodes[i] = Leaf(node.argindex, nothing))
    end
    return Compiled(g, tr, tensors, length(args), [o.id for o in outputs],
                    [collect(Int, o.dims) for o in outputs], eltypes, allocator)
end

# memoized lazy declaration; recursion follows data dependencies, so ops are
# pushed in topological order and dead nodes are skipped entirely
struct TensorFor
    g::Graph
    trace::Trace
    tensors::Vector{Union{Nothing,Tensor}}
    rank::Int
end
function (tf::TensorFor)(i::Int)
    t = tf.tensors[i]
    t === nothing || return t
    return tf.tensors[i] = emit!(tf.g, tf, i, tf.trace.nodes[i], tf.rank)
end

