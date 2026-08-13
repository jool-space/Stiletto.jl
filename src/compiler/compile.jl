# ## Compilation

struct Compiled
    graph::Graph
    trace::Trace
    tensors::Vector{Union{Nothing,Tensor}}
    nargs::Int
    outputs::Vector{Int}
    output_dims::Vector{Vector{Int}}
    output_eltypes::Vector{DataType}
    extra::IdDict{Tensor,Any}   # bindings decided at emission (mega-op auxiliaries)
    allocator::Any
end

default_allocator(::Type{T}, dims::Dims) where {T} = CuArray{T}(undef, dims)

"""
    compile(f, args...; io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32,
            allocator=(T, dims) -> CuArray{T}(undef, dims),
            max_workspace=nothing, deterministic=false, heuristics=nothing)

Trace `f` applied to symbolic stand-ins for `args`, build the resulting cuDNN
graph, and return a callable. Calling the result with arrays of the same
shapes as `args` executes the graph and returns outputs obtained from
`allocator`; in-place assignments (`mul!`, `y .= ...`) write into the
caller's buffers and allocate nothing.

Engine selection is steerable: `max_workspace` (bytes) caps the workspace a
plan may demand, `deterministic=true` rejects engines flagged numerically
nondeterministic, and `heuristics` overrides the heuristic modes consulted
(a tuple of `cuDNN.cudnnBackendHeurMode_t` values, tried in order). When no
engine satisfies the constraints, compilation throws
`cuDNN.UnsupportedGraphError` rather than silently relaxing them.
"""
function compile(f, args...; io_dtype=Float32, intermediate_dtype=Float32,
                 compute_dtype=Float32, allocator=default_allocator,
                 max_workspace::Union{Nothing,Integer}=nothing,
                 deterministic::Bool=false, heuristics=nothing)
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
    # a permuted computed value is a materialized output, not a graph value
    for node in tr.nodes, r in noderefs(node)
        tr.nodes[r] isa Permuted && throw(ArgumentError(
            "a permuted computed value materializes at the output boundary " *
            "and cannot feed further operations"))
    end

    R = graph_rank(tr)
    g = Graph(; io_dtype, intermediate_dtype, compute_dtype)
    tensors = Vector{Union{Nothing,Tensor}}(nothing, length(tr.nodes))
    tf = TensorFor(g, tr, tensors, R)
    for id in keys(tr.destinations)   # writes execute even when not returned
        tf(id)
    end
    seen = Set{Tensor}()   # mutable struct: hashed by identity
    for o in outputs
        t = tf(o.id)
        haskey(tr.destinations, o.id) && continue  # already a bound output
        t in seen && throw(ArgumentError(
            "the same computed value cannot be returned twice, permuted or not"))
        push!(seen, t)
        t.output && continue  # declared an output at emission (permuted layout)
        t.virtual || throw(ArgumentError(
            "traced function must return computed values, not inputs"))
        output!(t)
    end
    # cuDNN's select_plan owns the heuristic-mode default; forward it only
    # when the caller overrides
    build_kwargs = (; deterministic, max_workspace)
    heuristics === nothing || (build_kwargs = (; build_kwargs..., heuristics))
    build!(g; build_kwargs...)
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
                    [collect(Int, o.dims) for o in outputs], eltypes, tf.extra,
                    allocator)
end

# memoized lazy declaration; recursion follows data dependencies, so ops are
# pushed in topological order and dead nodes are skipped entirely
struct TensorFor
    g::Graph
    trace::Trace
    tensors::Vector{Union{Nothing,Tensor}}
    rank::Int
    extra::IdDict{Tensor,Any}   # emit! may record bindings for op-internal inputs
    dests::Dict{Int,Tensor}     # pre-built destination tensors (permuted outputs)
    aux::Dict{Tuple{Int,Symbol},Tensor}   # auxiliary results of multi-result ops
end
TensorFor(g, trace, tensors, rank) =
    TensorFor(g, trace, tensors, rank, IdDict{Tensor,Any}(), Dict{Int,Tensor}(),
              Dict{Tuple{Int,Symbol},Tensor}())
function (tf::TensorFor)(i::Int)
    t = tf.tensors[i]
    t === nothing || return t
    return tf.tensors[i] = emit!(tf.g, tf, i, tf.trace.nodes[i], tf.rank)
end

