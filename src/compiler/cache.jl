# ## Eager execution
#
# Named ops called on real GPU arrays compile a standalone graph once per
# (op, shapes, dtypes, strides) signature and reuse it; plan construction
# runs engine heuristics and is far too slow to repeat per call. Compiled
# graphs live in cuDNN's per-handle plan cache, inheriting its lifecycle:
# per-context, pooled with the handle, reclaimed under memory pressure, and
# negative results (unsupported graphs) cached alongside.

argkey(a::AbstractArray) = (typeof(a), size(a), strides(a))
argkey(x::Number) = typeof(x)   # scalar arguments rebind by value at each call
argkey(x) = x

function cached_compile(key, f, args...; kwargs...)
    ck = (:stiletto, key, map(argkey, args)...)
    cached = get!(cuDNN.handle().plans, ck) do
        try
            compile(f, args...; kwargs...)
        catch e
            e isa UnsupportedGraphError || rethrow()
            e
        end
    end
    cached isa UnsupportedGraphError && throw(cached)
    return cached::Compiled
end

"""
    jit(f, args...; kwargs...)

Compile `f` for this argument signature on first use and execute the cached
graph, so `jit` can be called inline in hot code (a forward pass). Entries
live in cuDNN's per-handle plan cache. Under CUDA graph capture the call
replays as recorded stream work; the in-place forms (`mul!`, `y .= ...`) are
the natural fit there, binding stable caller buffers and allocating nothing.

Arguments are runtime inputs — what the graph sees: arrays are keyed by
shape/dtype/strides and rebind their pointers per call, scalars become
by-value tensors rebound per call. Captured values are trace-time constants
— what the tracer sees: isbits captures (flags, dims) specialize the graph
per value and may be branched on; captured arrays are baked by identity.
Pass data as arguments; capture configuration.

Keyword arguments forward to [`compile`](@ref) — including the
engine-selection knobs (`max_workspace`, `deterministic`, `heuristics`) —
and participate in the plan key: the same call with a different knob is a
different cached plan.
"""
function jit(f::F, args...; kwargs...) where {F}
    fkey = ntuple(i -> capture_key(getfield(f, i)), fieldcount(F))
    key = (:jit, F, fkey, values(kwargs))
    return cached_compile(key, f, args...; kwargs...)(args...)
end

# closures from one site share a type; their captured fields tell them apart
capture_key(v::AbstractArray) = objectid(v)
capture_key(v) = isbits(v) ? v : objectid(v)

# ## Call macros
#
# `@compile f(a, b)` and `@jit f(a, b)` expand to `compile(f, a, b)` and
# `jit(f, a, b)`. Keyword arguments in the call ride a closure into the
# traced function — they become trace-time constants (captures), keeping
# positional arguments as the runtime inputs.

"""
    g = @compile f(args...; kwargs...)

Expand to `compile(f, args...)`, returning the compiled callable without
executing it. Call keywords are closed over as trace-time constants.
"""
macro compile(ex)
    return esc(tracing_call(compile, ex))
end

"""
    d = @jit f(args...; kwargs...)

Expand to `jit(f, args...)`: compile on first use, execute the cached graph.
Call keywords are closed over as trace-time constants.
"""
macro jit(ex)
    return esc(tracing_call(jit, ex))
end

function tracing_call(fn, ex)
    Meta.isexpr(ex, :call) || throw(ArgumentError(
        "@$(nameof(fn)) expects a function call, e.g. @$(nameof(fn)) f(a, b)"))
    callee = ex.args[1]
    kwargs, positional = Any[], Any[]
    for a in ex.args[2:end]
        Meta.isexpr(a, :parameters) ? append!(kwargs, a.args) :
        Meta.isexpr(a, :kw)         ? push!(kwargs, a) :
                                      push!(positional, a)
    end
    isempty(kwargs) && return Expr(:call, fn, callee, positional...)
    params, inner = Any[], Any[]
    for a in positional
        s = gensym(:arg)
        splat = Meta.isexpr(a, :...)
        push!(params, splat ? Expr(:..., s) : s)
        push!(inner, splat ? Expr(:..., s) : s)
    end
    lambda = Expr(:->, Expr(:tuple, params...),
                  Expr(:call, callee, Expr(:parameters, kwargs...), inner...))
    return Expr(:call, fn, lambda, positional...)
end
