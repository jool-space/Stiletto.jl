# ## Broadcasting
#
# Julia hands us the expression tree lazily as nested Broadcasted objects;
# each call maps onto one cuDNN pointwise node, which the runtime compiler
# fuses back together. Array +, -, scalar * and / come from Base's
# AbstractArray generics, which route through promote_shape and broadcast —
# landing here.

struct TracedStyle <: Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:TracedArray}) = TracedStyle()
Base.Broadcast.BroadcastStyle(s::TracedStyle, ::Broadcast.BroadcastStyle) = s
Base.Broadcast.BroadcastStyle(s::TracedStyle, ::Broadcast.Unknown) = s

# extensible function → mode maps; packages add methods for their functions
# (an NNlib extension maps relu, sigmoid, gelu, ...)
unary_mode(@nospecialize f) = nothing
unary_mode(::typeof(exp)) = CUDNN_POINTWISE_EXP
unary_mode(::typeof(log)) = CUDNN_POINTWISE_LOG
unary_mode(::typeof(sqrt)) = CUDNN_POINTWISE_SQRT
unary_mode(::typeof(tanh)) = CUDNN_POINTWISE_TANH_FWD
unary_mode(::typeof(abs)) = CUDNN_POINTWISE_ABS
unary_mode(::typeof(ceil)) = CUDNN_POINTWISE_CEIL
unary_mode(::typeof(floor)) = CUDNN_POINTWISE_FLOOR
unary_mode(::typeof(cos)) = CUDNN_POINTWISE_COS
unary_mode(::typeof(sin)) = CUDNN_POINTWISE_SIN
unary_mode(::typeof(tan)) = CUDNN_POINTWISE_TAN
unary_mode(::typeof(inv)) = CUDNN_POINTWISE_RECIPROCAL
unary_mode(::typeof(-)) = CUDNN_POINTWISE_NEG
unary_mode(::typeof(+)) = CUDNN_POINTWISE_IDENTITY
unary_mode(::typeof(identity)) = CUDNN_POINTWISE_IDENTITY
unary_mode(::typeof(!)) = CUDNN_POINTWISE_LOGICAL_NOT

binary_mode(@nospecialize f) = nothing
binary_mode(::typeof(+)) = CUDNN_POINTWISE_ADD
binary_mode(::typeof(*)) = CUDNN_POINTWISE_MUL
binary_mode(::typeof(-)) = CUDNN_POINTWISE_SUB
binary_mode(::typeof(/)) = CUDNN_POINTWISE_DIV
binary_mode(::typeof(max)) = CUDNN_POINTWISE_MAX
binary_mode(::typeof(min)) = CUDNN_POINTWISE_MIN
binary_mode(::typeof(^)) = CUDNN_POINTWISE_POW
binary_mode(::typeof(atan)) = CUDNN_POINTWISE_ATAN2
binary_mode(::typeof(==)) = CUDNN_POINTWISE_CMP_EQ
binary_mode(::typeof(!=)) = CUDNN_POINTWISE_CMP_NEQ
binary_mode(::typeof(>)) = CUDNN_POINTWISE_CMP_GT
binary_mode(::typeof(>=)) = CUDNN_POINTWISE_CMP_GE
binary_mode(::typeof(<)) = CUDNN_POINTWISE_CMP_LT
binary_mode(::typeof(<=)) = CUDNN_POINTWISE_CMP_LE
binary_mode(::typeof(&)) = CUDNN_POINTWISE_LOGICAL_AND
binary_mode(::typeof(|)) = CUDNN_POINTWISE_LOGICAL_OR

# comparisons and logicals produce boolean tensors
boolean_output(mode::cudnnPointwiseMode_t) = mode in (
    CUDNN_POINTWISE_CMP_EQ, CUDNN_POINTWISE_CMP_NEQ, CUDNN_POINTWISE_CMP_GT,
    CUDNN_POINTWISE_CMP_GE, CUDNN_POINTWISE_CMP_LT, CUDNN_POINTWISE_CMP_LE,
    CUDNN_POINTWISE_LOGICAL_AND, CUDNN_POINTWISE_LOGICAL_OR, CUDNN_POINTWISE_LOGICAL_NOT)

# composite functions rewrite to mapped primitives
pointwise(::typeof(abs2), x::TracedArray) = pointwise(*, x, x)

# Float32.(x) and friends: an identity node with a typed output; cuDNN
# converts dtypes at node boundaries
function pointwise(::Type{T}, x::TracedArray) where {T<:Number}
    eltype(x) === T && return x
    return traced(x.trace, T, x.dims, Cast(x.id, T))
end

function pointwise(f, args::TracedArray...)
    tr = sametrace(args...)
    mode = length(args) == 1 ? unary_mode(f) :
           length(args) == 2 ? binary_mode(f) :
           length(args) == 3 && f === ifelse ? CUDNN_POINTWISE_BINARY_SELECT :
           nothing
    if mode === nothing && length(args) > 2 && (f === (+) || f === (*))
        return foldl((a, b) -> pointwise(f, a, b), args)
    end
    mode === nothing && throw(ArgumentError(
        "cannot trace $f with $(length(args)) arguments: no cuDNN pointwise mode"))
    if f === ifelse
        pred, x, b = args
        args = (x, b, pred)  # cuDNN binary select: y = t ? x : b, predicate third
    end
    shape = Broadcast.broadcast_shape(map(axes, args)...)
    # boolean masks don't participate in value promotion (predicates, masks)
    value_eltypes = [eltype(a) for a in args if eltype(a) != Bool]
    T = boolean_output(mode) || isempty(value_eltypes) ? Bool : promote_type(value_eltypes...)
    return traced(tr, T, map(length, shape), Pointwise(mode, Int[a.id for a in args]))
end

find_trace(t::TracedArray) = t.trace
find_trace(bc::Broadcast.Broadcasted) = findtrace_args(bc.args...)
find_trace(x) = nothing
findtrace_args() = nothing
function findtrace_args(x, rest...)
    tr = find_trace(x)
    return tr === nothing ? findtrace_args(rest...) : tr
end

walkbc(tr::Trace, t::TracedArray) = t
walkbc(tr::Trace, x::Number) = traced(tr, Float32, (), Constant(Float32(x)))
walkbc(tr::Trace, a::AbstractArray) = capture(tr, a)
function walkbc(tr::Trace, bc::Broadcast.Broadcasted)
    # scalar subtrees (x .+ 2f0 .* 3f0) fold at trace time instead of
    # becoming degenerate graph nodes
    scalar_subtree(bc) && return walkbc(tr, Broadcast.materialize(bc))
    if bc.f === Base.literal_pow  # x .^ 2 lowers to literal_pow(^, x, Val(2))
        _, x, v = bc.args
        p = pow_exponent(v isa Ref ? v[] : v)
        return pointwise(^, walkbc(tr, x), walkbc(tr, p))
    end
    return pointwise(bc.f, map(x -> walkbc(tr, x), bc.args)...)
end
walkbc(tr::Trace, x) = throw(ArgumentError("cannot trace broadcast over $(typeof(x))"))
pow_exponent(::Val{p}) where {p} = p

scalar_subtree(x::Number) = true
scalar_subtree(bc::Broadcast.Broadcasted) = all(scalar_subtree, bc.args)
scalar_subtree(x) = false

function Base.copy(bc::Broadcast.Broadcasted{TracedStyle})
    tr = find_trace(bc)
    tr === nothing && throw(ArgumentError("broadcast contains no TracedArray"))
    return walkbc(tr, bc)
end
