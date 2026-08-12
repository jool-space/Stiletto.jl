# ## Norms
#
# Per-sample and per-channel normalizations as named functions (cuDNN
# mega-ops don't trace from primitives), in two tiers: TracedArray methods
# become nodes of the surrounding trace; the non-traced methods execute a
# cached destination graph eagerly, with allocating verbs delegating to the
# mutating ones through `similar`. Engine availability is cuDNN's: a norm
# without an engine throws UnsupportedGraphError. The reference
# implementations live in the test suite.
#
# Normalization happens over the dimensions `scale` spans: a rank-1 scale of
# length size(x, 1) normalizes over the first dimension.

rmsnorm(x::TracedArray, scale; bias=nothing, epsilon::Real=1f-5) =
    norm_traced(:rmsnorm, x, scale, bias, epsilon)
layernorm(x::TracedArray, scale, bias; epsilon::Real=1f-5) =
    norm_traced(:layernorm, x, scale, bias, epsilon)
rmsnorm!(y::TracedArray, x::TracedArray, scale; bias=nothing, epsilon::Real=1f-5) =
    assign!(y, rmsnorm(x, scale; bias, epsilon))
layernorm!(y::TracedArray, x::TracedArray, scale, bias; epsilon::Real=1f-5) =
    assign!(y, layernorm(x, scale, bias; epsilon))

function norm_traced(mode::Symbol, x::TracedArray, scale, bias, epsilon)
    tr = x.trace
    s = scale isa TracedArray ? scale : capture(tr, scale)
    b = bias === nothing ? nothing : bias isa TracedArray ? bias : capture(tr, bias)
    # cuDNN norm parameters are graph inputs: a computed scale (e.g. w .+ 1)
    # fails op-graph finalization, so demand it be precomputed before the trace
    for (name, t) in ("scale" => s, "bias" => b)
        t === nothing && continue
        tr.nodes[t.id] isa Union{Leaf,Captured,Presented,Reshaped} || throw(ArgumentError(
            "norm $name must be an input of the trace, not a computed value; " *
            "precompute it before compiling"))
    end
    eps = traced(tr, Float32, (), Constant(Float32(epsilon)))
    return traced(tr, eltype(x), x.dims,
                  Norm(mode, x.id, s.id, b === nothing ? nothing : b.id,
                       nothing, nothing, eps.id))
end

"""
    Stiletto.batchnorm(x, scale, bias, mean, inv_variance)
    Stiletto.batchnorm!(y, x, scale, bias, mean, inv_variance)

Inference-phase batch normalization over the channel dimension — the
second-to-last of `x`, which must be at least 3-dimensional
(`(spatial..., channels, batch)`). Parameters and statistics are per-channel
vectors of length `size(x, ndims(x) - 1)`. `inv_variance` is the folded
inverse standard deviation `1 ./ sqrt.(running_var .+ epsilon)`; fold it once
when loading parameters.
"""
function batchnorm(x::TracedArray, scale, bias, mean, inv_variance)
    cdims = batchnorm_cdims(x)
    tr = x.trace
    # in-trace reshape puts per-channel vectors on the channel axis and
    # enforces that parameters are trace inputs
    channels(p) = p isa TracedArray ? reshape(p, cdims) : capture(tr, reshape(p, cdims))
    s, b, m, iv = channels.((scale, bias, mean, inv_variance))
    return traced(tr, eltype(x), x.dims,
                  Norm(:batchnorm, x.id, s.id, b.id, m.id, iv.id, nothing))
end

batchnorm!(y::TracedArray, x::TracedArray, scale, bias, mean, inv_variance) =
    assign!(y, batchnorm(x, scale, bias, mean, inv_variance))

function batchnorm_cdims(x)
    ndims(x) >= 3 || throw(ArgumentError(
        "batchnorm input must be at least 3-dimensional ((spatial..., channels, batch))"))
    return ntuple(i -> i == ndims(x) - 1 ? size(x, ndims(x) - 1) : 1, ndims(x))
end

# ## Eager tier
#
# jit of the traced form: one destination graph per op and signature, cached
# per handle. Allocating verbs delegate through `similar`. If the stack has
# no engine for a norm, the UnsupportedGraphError propagates — the library
# does not silently degrade to unfused kernels.

rmsnorm(x::AbstractArray, scale; bias=nothing, epsilon::Real=1f-5) =
    rmsnorm!(similar(x), x, scale; bias, epsilon)
layernorm(x::AbstractArray, scale, bias; epsilon::Real=1f-5) =
    layernorm!(similar(x), x, scale, bias; epsilon)
batchnorm(x::AbstractArray, scale, bias, mean, inv_variance) =
    batchnorm!(similar(x), x, scale, bias, mean, inv_variance)

function rmsnorm!(y::AbstractArray, x::AbstractArray, scale;
                   bias=nothing, epsilon::Real=1f-5)
    eps = Float32(epsilon)
    if bias === nothing
        jit((y, x, s) -> (rmsnorm!(y, x, s; epsilon=eps); nothing), y, x, scale)
    else
        jit((y, x, s, b) -> (rmsnorm!(y, x, s; bias=b, epsilon=eps); nothing),
            y, x, scale, bias)
    end
    return y
end

function layernorm!(y::AbstractArray, x::AbstractArray, scale, bias; epsilon::Real=1f-5)
    eps = Float32(epsilon)
    jit((y, x, s, b) -> (layernorm!(y, x, s, b; epsilon=eps); nothing), y, x, scale, bias)
    return y
end

function batchnorm!(y::AbstractArray, x::AbstractArray, scale, bias, mean, inv_variance)
    ps = map(p -> reshape(p, batchnorm_cdims(x)), (scale, bias, mean, inv_variance))
    jit((y, x, ps...) -> (batchnorm!(y, x, ps...); nothing), y, x, ps...)
    return y
end
