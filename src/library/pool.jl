# ## Pooling
#
# Windowed spatial reductions (cuDNN resample ops) over
# `(spatial..., channels, batch)` operands, in the usual tiers.

"""
    Stiletto.maxpool(x, window; stride=window, pre_padding=0, post_padding=pre_padding)
    Stiletto.meanpool(x, window; ...; include_padding=false)
    Stiletto.maxpool!(y, x, window; ...) / meanpool!(y, x, window; ...)

Pool `x :: (spatial..., channels, batch)` over `window`, producing
`(out_spatial..., channels, batch)`. `stride` defaults to the window
(non-overlapping). `meanpool` divides by the number of contributing
elements; `include_padding=true` divides by the full window size instead.
"""
maxpool(x::TracedArray, window; kwargs...) = pool_traced(:maxpool, x, window; kwargs...)
meanpool(x::TracedArray, window; include_padding::Bool=false, kwargs...) =
    pool_traced(include_padding ? :avgpool_include_padding : :avgpool_exclude_padding,
                x, window; kwargs...)
maxpool!(y::TracedArray, x::TracedArray, window; kwargs...) =
    assign!(y, maxpool(x, window; kwargs...))
meanpool!(y::TracedArray, x::TracedArray, window; kwargs...) =
    assign!(y, meanpool(x, window; kwargs...))

function pool_traced(mode::Symbol, x::TracedArray, window;
                     stride=window, pre_padding=0, post_padding=pre_padding)
    ndims(x) >= 3 || throw(ArgumentError(
        "pooling input is (spatial..., channels, batch), at least rank 3"))
    rank = ndims(x) - 2
    win, str = spatial_params(window, rank), spatial_params(stride, rank)
    pre, post = spatial_params(pre_padding, rank), spatial_params(post_padding, rank)
    outsp = pool_output_spatial(size(x), win, pre, post, str, rank)
    all(>(0), outsp) || throw(DimensionMismatch(
        "pooling output spatial dimensions $(outsp) must be positive"))
    return traced(x.trace, eltype(x), (outsp..., size(x, rank + 1), size(x, rank + 2)),
                  Resample(x.id, mode, win, pre, post, str))
end

pool_output_spatial(x, win, pre, post, stride, rank) =
    ntuple(i -> fld(x[i] + pre[i] + post[i] - win[i], stride[i]) + 1, rank)

# ## Eager tier

maxpool(x::AbstractArray, window; kwargs...) = pool_eager(:maxpool, x, window; kwargs...)
meanpool(x::AbstractArray, window; include_padding::Bool=false, kwargs...) =
    pool_eager(include_padding ? :avgpool_include_padding : :avgpool_exclude_padding,
               x, window; kwargs...)

function pool_eager(mode, x, window; stride=window, pre_padding=0,
                    post_padding=pre_padding)
    rank = ndims(x) - 2
    win, str = spatial_params(window, rank), spatial_params(stride, rank)
    pre, post = spatial_params(pre_padding, rank), spatial_params(post_padding, rank)
    outsp = pool_output_spatial(size(x), win, pre, post, str, rank)
    y = similar(x, (outsp..., size(x, rank + 1), size(x, rank + 2)))
    return pool!(mode, y, x, window; stride, pre_padding, post_padding)
end

maxpool!(y::AbstractArray, x::AbstractArray, window; kwargs...) =
    pool!(:maxpool, y, x, window; kwargs...)
meanpool!(y::AbstractArray, x::AbstractArray, window; include_padding::Bool=false,
          kwargs...) =
    pool!(include_padding ? :avgpool_include_padding : :avgpool_exclude_padding,
          y, x, window; kwargs...)

function pool!(mode::Symbol, y::AbstractArray, x::AbstractArray, window;
               stride=window, pre_padding=0, post_padding=pre_padding)
    kw = (; stride, pre_padding, post_padding)
    jit((y, x) -> (assign!(y, pool_traced(mode, x, window; kw...)); nothing), y, x)
    return y
end
