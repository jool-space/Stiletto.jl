# ## Convolution
#
# Cross-correlation over `(spatial..., channels, batch)` operands, in the
# usual tiers. The group count is inferred from the channel ratio, cuDNN
# style, so depthwise convolution is `w :: (spatial..., 1, channels)`.
# Asymmetric `pre_padding`/`post_padding` express causal windows: a
# depthwise causal conv1d is `conv(x, w; pre_padding=K-1, post_padding=0)`.
#
# Unlike matmul and pointwise, convolution reads meaning from dimension
# positions counted from the end, so it is not invariant under the trailing
# singleton lift: the conv input must be a maximal-rank tensor of its trace.

function spatial_params(v, rank)
    out = v isa Integer ? fill(Int(v), rank) : collect(Int, v)
    length(out) == rank || throw(DimensionMismatch(
        "expected $rank spatial parameters, got $v"))
    return out
end

conv_output_spatial(x, w, pre, post, stride, dilation, rank) =
    ntuple(i -> 1 + (x[i] + pre[i] + post[i] - dilation[i] * (w[i] - 1) - 1) ÷ stride[i],
           rank)

"""
    Stiletto.conv(x, w; stride=1, dilation=1, pre_padding=0, post_padding=pre_padding)
    Stiletto.conv!(y, x, w; ...)

Cross-correlation of `x :: (spatial..., channels, batch)` with filters
`w :: (spatial..., channels ÷ groups, out_channels)`, producing
`(out_spatial..., out_channels, batch)`. The group count is inferred from
the channel ratio (depthwise: filter channel extent 1). Spatial keywords
take a scalar or one value per spatial dimension; asymmetric padding
expresses causal windows (`pre_padding=K-1, post_padding=0`).
"""
function conv(x::TracedArray, w; stride=1, dilation=1, pre_padding=0,
              post_padding=pre_padding)
    tr = x.trace
    wt = w isa TracedArray ? w : capture(tr, w)
    sametrace(x, wt)
    ndims(x) >= 3 || throw(ArgumentError(
        "conv input is (spatial..., channels, batch), at least rank 3"))
    ndims(wt) == ndims(x) || throw(DimensionMismatch(
        "conv filter rank must match input rank: x $(size(x)), w $(size(wt))"))
    rank = ndims(x) - 2
    pre, post = spatial_params(pre_padding, rank), spatial_params(post_padding, rank)
    str, dil = spatial_params(stride, rank), spatial_params(dilation, rank)
    cin, cw, cout = size(x, rank + 1), size(wt, rank + 1), size(wt, rank + 2)
    cin % cw == 0 && cout % (cin ÷ cw) == 0 || throw(DimensionMismatch(
        "conv channels do not divide into groups: x $(size(x)), w $(size(wt))"))
    outsp = conv_output_spatial(size(x), size(wt), pre, post, str, dil, rank)
    all(>(0), outsp) || throw(DimensionMismatch(
        "conv output spatial dimensions $(outsp) must be positive"))
    T = promote_type(eltype(x), eltype(wt))
    return traced(tr, T, (outsp..., cout, size(x, rank + 2)),
                  Conv(x.id, wt.id, pre, post, str, dil))
end

conv!(y::TracedArray, x::TracedArray, w; kwargs...) = assign!(y, conv(x, w; kwargs...))

# ## Eager tier

function conv(x::AbstractArray, w; stride=1, dilation=1, pre_padding=0,
              post_padding=pre_padding)
    rank = ndims(x) - 2
    pre, post = spatial_params(pre_padding, rank), spatial_params(post_padding, rank)
    str, dil = spatial_params(stride, rank), spatial_params(dilation, rank)
    outsp = conv_output_spatial(size(x), size(w), pre, post, str, dil, rank)
    y = similar(x, (outsp..., size(w, rank + 2), size(x, rank + 2)))
    return conv!(y, x, w; stride, dilation, pre_padding, post_padding)
end

function conv!(y::AbstractArray, x::AbstractArray, w; stride=1, dilation=1,
               pre_padding=0, post_padding=pre_padding)
    kw = (; stride, dilation, pre_padding, post_padding)
    jit((y, x, w) -> (conv!(y, x, w; kw...); nothing), y, x, w)
    return y
end
