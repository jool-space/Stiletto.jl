# ## Block-scale quantization
#
# `quantize!` writes a computed value into a block-scaled composite buffer —
# the output half of the narrow-type story (dequantize is an input
# decoration on BlockscaledArray arguments). Methods live in the
# Microscaling extension, which owns the composite type; this stub carries
# the name and contract.

"""
    Stiletto.quantize!(dest, x)

Quantize `x` into `dest::BlockscaledArray`, writing quantized elements and
block scales into `dest`'s storage components. Block size, element type, and
scale type are read from `dest`; the blocked dimension is the one `dest`'s
block spans. Inside a trace, `dest` must be an argument and `x` is fused
into the graph — cuDNN's engines support quantize fused after a matmul
(`quantize!(dest, w' * x)`), the full narrow-precision pipeline in one
kernel.

Assignment into a block-scaled destination is the same store — quantization
is how the destination stores values — so traced `dest .= x` and
`mul!(dest, a, b)` route here. Partial writes (`view(dest, ...) .= x`) are
refused: block scales couple elements, so only whole-array stores have a
meaning.

Requires the Microscaling extension (`using Microscaling`).
"""
function quantize! end
