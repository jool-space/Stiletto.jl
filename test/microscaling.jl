# Block-scaled (MXFP8/NVFP4) arguments: a BlockscaledArray traced like any
# array declares element + swizzled-scale tensors joined by a dequantize node,
# and binds its storage components at execution. cuDNN's block-scale recipes
# want the blocked (reduction) dimension innermost in storage, so quantized
# gemm operands are stored (K, ·) and the left operand is transposed in the
# traced code.

using Microscaling: BlockscaledArray, Sm1xxArray, sm1xx, elements, scales,
    Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN
using BitPacking: NarrowArray

# test-owned reference: expand each scale over its block of elements
dequant_ref(e, s, block) =
    Float32.(e) .* repeat(Float32.(s); inner=(block, ntuple(_ -> 1, ndims(e) - 1)...))

# the block-scale graph ops only have Blackwell engines; the graph always
# builds, so a wrong gate fails loudly instead of silently skipping
blockscale_claimed = CUDACore.capability(CUDACore.device()) >= v"10.0"

compile_blockscale(f, args...) = try
    compile(f, args...)
catch e
    e isa cuDNN.UnsupportedGraphError || rethrow()
    nothing
end

gpu_elements(a) = eltype(a) === Float4_E2M1FN ?
    NarrowArray{Float4_E2M1FN}(CuArray(a)) : CuArray(a)

@testset "block-scaled matmul" begin
    @testset "$label" for (label, ElemW, ElemX, Scale, block) in (
        ("MXFP8", Float8_E4M3FN, Float8_E4M3FN, Float8_E8M0FNU, 32),
        ("MXFP8 e4m3×e5m2", Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, 32),
        ("MXFP4", Float4_E2M1FN, Float4_E2M1FN, Float8_E8M0FNU, 32),
        ("NVFP4", Float4_E2M1FN, Float4_E2M1FN, Float8_E4M3FN, 16),
    )
        M, N, K = 256, 128, 256
        w_element = ElemW.(randn(Float32, K, M))
        x_element = ElemX.(randn(Float32, K, N))
        w_scale = Scale.(rand(Float32, K ÷ block, M))
        x_scale = Scale.(rand(Float32, K ÷ block, N))
        C_ref = dequant_ref(w_element, w_scale, block)' *
                dequant_ref(x_element, x_scale, block)

        W = BlockscaledArray(sm1xx(CuArray(w_scale)), gpu_elements(w_element))
        X = BlockscaledArray(sm1xx(CuArray(x_scale)), gpu_elements(x_element))

        c = compile_blockscale((w, x) -> w' * x, W, X)
        @test (c !== nothing) == blockscale_claimed
        c === nothing && continue
        @test Array(c(W, X)) ≈ C_ref rtol=1e-4 atol=1e-4

        # a compiled graph rebinds to new storage of the same shapes
        w_element2 = ElemW.(randn(Float32, K, M))
        W2 = BlockscaledArray(sm1xx(CuArray(w_scale)), gpu_elements(w_element2))
        @test Array(c(W2, X)) ≈
              dequant_ref(w_element2, w_scale, block)' *
              dequant_ref(x_element, x_scale, block) rtol=1e-4 atol=1e-4
    end
end

# the loud claim lives above; the remaining tests only run where the engines
# exist
if blockscale_claimed

@testset "block-scaled captures and jit" begin
    Scale, Element, block = Float8_E8M0FNU, Float8_E4M3FN, 32
    M, N, K = 256, 128, 256
    w_element = Element.(randn(Float32, K, M))
    x_element = Element.(randn(Float32, K, N))
    w_scale = Scale.(rand(Float32, K ÷ block, M))
    x_scale = Scale.(rand(Float32, K ÷ block, N))
    C_ref = dequant_ref(w_element, w_scale, block)' *
            dequant_ref(x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_element))

    # quantized weights close over the traced function like any array capture
    layer = x -> W' * x
    c = compile(layer, X)
    @test Array(c(X)) ≈ C_ref rtol=1e-4 atol=1e-4

    # jit keys blockscaled arguments by their components' shapes
    f = (w, x) -> w' * x
    nl = length(cuDNN.handle().plans)
    @test Array(jit(f, W, X)) ≈ C_ref rtol=1e-4 atol=1e-4
    @test length(cuDNN.handle().plans) == nl + 1
    jit(f, W, X)
    @test length(cuDNN.handle().plans) == nl + 1
end

@testset "block-scaled matmul epilogues" begin
    Scale, Element, block = Float8_E8M0FNU, Float8_E4M3FN, 32
    M, N, K = 256, 128, 256
    w_element = Element.(randn(Float32, K, M))
    x_element = Element.(randn(Float32, K, N))
    w_scale = Scale.(rand(Float32, K ÷ block, M) / √K)
    x_scale = Scale.(rand(Float32, K ÷ block, N) / √K)
    C_ref = dequant_ref(w_element, w_scale, block)' *
            dequant_ref(x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_element))
    b = CuArray(rand(Float32, M))

    # engine support for fusing an epilogue onto the dequantize→matmul graph
    # is cuDNN's to grant; skip rather than claim
    c = compile_blockscale((w, x, b) -> relu.(w' * x .+ b), W, X, b)
    if c === nothing
        @test_skip blockscale_claimed
    else
        @test Array(c(W, X, b)) ≈ relu.(C_ref .+ Array(b)) rtol=1e-4 atol=1e-4
    end

    # in-place: quantized inputs, dense destination
    C = CuArray(zeros(Float32, M, N))
    ci = compile_blockscale((c, w, x) -> mul!(c, w', x), C, W, X)
    if ci === nothing
        @test_skip blockscale_claimed
    else
        ci(C, W, X)
        @test Array(C) ≈ C_ref rtol=1e-4 atol=1e-4
    end

    # materializing a dequantize through pointwise alone (`d .= a`) is the
    # mirror of standalone quantize: the graph is valid, but engines only
    # take dequantize on the matmul path; skip rather than claim
    dm = CuArray(zeros(Float32, K, M))
    cm = compile_blockscale((d, a) -> (d .= a; nothing), dm, W)
    if cm === nothing
        @test_skip blockscale_claimed
    else
        cm(dm, W)
        @test Array(dm) ≈ dequant_ref(w_element, w_scale, block) rtol=1e-5
    end
end

@testset "fused matmul→quantize" begin
    # the full narrow-precision pipeline in one graph: dequantize both
    # operands, matmul, quantize back into a BlockscaledArray destination —
    # elements and swizzled scales written into the composite's storage
    Scale, Element, block = Float8_E8M0FNU, Float8_E4M3FN, 32
    M, N, K = 256, 128, 256
    w_element = Element.(randn(Float32, K, M))
    x_element = Element.(randn(Float32, K, N))
    w_scale = Scale.(rand(Float32, K ÷ block, M) / √K)
    x_scale = Scale.(rand(Float32, K ÷ block, N) / √K)
    C_ref = dequant_ref(w_element, w_scale, block)' *
            dequant_ref(x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_element))
    D = BlockscaledArray(
        Sm1xxArray(CuArray{Scale}(undef, 4, 4, 32, (M ÷ block) ÷ 4, N ÷ 128)),
        CuArray{Element}(undef, M, N))

    # measured supported on sm_121 / cuDNN 9.24
    c = compile((d, w, x) -> Stiletto.quantize!(d, w' * x), D, W, X)
    c(D, W, X)
    # block-quantization error dominates the loose tolerance; a mislaid
    # swizzle misses by powers of two, so this still catches layout errors
    @test Float32.(Array(copy(D))) ≈ C_ref rtol=0.15 atol=0.15

    # chained layers: the quantized activation feeds the next quantized
    # matmul as an ordinary argument — per-layer narrow-precision end to end
    P = 128
    w2_element = Element.(randn(Float32, M, P))
    w2_scale = Scale.(rand(Float32, M ÷ block, P) / √M)
    W2 = BlockscaledArray(sm1xx(CuArray(w2_scale)), CuArray(w2_element))
    Y = jit((d, w) -> d' * w, D, W2)
    hq = Float32.(Array(copy(D)))   # layer 2 consumes the quantized activation
    @test Array(Y) ≈ hq' * dequant_ref(w2_element, w2_scale, block) rtol=1e-2 atol=1e-2

    # standalone quantize (input, nothing to fuse with) is engine inventory;
    # skip rather than claim
    xd = CuArray(randn(Float32, M, N))
    cs = compile_blockscale((d, x) -> Stiletto.quantize!(d, x), D, xd)
    if cs === nothing
        @test_skip blockscale_claimed
    else
        cs(D, xd)
        @test Float32.(Array(copy(D))) ≈ Array(xd) rtol=0.15 atol=0.15
    end

    # contracts surface at trace time
    @test_throws ArgumentError compile((d, w, x) -> Stiletto.quantize!(d, w' * x),
                                       CuArray(zeros(Float32, M, N)), W, X)
    @test_throws ArgumentError compile(
        (d, w, x) -> (q = w' * x; Stiletto.quantize!(d, q); Stiletto.quantize!(d, q)),
        D, W, X)
    @test_throws DimensionMismatch compile((d, w, x) -> Stiletto.quantize!(d, x' * w),
                                           D, W, X)
end

end
