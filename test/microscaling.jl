# Block-scaled (MXFP8/NVFP4) arguments: a BlockscaledArray traced like any
# array declares element + swizzled-scale tensors joined by a dequantize node,
# and binds its storage components at execution. cuDNN's block-scale recipes
# want the blocked (reduction) dimension innermost in storage, so quantized
# gemm operands are stored (K, ·) and the left operand is transposed in the
# traced code.

using Microscaling: BlockscaledArray, F8_4x128Array, f8_4x128, elements, scales,
    Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN
using BitPacking: NarrowArray
using cuDNN: Graph, tensor!, scalar!, output!, matmul!, pointwise!, reduction!,
    block_scale_dequantize!, execute!, is_supported,
    CUDNN_TENSOR_REORDERING_F8_128x4

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

        W = BlockscaledArray(f8_4x128(CuArray(w_scale)), gpu_elements(w_element))
        X = BlockscaledArray(f8_4x128(CuArray(x_scale)), gpu_elements(x_element))

        c = compile_blockscale((w, x) -> w' * x, W, X)
        @test (c !== nothing) == blockscale_claimed
        c === nothing && continue
        @test Array(c(W, X)) ≈ C_ref rtol=1e-4 atol=1e-4

        # a compiled graph rebinds to new storage of the same shapes
        w_element2 = ElemW.(randn(Float32, K, M))
        W2 = BlockscaledArray(f8_4x128(CuArray(w_scale)), gpu_elements(w_element2))
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

    W = BlockscaledArray(f8_4x128(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(f8_4x128(CuArray(x_scale)), CuArray(x_element))

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

    W = BlockscaledArray(f8_4x128(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(f8_4x128(CuArray(x_scale)), CuArray(x_element))
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

    W = BlockscaledArray(f8_4x128(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(f8_4x128(CuArray(x_scale)), CuArray(x_element))
    D = BlockscaledArray(
        F8_4x128Array(CuArray{Scale}(undef, 4, 4, 32, (M ÷ block) ÷ 4, N ÷ 128)),
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
    W2 = BlockscaledArray(f8_4x128(CuArray(w2_scale)), CuArray(w2_element))
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

@testset "block-scaled transformer block" begin
    # the serving composition: fp8 QKV projections (quantized weights ×
    # quantized activations), fp16 attention, dense out-projection — chained
    # jits sharing stable buffers, one plan per distinct signature
    Scale, Element, blk = Float8_E8M0FNU, Float8_E4M3FN, 32
    dh, hh, s, b = 64, 4, 32, 4   # s·b ≥ 128: the scale swizzle tiles 128 wide
    D, N = dh * hh, s * b
    quantized(dims...) = BlockscaledArray(
        f8_4x128(CuArray(Scale.(rand(Float32, dims[1] ÷ blk, dims[2:end]...) / √D))),
        CuArray(Element.(randn(Float32, dims...))))
    Wq, Wk, Wv, X = quantized(D, D), quantized(D, D), quantized(D, D), quantized(D, N)
    Wo = CuArray(randn(Float16, D, D) ./ √Float16(D))
    qb, kb, vb, y = (CuArray(zeros(Float16, D, N)) for _ in 1:4)
    O = CuArray(zeros(Float16, dh, hh, s, b))

    # the SwizzledArray broadcast linearizes the swizzled scales to logical shape
    dq(A) = dequant_ref(Array(elements(A)), Array(Float32.(scales(A))), blk)

    proj!(d, w, x) = (d .= w' * x; nothing)
    out!(y, w, o) = (mul!(y, w', o); nothing)
    heads(a) = reshape(a, dh, hh, s, b)
    function block!()
        jit(proj!, qb, Wq, X); jit(proj!, kb, Wk, X); jit(proj!, vb, Wv, X)
        Stiletto.attention!(O, heads(qb), heads(kb), heads(vb))
        jit(out!, y, Wo, reshape(O, D, N))
        return nothing
    end
    nplans = length(cuDNN.handle().plans)
    block!()
    # three projections share one plan (same signature, rebinding)
    @test length(cuDNN.handle().plans) <= nplans + 3

    Wq32, Wk32, Wv32, X32 = dq(Wq), dq(Wk), dq(Wv), dq(X)
    o_ref = ref_attention(reshape(Wq32' * X32, dh, hh, s, b),
                          reshape(Wk32' * X32, dh, hh, s, b),
                          reshape(Wv32' * X32, dh, hh, s, b);
                          scale=inv(sqrt(Float32(dh))))
    y_ref = Float32.(Array(Wo))' * reshape(o_ref, D, N)
    @test Float32.(Array(y)) ≈ y_ref rtol=5e-2 atol=5e-2

    # the whole block replays under CUDA graph capture: quantized matmuls,
    # attention, and dense matmul are all just stream work
    graph = CUDACore.capture(throw_error=false) do
        block!()
    end
    if graph === nothing
        @test_skip false
    else
        exec = CUDACore.instantiate(graph)
        fill!(y, Float16(0))
        CUDACore.launch(exec)
        CUDACore.synchronize()
        @test Float32.(Array(y)) ≈ y_ref rtol=5e-2 atol=5e-2
    end

    # note: block-scaled attention inputs (cuDNN's MXFP8 attention recipe)
    # are constructible — BHSD storage puts the sequence axis on the swizzle's
    # 128-tile dimension, presented head-major — and dequant→SDPA graphs
    # finalize with those strides, but no engine takes them on sm_121/9.24
    # in any mode probed; the frontend recipe lowers to a composed fMHA
    # graph these bindings don't build. Quantized attention inputs go
    # through the projection chain instead. (See TODO.md for the probe
    # record; the composed graph itself is tested below.)
end

end

@testset "MXFP8 attention (composed fMHA)" begin
    # cudnn-frontend's MXFP8 SDPA recipe, mirrored op for op from its node
    # expansion: block-scale-dequantize Q/K/V before the BMMs, composed
    # softmax, P quantized at fixed scale 256 by typing the pointwise output
    # E4M3 (cast-by-dtype), no post-descale. NVIDIA gates the recipe to
    # cuDNN ≥ 9.21 on Blackwell Data Center (CC 10.x) — GB10 (CC 12.x) is
    # excluded — so support is CLAIMED there and the numerics have never run
    # anywhere yet: the first execution on CC 10 hardware is the arbiter.
    Scale, Element, blk = Float8_E8M0FNU, Float8_E4M3FN, 32
    d, hh, s, b = 128, 4, 256, 2      # tile-exact: d/32 = 4, s multiple of 128
    ds, dsh = d * s, d * s * hh
    sc = d ÷ blk

    # V's column-wise scales need the generalized check_swizzled_scale_dims
    # (branch cudnn-blockscale-check); broken until the resolved cuDNN has it
    built = try
    g = Graph(io_dtype=Float16, intermediate_dtype=Float32, compute_dtype=Float32)
    # BHSD storage (d, s, h, b); Q and V presented (s, d, h, b), K natural
    qe = tensor!(g; dims=(s, d, hh, b), strides=(d, 1, ds, dsh), dtype=Element, name="Q")
    qs = tensor!(g; dims=(s, sc, hh, b), strides=(sc, 1, sc * s, sc * s * hh),
                 dtype=Scale, name="Q.scale", reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    ke = tensor!(g; dims=(d, s, hh, b), strides=(1, d, ds, dsh), dtype=Element, name="K")
    ks = tensor!(g; dims=(sc, s, hh, b), strides=(1, sc, sc * s, sc * s * hh),
                 dtype=Scale, name="K.scale", reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    ve = tensor!(g; dims=(s, d, hh, b), strides=(d, 1, ds, dsh), dtype=Element, name="V")
    # V blocks the sequence dim; tile roles are positional (fastest dim pads
    # to 4, second-fastest to 128), so the s÷32 = 8 scale rows pad to 128
    vs = tensor!(g; dims=(128, d, hh, b), strides=(d, 1, d * 128, d * 128 * hh),
                 dtype=Scale, name="V.scale", reordering=CUDNN_TENSOR_REORDERING_F8_128x4)

    qd = block_scale_dequantize!(g, qe, qs; block_size=blk, name="Qd")
    kd = block_scale_dequantize!(g, ke, ks; block_size=blk, name="Kd")
    vd = block_scale_dequantize!(g, ve, vs; block_size=blk, name="Vd")
    S = matmul!(g, qd, kd; name="bmm1")
    attn_scale = scalar!(g, Float32; rank=4, name="AttnScale")
    Ss = pointwise!(g, :mul, S, attn_scale; name="scaled")
    m = reduction!(g, :max, Ss; dims=2, name="rowmax")
    e = pointwise!(g, :exp, pointwise!(g, :sub, Ss, m; name="centered"); name="expS")
    z = reduction!(g, :add, e; dims=2, name="rowsum")
    P = pointwise!(g, :div, e, z; name="softmax")
    scale_s = scalar!(g, Float32; rank=4, name="ScaleS")
    Pq = tensor!(g; dims=P.dims, dtype=Element, virtual=true, name="Pq")
    pointwise!(g, :mul, P, scale_s; y=Pq)
    O = tensor!(g; dims=(s, d, hh, b), dtype=Float16, output=true, name="O")
    matmul!(g, Pq, vd; c=O)
    (; g, qe, qs, ke, ks, ve, vs, attn_scale, scale_s, O)
    catch err
        err isa DimensionMismatch || rethrow()
        nothing
    end

    cc = CUDACore.capability(CUDACore.device())
    mxfp8_sdpa_claimed = cc.major == 10 && cuDNN.version() >= v"9.21"
    if built === nothing
        @test_broken false
    else
    (; g, qe, qs, ke, ks, ve, vs, attn_scale, scale_s, O) = built
    if !is_supported(g)
        @test !mxfp8_sdpa_claimed   # vendor-documented on CC 10: fail loudly there
        @test_skip mxfp8_sdpa_claimed
    else
        qeh, keh, veh = (Element.(randn(Float32, d, s, hh, b)) for _ in 1:3)
        qsh, ksh = (Scale.(rand(Float32, sc, s, hh, b) / √d) for _ in 1:2)
        vsh = Scale.(rand(Float32, d, s ÷ blk, hh, b) / √d)  # natural (4-axis, 128-axis)
        Oa = CuArray(zeros(Float16, s, d, hh, b))
        execute!(g, qe => CuArray(qeh), qs => f8_4x128(CuArray(qsh)),
                    ke => CuArray(keh), ks => f8_4x128(CuArray(ksh)),
                    ve => CuArray(veh), vs => f8_4x128(CuArray(vsh)),  # s-blocks auto-pad to a whole tile
                    attn_scale => inv(sqrt(Float32(d))), scale_s => 256f0, O => Oa)
        # reference with the fixed-256 fp8 P model
        dq4(el, sf, bdim) = Float32.(el) .*
            repeat(Float32.(sf); inner=ntuple(i -> i == bdim ? blk : 1, 4))
        Qd = dq4(qeh, qsh, 1)
        Kd = dq4(keh, ksh, 1)
        Vd = dq4(veh, vsh, 2)
        for bi in 1:b, hi in 1:hh
            Sm = inv(sqrt(Float32(d))) .* (Qd[:, :, hi, bi]' * Kd[:, :, hi, bi])
            Pm = mapslices(r -> (w = exp.(r .- maximum(r)); w ./ sum(w)), Sm; dims=2)
            Pf = Float32.(Element.(256f0 .* Pm))
            @test Float32.(Array(@view Oa[:, :, hi, bi])) ≈
                  Pf * Vd[:, :, hi, bi]' rtol=6e-2 atol=6e-2
        end
    end
    end
end
