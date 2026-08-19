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
    block_scale_dequantize!, build!, execute!, is_supported,
    CUDNN_TENSOR_REORDERING_F8_128x4, CUDNN_DATA_INT32, CUDNN_DATA_BOOLEAN,
    CUDNN_POINTWISE_GEN_INDEX, CUDNN_POINTWISE_CMP_GE,
    CUDNN_POINTWISE_BINARY_SELECT, CUDNN_REDUCE_TENSOR_AMAX

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

    # vector outputs: every non-blocked extent is 1 — the case where cuDNN's
    # auto-created scale tensor finds no partner axis and skips the
    # 128-padding; declaring the scale from the destination's padded tile
    # extents keeps it a whole (4, 128) tile
    Kv, Mv = 32, 32
    wv_element = Element.(randn(Float32, Kv, Mv))
    wv_scale = Scale.(rand(Float32, Kv ÷ block, Mv) / √Kv)
    xv_element = Element.(randn(Float32, Kv, 1))
    xv_scale = Scale.(rand(Float32, Kv ÷ block, 1) / √Kv)
    Wv = BlockscaledArray(f8_4x128(CuArray(wv_scale)), CuArray(wv_element))
    Xv = BlockscaledArray(f8_4x128(CuArray(xv_scale)), CuArray(xv_element))
    Dv = BlockscaledArray(f8_4x128(CuArray(rand(Scale, Mv ÷ block, 1))),
                          CuArray{Element}(undef, Mv, 1))
    cv = compile_blockscale((d, w, x) -> Stiletto.quantize!(d, w' * x), Dv, Wv, Xv)
    if cv === nothing
        @test_skip blockscale_claimed
    else
        cv(Dv, Wv, Xv)
        yv_ref = dequant_ref(wv_element, wv_scale, block)' *
                 dequant_ref(xv_element, xv_scale, block)
        @test Float32.(Array(copy(Dv))) ≈ yv_ref rtol=0.15 atol=0.15
    end

    # contracts surface at trace time
    @test_throws ArgumentError compile((d, w, x) -> Stiletto.quantize!(d, w' * x),
                                       CuArray(zeros(Float32, M, N)), W, X)
    @test_throws ArgumentError compile(   # destination scales must be swizzled
        (d, w, x) -> Stiletto.quantize!(d, w' * x),
        BlockscaledArray(CuArray(rand(Scale, M ÷ block, N)), CuArray{Element}(undef, M, N)),
        W, X)
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
    # cudnn-frontend's MXFP8 SDPA recipe, mirrored op for op from its
    # CompositeSDPANode expansion and execution-verified on B200 (sm_100,
    # 9.24, dense + causal): Q and V dequantize from natural BHSD
    # declarations, K enters pre-transposed (dims/strides swapped, SF_K's
    # declaration swapped with it), softmax from primitives with LSE stats,
    # P narrowed to E4M3 by typing the div output (bmm2's aType must be the
    # io dtype; the MX recipe has no fixed-scale multiply), BF16 O with
    # fused AMAX. Scale tiles pad semantically — the blocked axis's scale
    # count to a multiple of 4, its partner axis to 128 — so SF_V, which
    # blocks the sequence axis, needs cuDNN's generalized tile check.
    # NVIDIA gates the recipe to cuDNN ≥ 9.21 on CC 10.x; GB10 builds the
    # graphs but has no engine.
    Scale, Element, blk = Float8_E8M0FNU, Float8_E4M3FN, 32
    d, hh, s, b = 128, 4, 256, 2
    ds, dsh = d * s, d * s * hh
    dsp = cld(cld(d, blk), 4) * 4     # Q/K d-blocks padded to 4
    sp = cld(s, 128) * 128            # Q/K partner rows padded to 128
    svp = cld(cld(s, blk), 4) * 4     # V s-blocks padded to 4
    dp = cld(d, 128) * 128            # V partner axis padded to 128
    attn_scale = inv(sqrt(Float32(d)))

    function build_fwd(causal)
        g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
        q = tensor!(g; dims=(d, s, hh, b), dtype=Element, name="Q")
        sfq = tensor!(g; dims=(dsp, sp, hh, b), dtype=Scale, name="SF_Q",
                      reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
        dq = block_scale_dequantize!(g, q, sfq; block_size=blk, name="Qdq")
        kt = tensor!(g; dims=(s, d, hh, b), strides=(d, 1, ds, dsh),
                     dtype=Element, name="KT")
        sfk = tensor!(g; dims=(sp, dsp, hh, b),
                      strides=(dsp, 1, dsp * sp, dsp * sp * hh),
                      dtype=Scale, name="SF_K",
                      reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
        dk = block_scale_dequantize!(g, kt, sfk; block_size=blk, name="Kdq")
        v = tensor!(g; dims=(d, s, hh, b), dtype=Element, name="V")
        sfv = tensor!(g; dims=(dp, svp, hh, b), dtype=Scale, name="SF_V",
                      reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
        dv = block_scale_dequantize!(g, v, sfv; block_size=blk, name="Vdq")

        S = matmul!(g, dk, dq; c=tensor!(g; dims=(s, s, hh, b), dtype=nothing,
                                         virtual=true, name="S"))
        scale = scalar!(g, Float32; rank=4, name="AttnScale")
        ss = pointwise!(g, :mul, S, scale; name="mul_attn_scale")
        ninf = nothing
        if causal
            ninf = scalar!(g, Float32; rank=4, name="MaskValue")
            row = tensor!(g; dims=(s, s, hh, b), dtype=CUDNN_DATA_INT32,
                          virtual=true, name="row_idx")
            col = tensor!(g; dims=(s, s, hh, b), dtype=CUDNN_DATA_INT32,
                          virtual=true, name="col_idx")
            pointwise!(g, CUDNN_POINTWISE_GEN_INDEX, ss; y=row, axis=2,
                       compute_dtype=CUDNN_DATA_INT32, name="gen_row_idx")
            pointwise!(g, CUDNN_POINTWISE_GEN_INDEX, ss; y=col, axis=3,
                       compute_dtype=CUDNN_DATA_INT32, name="gen_col_idx")
            mask = tensor!(g; dims=(s, s, hh, b), dtype=CUDNN_DATA_BOOLEAN,
                           virtual=true, name="causal_mask")
            pointwise!(g, CUDNN_POINTWISE_CMP_GE, row, col; y=mask,
                       compute_dtype=CUDNN_DATA_INT32, name="row_ge_col")
            ss = pointwise!(g, CUDNN_POINTWISE_BINARY_SELECT, ss, ninf, mask,
                            name="select_causal")
        end
        m = reduction!(g, :max, ss; dims=1, name="row_max")
        ex = pointwise!(g, :exp, pointwise!(g, :sub, ss, m; name="sub_s_max");
                        name="exp_s")
        z = reduction!(g, :add, ex; dims=1, name="row_sum")
        p8 = tensor!(g; dims=(s, s, hh, b), dtype=Element, virtual=true,
                     name="P8")
        pointwise!(g, :div, ex, z; y=p8, name="div_p")
        stats = tensor!(g; dims=(1, s, hh, b), dtype=Float32, output=true,
                        name="Stats")
        pointwise!(g, :add, m, pointwise!(g, :log, z; name="log_sum");
                   y=stats, name="add_stats")
        o = tensor!(g; dims=(d, s, hh, b), dtype=BFloat16, output=true,
                    name="O")
        matmul!(g, dv, p8; c=o)
        amax = tensor!(g; dims=(1, 1, 1, 1), dtype=Float32, output=true,
                       name="AmaxO")
        reduction!(g, CUDNN_REDUCE_TENSOR_AMAX, o; y=amax, dims=(1, 2, 3, 4),
                   name="amax_o")
        return (; g, q, sfq, kt, sfk, v, sfv, scale, ninf, stats, o, amax)
    end

    cc = CUDACore.capability(CUDACore.device())
    mxfp8_sdpa_claimed = cc.major == 10 && cuDNN.version() >= v"9.21"
    @testset "causal=$causal" for causal in (false, true)
        t = build_fwd(causal)   # must construct everywhere (semantic tiles)
        if !is_supported(t.g)
            @test !mxfp8_sdpa_claimed   # vendor-documented on CC 10: loud
            continue
        end
        @test mxfp8_sdpa_claimed
        build!(t.g)
        qeh, keh, veh = (Element.(randn(Float32, d, s, hh, b) / 4) for _ in 1:3)
        qsh, ksh = (Scale.(exp2.(rand(-2:2, d ÷ blk, s, hh, b))) for _ in 1:2)
        vsh = Scale.(exp2.(rand(-2:2, d, s ÷ blk, hh, b)))
        Oa = CuArray(zeros(BFloat16, d, s, hh, b))
        St = CuArray(zeros(Float32, 1, s, hh, b))
        Am = CuArray(zeros(Float32, 1, 1, 1, 1))
        binds = Dict{cuDNN.Tensor,Any}(
            t.q => CuArray(qeh), t.sfq => f8_4x128(CuArray(qsh)),
            t.kt => CuArray(keh), t.sfk => f8_4x128(CuArray(ksh)),
            t.v => CuArray(veh), t.sfv => f8_4x128(CuArray(vsh)),
            t.scale => attn_scale, t.stats => St, t.o => Oa, t.amax => Am)
        t.ninf === nothing || (binds[t.ninf] = -Inf32)
        execute!(t.g, binds)

        # plain-softmax reference with the P quantization emulated; O error
        # sits in the fp8-P band, stats are computed pre-quantization and
        # must be tight
        Oref = zeros(Float32, d, s, hh, b)
        for bi in 1:b, hi in 1:hh
            Qm = dequant_ref(qeh[:, :, hi, bi], qsh[:, :, hi, bi], blk)
            Km = dequant_ref(keh[:, :, hi, bi], ksh[:, :, hi, bi], blk)
            Vm = Float32.(veh[:, :, hi, bi]) .*
                 repeat(Float32.(vsh[:, :, hi, bi]); inner=(1, blk))
            Sm = (Km' * Qm) .* attn_scale                        # (s_kv, s_q)
            causal && (Sm .= [kk <= qq ? Sm[kk, qq] : -Inf32
                              for kk in 1:s, qq in 1:s])
            mx = maximum(Sm; dims=1)
            P = exp.(Sm .- mx)
            z = sum(P; dims=1)
            Pq = Float32.(Element.(P ./ z))
            Oref[:, :, hi, bi] = Vm * Pq
            @test Float32.(Array(@view Oa[:, :, hi, bi])) ≈
                  Oref[:, :, hi, bi] rtol=6e-2 atol=6e-2
            @test vec(Array(St)[1, :, hi, bi]) ≈ vec(mx .+ log.(z)) atol=2e-3
        end
        @test Array(Am)[1] ≈ maximum(abs, Oref) rtol=5e-2
    end
end
