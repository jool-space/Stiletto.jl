using Test
using Stiletto
using CUDACore: CUDACore, CuArray, allowscalar
import cuDNN
using BFloat16s: BFloat16
using LinearAlgebra: mul!
using NNlib: NNlib, relu, sigmoid, gelu, swish, softplus, elu
using Stiletto: ⊠
using SpecialFunctions: erf
using Statistics: mean

allowscalar(false)

if !cuDNN.functional()
    @info "cuDNN not functional; skipping Stiletto tests"
    exit()
end

struct Dense{W}
    w::W
end
(d::Dense)(x) = d.w * x

# authoritative norm references: the library computes these via cuDNN graphs
function ref_spans(x, s)
    rs = reshape(s, size(s)..., ntuple(_ -> 1, ndims(x) - ndims(s))...)
    return rs, Tuple(i for i in 1:ndims(x) if size(rs, i) != 1)
end
function ref_rmsnorm(x, s; bias=nothing, eps=1f-5)
    rs, dims = ref_spans(x, s)
    y = rs .* x ./ sqrt.(mean(abs2, x; dims) .+ eps)
    return bias === nothing ? y : y .+ reshape(bias, size(rs))
end
function ref_layernorm(x, s, b; eps=1f-5)
    rs, dims = ref_spans(x, s)
    mu = mean(x; dims)
    return rs .* (x .- mu) ./ sqrt.(mean(abs2, x .- mu; dims) .+ eps) .+ reshape(b, size(rs))
end
function ref_batchnorm(x, s, b, m, iv)
    cd = ntuple(i -> i == ndims(x) - 1 ? size(x, ndims(x) - 1) : 1, ndims(x))
    return reshape(s, cd) .* (x .- reshape(m, cd)) .* reshape(iv, cd) .+ reshape(b, cd)
end
function ref_attention(q, k, v; scale, causal=false)  # (d, h, s, b); causal assumes sq == skv
    d, h, sq, b = size(q)
    skv, hk = size(k, 3), size(k, 2)
    q32, k32, v32 = Float32.(q), Float32.(k), Float32.(v)
    o = zeros(Float32, d, h, sq, b)
    for bi in 1:b, hi in 1:h, i in 1:sq
        hkv = fld1(hi, h ÷ hk)
        s = [scale * sum(@view(q32[:, hi, i, bi]) .* @view(k32[:, hkv, j, bi]))
             for j in 1:skv]
        causal && (s[i+1:end] .= -Inf32)
        p = exp.(s .- maximum(s)); p ./= sum(p)
        o[:, hi, i, bi] = @view(v32[:, hkv, :, bi]) * p
    end
    return o
end

function ref_conv(x, w; stride=1, dilation=1, pre_padding=0, post_padding=pre_padding)
    rank = ndims(x) - 2
    tup(v) = v isa Integer ? ntuple(_ -> Int(v), rank) : Tuple(Int.(v))
    pre, post, str, dil = tup(pre_padding), tup(post_padding), tup(stride), tup(dilation)
    cw, cout, N = size(w, rank + 1), size(w, rank + 2), size(x, rank + 2)
    groups = size(x, rank + 1) ÷ cw
    outsp = ntuple(i -> 1 + (size(x, i) + pre[i] + post[i] - dil[i] * (size(w, i) - 1) - 1) ÷ str[i],
                   rank)
    y = zeros(Float32, outsp..., cout, N)
    for n in 1:N, o in 1:cout, oi in CartesianIndices(outsp),
        ki in CartesianIndices(size(w)[1:rank])
        xi = ntuple(i -> (oi[i] - 1) * str[i] + (ki[i] - 1) * dil[i] + 1 - pre[i], rank)
        all(1 .<= xi .<= size(x)[1:rank]) || continue
        gr = fld1(o, cout ÷ groups)
        for ci in 1:cw
            y[oi, o, n] += Float32(x[xi..., (gr - 1) * cw + ci, n]) * Float32(w[ki, ci, o])
        end
    end
    return y
end

function ref_pool(mode, x, window; stride=window, pre_padding=0, post_padding=pre_padding)
    rank = ndims(x) - 2
    tup(v) = v isa Integer ? ntuple(_ -> Int(v), rank) : Tuple(Int.(v))
    win, str, pre, post = tup(window), tup(stride), tup(pre_padding), tup(post_padding)
    outsp = ntuple(i -> fld(size(x, i) + pre[i] + post[i] - win[i], str[i]) + 1, rank)
    y = zeros(Float32, outsp..., size(x, rank + 1), size(x, rank + 2))
    for n in axes(y, rank + 2), c in axes(y, rank + 1), oi in CartesianIndices(outsp)
        vals = Float32[]
        for ki in CartesianIndices(win)
            xi = ntuple(i -> (oi[i] - 1) * str[i] + ki[i] - pre[i], rank)
            all(1 .<= xi .<= size(x)[1:rank]) && push!(vals, Float32(x[xi..., c, n]))
        end
        y[oi, c, n] = mode === :maxpool ? maximum(vals) :
                      mode === :include ? sum(vals) / prod(win) : sum(vals) / length(vals)
    end
    return y
end

@testset "Stiletto" begin

@testset "matmul" begin
    A, B = CuArray(rand(Float32, 64, 48)), CuArray(rand(Float32, 48, 32))
    c = compile((a, b) -> a * b, A, B)
    @test Array(c(A, B)) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2
    # a compiled graph rebinds to new arrays of the same shapes
    A2, B2 = CuArray(rand(Float32, 64, 48)), CuArray(rand(Float32, 48, 32))
    @test Array(c(A2, B2)) ≈ Array(A2) * Array(B2) rtol=1e-2 atol=1e-2

    # outputs come from the allocator, which the caller can replace
    allocations = 0
    counting = (T, dims) -> (allocations += 1; CuArray{T}(undef, dims))
    ca = compile((a, b) -> a * b, A, B; allocator=counting)
    ca(A, B)
    @test allocations == 1
end

@testset "engine-selection knobs" begin
    A, B = CuArray(rand(Float32, 64, 48)), CuArray(rand(Float32, 48, 32))
    ref = Array(A) * Array(B)

    # all measured supported on sm_121 / cuDNN 9.24: deterministic engines
    # exist for matmul, and capping the workspace steers selection to a
    # workspace-free plan (the unconstrained pick demands ~32 MB)
    cd = compile((a, b) -> a * b, A, B; deterministic=true)
    @test Array(cd(A, B)) ≈ ref rtol=1e-2 atol=1e-2

    cw = compile((a, b) -> a * b, A, B; max_workspace=0)
    @test cw.graph.workspace_size == 0
    @test Array(cw(A, B)) ≈ ref rtol=1e-2 atol=1e-2

    ch = compile((a, b) -> a * b, A, B;
                 heuristics=(cuDNN.CUDNN_HEUR_MODE_FALLBACK,))
    @test Array(ch(A, B)) ≈ ref rtol=1e-2 atol=1e-2

    # knobs participate in jit's plan key: same signature, different knobs,
    # distinct cached plans
    f = (a, b) -> a * b
    n0 = length(cuDNN.handle().plans)
    jit(f, A, B)
    jit(f, A, B; max_workspace=0)
    @test length(cuDNN.handle().plans) == n0 + 2
    jit(f, A, B; max_workspace=0)   # cached, no new plan
    @test length(cuDNN.handle().plans) == n0 + 2
end

@testset "matmul epilogue" begin
    A, B = CuArray(rand(Float32, 64, 64)), CuArray(rand(Float32, 64, 64))
    c = compile((a, b) -> max.(a * b .- 1f0, 0f0), A, B)
    @test Array(c(A, B)) ≈ max.(Array(A) * Array(B) .- 1f0, 0f0) rtol=1e-2 atol=1e-2
end

@testset "half io" begin
    # by default outputs keep Julia's promotion: a pure-half chain returns
    # half, and a Float32 scalar promotes the result exactly as Base would
    A16 = CuArray(rand(Float16, 32, 16) ./ 4)
    B16 = CuArray(rand(Float16, 16, 24) ./ 4)
    ref = Float32.(Array(A16)) * Float32.(Array(B16))

    cb = compile((a, b) -> a * b, A16, B16)
    yb = cb(A16, B16)
    @test eltype(yb) == Float16
    @test Array(yb) ≈ ref rtol=1e-2 atol=1e-2

    cp = compile((a, b) -> tanh.(a * b .- 1f0), A16, B16)
    yp = cp(A16, B16)
    @test eltype(yp) == Float32
    @test Array(yp) ≈ tanh.(ref .- 1f0) rtol=1e-2 atol=1e-2

    # explicit io_dtype is a precision policy: non-cast outputs follow it,
    # scalar promotion notwithstanding
    c = compile((a, b) -> tanh.(a * b .- 1f0), A16, B16; io_dtype=Float16)
    y = c(A16, B16)
    @test eltype(y) == Float16
    @test Array(y) ≈ tanh.(ref .- 1f0) rtol=1e-2 atol=1e-2
end

@testset "pointwise chain" begin
    X = CuArray(rand(Float32, 32, 32))
    c = compile(x -> exp.(.-(2f0 .* x)), X)
    @test Array(c(X)) ≈ exp.(-2f0 .* Array(X)) rtol=1e-5

    # scalar subtrees fold at trace time; only their value reaches the graph
    c = compile(x -> x .+ sqrt.(4f0) .* (1 + 1), X)
    @test Array(c(X)) ≈ Array(X) .+ 4f0 rtol=1e-6
end

@testset "transposed operand" begin
    A, B = CuArray(rand(Float32, 48, 64)), CuArray(rand(Float32, 48, 32))
    c = compile((a, b) -> a' * b, A, B)
    @test Array(c(A, B)) ≈ Array(A)' * Array(B) rtol=1e-2 atol=1e-2
end

@testset "batched matmul (⊠)" begin
    # torch-style batched matmul: any number of trailing batch dims,
    # broadcasting pairwise, missing dims counting as 1. The trace accepts
    # the general form; engine inventory decides what executes — sm_121 /
    # cuDNN 9.24 has matmul engines only at rank 3 (one batch dim).
    function ref_bmm(a, b)
        N = max(ndims(a), ndims(b))
        batch = ntuple(i -> max(size(a, i + 2), size(b, i + 2)), N - 2)
        c = zeros(Float32, size(a, 1), size(b, 2), batch...)
        for I in CartesianIndices(batch)
            ia = CartesianIndex(ntuple(i -> min(I[i], size(a, i + 2)), N - 2))
            ib = CartesianIndex(ntuple(i -> min(I[i], size(b, i + 2)), N - 2))
            c[:, :, I] = Float32.(a[:, :, ia]) * Float32.(b[:, :, ib])
        end
        return c
    end

    # `*` and `mul!` keep Base's 2-D semantics; batching is spelled ⊠
    @test_throws MethodError compile((a, b) -> a * b,
        CuArray(rand(Float32, 8, 16, 3)), CuArray(rand(Float32, 16, 12, 3)))

    # one batch dim, equal extents
    A, B = CuArray(rand(Float32, 32, 16, 4)), CuArray(rand(Float32, 16, 24, 4))
    c = compile((a, b) -> a ⊠ b, A, B)
    @test Array(c(A, B)) ≈ ref_bmm(Array(A), Array(B)) rtol=1e-2 atol=1e-2

    # missing batch dims count as 1: shared weights against a batch
    W, X = CuArray(rand(Float32, 32, 16)), CuArray(rand(Float32, 16, 24, 4))
    cw = compile((w, x) -> w ⊠ x, W, X)
    @test Array(cw(W, X)) ≈ ref_bmm(Array(W), Array(X)) rtol=1e-2 atol=1e-2

    # eager tier allocates and jits; mutating tier writes the destination
    @test Array(Stiletto.batched_mul(A, B)) ≈ ref_bmm(Array(A), Array(B)) rtol=1e-2 atol=1e-2
    C = CuArray(zeros(Float32, 32, 24, 4))
    @test Stiletto.batched_mul!(C, A, B) === C
    @test Array(C) ≈ ref_bmm(Array(A), Array(B)) rtol=1e-2 atol=1e-2

    # traced mutating form
    fill!(C, 0f0)
    cm = compile((c, a, b) -> (Stiletto.batched_mul!(c, a, b); nothing), C, A, B)
    cm(C, A, B)
    @test Array(C) ≈ ref_bmm(Array(A), Array(B)) rtol=1e-2 atol=1e-2

    # α/β epilogue on the batched verb: β = 0 never reads the destination
    # (eager tier, NaN-proof); β = 1 accumulates through the NNlib method
    fill!(C, NaN32)
    @test Stiletto.batched_mul!(C, A, B, 2f0, 0f0) === C
    @test Array(C) ≈ 2 .* ref_bmm(Array(A), Array(B)) rtol=1e-2 atol=1e-2
    cacc = try
        compile((c, a, b) -> (NNlib.batched_mul!(c, a, b, 1f0, 1f0); nothing), C, A, B)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if cacc === nothing
        @test_skip false
    else
        C0 = copy(Array(C))
        cacc(C, A, B)
        @test Array(C) ≈ ref_bmm(Array(A), Array(B)) .+ C0 rtol=1e-2 atol=1e-2
    end

    # NNlib's verbs route onto the traced matmul, including batched_transpose
    A4, B4 = CuArray(rand(Float32, 16, 8, 4)), CuArray(rand(Float32, 16, 12, 4))
    cn = compile((a, b) -> NNlib.batched_mul(NNlib.batched_transpose(a), b), A4, B4)
    @test Array(cn(A4, B4)) ≈
          ref_bmm(permutedims(Array(A4), (2, 1, 3)), Array(B4)) rtol=1e-2 atol=1e-2

    # multiple batch dims trace and build, but engine inventory currently
    # stops at rank 3; run wherever an engine appears, skip honestly here
    A5 = CuArray(rand(Float32, 8, 16, 3, 1))
    B5 = CuArray(rand(Float32, 16, 12, 1, 5))
    c5 = try
        compile((a, b) -> Stiletto.batched_mul(a, b), A5, B5)
    catch err
        err isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if c5 === nothing
        @test_skip false
    else
        got = c5(A5, B5)
        @test size(got) == (8, 12, 3, 5)
        @test Array(got) ≈ ref_bmm(Array(A5), Array(B5)) rtol=1e-2 atol=1e-2
    end

    # shape contracts surface at trace time
    @test_throws DimensionMismatch compile((a, b) -> Stiletto.batched_mul(a, b),
        CuArray(rand(Float32, 8, 15, 3)), CuArray(rand(Float32, 16, 12, 3)))
    @test_throws DimensionMismatch compile((a, b) -> Stiletto.batched_mul(a, b),
        CuArray(rand(Float32, 8, 16, 3)), CuArray(rand(Float32, 16, 12, 4)))
end

@testset "captured array" begin
    W, X = CuArray(rand(Float32, 32, 32)), CuArray(rand(Float32, 32, 8))
    c = compile(x -> W * x, X)
    @test Array(c(X)) ≈ Array(W) * Array(X) rtol=1e-2 atol=1e-2
end

@testset "one input, two contexts" begin
    # the design case: `a` feeds a pointwise prologue at logical rank 2 and a
    # matmul that needs rank 3; rank unification declares both at rank 3
    A, B = CuArray(rand(Float32, 32, 32)), CuArray(rand(Float32, 32, 32))
    c = compile((a, b) -> a .+ a * b, A, B)
    @test Array(c(A, B)) ≈ Array(A) .+ Array(A) * Array(B) rtol=1e-2 atol=1e-2
end

@testset "reduction epilogue" begin
    A, B = CuArray(rand(Float32, 32, 32)), CuArray(rand(Float32, 32, 32))
    c = compile((a, b) -> sum(a * b; dims=2), A, B)
    @test vec(Array(c(A, B))) ≈ vec(sum(Array(A) * Array(B); dims=2)) rtol=1e-2 atol=1e-2
end

@testset "dtype casts" begin
    Xh = randn(BFloat16, 32, 32)
    Yh = randn(Float32, 32, 32)
    X, Y = CuArray(Xh), CuArray(Yh)

    # explicit promotion mid-graph
    c = compile((x, y) -> Float32.(x) .* y, X, Y)
    @test Array(c(X, Y)) ≈ Float32.(Xh) .* Yh rtol=1e-2 atol=1e-2

    # mixed dtypes work without an explicit cast (implicit conversion)
    c = compile((x, y) -> x .* y, X, Y)
    @test Array(c(X, Y)) ≈ Float32.(Xh) .* Yh rtol=1e-2 atol=1e-2

    # demoted output allocates its own eltype
    c = compile((x, y) -> BFloat16.(x .* y), CuArray(Yh), Y)
    out = c(CuArray(Yh), Y)
    @test eltype(out) == BFloat16
    @test Float32.(Array(out)) ≈ Float32.(BFloat16.(Yh .* Yh)) rtol=1e-2 atol=1e-2

    # casting to the current eltype is a no-op, not a graph node
    t = compile((x, y) -> Float32.(x) .+ y, Y, Y)
    @test Array(t(Y, Y)) ≈ Yh .+ Yh rtol=1e-5
end

@testset "norms" begin
    H, N = 64, 32
    xh = randn(Float32, H, N)
    sh = rand(Float32, H) .+ 0.5f0
    bh = randn(Float32, H)
    x, s, b = CuArray(xh), CuArray(sh), CuArray(bh)

    c = compile((x, s) -> Stiletto.rmsnorm(x, s), x, s)
    @test Array(c(x, s)) ≈ ref_rmsnorm(xh, sh) rtol=1e-2 atol=1e-2

    c = compile((x, s, b) -> Stiletto.rmsnorm(x, s; bias=b), x, s, b)
    @test Array(c(x, s, b)) ≈ ref_rmsnorm(xh, sh; bias=bh) rtol=1e-2 atol=1e-2

    c = compile((x, s, b) -> Stiletto.layernorm(x, s, b), x, s, b)
    @test Array(c(x, s, b)) ≈ ref_layernorm(xh, sh, bh) rtol=1e-2 atol=1e-2

    # eager tier: CuArray methods run a cached standalone graph
    @test Array(Stiletto.rmsnorm(x, s)) ≈ ref_rmsnorm(xh, sh) rtol=1e-2 atol=1e-2
    @test Array(Stiletto.layernorm(x, s, b)) ≈ ref_layernorm(xh, sh, bh) rtol=1e-2 atol=1e-2
    nplans = length(cuDNN.handle().plans)
    Stiletto.rmsnorm(x, s)                    # same signature: cache hit
    @test length(cuDNN.handle().plans) == nplans

    # mutating tier writes the caller's buffer and returns it
    y = CuArray(zeros(Float32, H, N))
    @test Stiletto.rmsnorm!(y, x, s) === y
    @test Array(y) ≈ ref_rmsnorm(xh, sh) rtol=1e-2 atol=1e-2
    @test Stiletto.layernorm!(y, x, s, b) === y
    @test Array(y) ≈ ref_layernorm(xh, sh, bh) rtol=1e-2 atol=1e-2

    # mutating form inside a trace
    c = compile((y, x, s) -> (Stiletto.rmsnorm!(y, x, s); nothing), y, x, s)
    c(y, x, s)
    @test Array(y) ≈ ref_rmsnorm(xh, sh) rtol=1e-2 atol=1e-2

    # batch norm (inference): per-channel vectors over a (spatial, C, N) input,
    # with the epsilon fold done once at parameter load
    C = 8
    x4h = randn(Float32, 6, 5, C, 3)
    sc, bi = rand(Float32, C) .+ 0.5f0, randn(Float32, C)
    mu, iv = randn(Float32, C), rand(Float32, C) .+ 0.5f0
    ref = ref_batchnorm(x4h, sc, bi, mu, iv)
    x4 = CuArray(x4h)
    scu, bcu, mcu, ivcu = CuArray.((sc, bi, mu, iv))
    # engine availability is cuDNN's: no engine means UnsupportedGraphError,
    # not a silent unfused substitute
    bn = try
        Stiletto.batchnorm(x4, scu, bcu, mcu, ivcu)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if bn === nothing
        @test_skip false
    else
        @test Array(bn) ≈ ref rtol=1e-3 atol=1e-3
        y4 = CuArray(zeros(Float32, size(x4h)))
        @test Stiletto.batchnorm!(y4, x4, scu, bcu, mcu, ivcu) === y4
        @test Array(y4) ≈ ref rtol=1e-3 atol=1e-3
    end
    # traced tier: captured parameters, and vector parameters as arguments
    # (reshaped onto the channel axis in-trace)
    for (fargs, cargs) in (
        ((x -> Stiletto.batchnorm(x, scu, bcu, mcu, ivcu), x4), (x4,)),
        (((x, s, b, m, v) -> Stiletto.batchnorm(x, s, b, m, v), x4, scu, bcu, mcu, ivcu),
         (x4, scu, bcu, mcu, ivcu)),
    )
        c4 = try
            compile(fargs...)
        catch e
            e isa cuDNN.UnsupportedGraphError || rethrow()
            nothing
        end
        if c4 === nothing
            @test_skip false
        else
            @test Array(c4(cargs...)) ≈ ref rtol=1e-3 atol=1e-3
        end
    end
end

@testset "attention" begin
    d, h, s, b = 64, 4, 32, 2
    qh, kh, vh = (randn(Float16, d, h, s, b) for _ in 1:3)
    Q, K, V = CuArray.((qh, kh, vh))
    scale = inv(sqrt(Float32(d)))
    ref = ref_attention(qh, kh, vh; scale)

    c = compile((q, k, v) -> Stiletto.attention(q, k, v), Q, K, V)
    o = c(Q, K, V)
    @test eltype(o) == Float16   # attention output follows q's dtype
    @test Float32.(Array(o)) ≈ ref rtol=2e-2 atol=2e-2

    # causal mask baked into the graph; custom scale bound by value
    cc = compile((q, k, v) -> Stiletto.attention(q, k, v; causal=true, scale=0.5f0), Q, K, V)
    @test Float32.(Array(cc(Q, K, V))) ≈
          ref_attention(qh, kh, vh; scale=0.5f0, causal=true) rtol=2e-2 atol=2e-2

    # eager tiers: allocating and mutating, cached per signature
    @test Float32.(Array(Stiletto.attention(Q, K, V))) ≈ ref rtol=2e-2 atol=2e-2
    O = CuArray(zeros(Float16, d, h, s, b))
    @test Stiletto.attention!(O, Q, K, V) === O
    @test Float32.(Array(O)) ≈ ref rtol=2e-2 atol=2e-2
    nplans = length(cuDNN.handle().plans)
    Stiletto.attention!(O, Q, K, V)
    @test length(cuDNN.handle().plans) == nplans

    # captured k/v close over the trace like any array
    ck = compile(q -> Stiletto.attention(q, K, V), Q)
    @test Float32.(Array(ck(Q))) ≈ ref rtol=2e-2 atol=2e-2

    # cross-attention: kv sequence length independent of q's
    kx, vx = randn(Float16, d, h, 16, b), randn(Float16, d, h, 16, b)
    cx = compile((q, k, v) -> Stiletto.attention(q, k, v), Q, CuArray(kx), CuArray(vx))
    @test Float32.(Array(cx(Q, CuArray(kx), CuArray(vx)))) ≈
          ref_attention(qh, kx, vx; scale) rtol=2e-2 atol=2e-2

    # grouped-query attention: k/v heads divide q's, output follows q
    kg, vg = randn(Float16, d, 2, s, b), randn(Float16, d, 2, s, b)
    cg = compile((q, k, v) -> Stiletto.attention(q, k, v), Q, CuArray(kg), CuArray(vg))
    @test Float32.(Array(cg(Q, CuArray(kg), CuArray(vg)))) ≈
          ref_attention(qh, kg, vg; scale) rtol=2e-2 atol=2e-2

    # varlen: per-batch valid lengths as runtime Int32 inputs (padding mask);
    # output rows past seq_len_q are undefined, so compare valid regions only
    function check_valid(o, lq, lkv)
        for (bi, (nq, nkv)) in enumerate(zip(lq, lkv))
            @test o[:, :, 1:nq, bi:bi] ≈
                  ref_attention(qh[:, :, 1:nq, bi:bi], kh[:, :, 1:nkv, bi:bi],
                                vh[:, :, 1:nkv, bi:bi]; scale) rtol=2e-2 atol=2e-2
        end
    end
    slq, slkv = CuArray(Int32[20, 32]), CuArray(Int32[24, 16])
    cv = try
        compile((q, k, v, lq, lkv) ->
                    Stiletto.attention(q, k, v; seq_len_q=lq, seq_len_kv=lkv),
                Q, K, V, slq, slkv)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if cv === nothing
        @test_skip false
    else
        check_valid(Float32.(Array(cv(Q, K, V, slq, slkv))), (20, 32), (24, 16))
        # raggedness is data: rebinding lengths reuses the built graph
        check_valid(Float32.(Array(cv(Q, K, V, CuArray(Int32[8, 12]),
                                      CuArray(Int32[8, 12])))), (8, 12), (8, 12))
        # eager tier threads the lengths as runtime arguments
        ov = Stiletto.attention(Q, K, V; seq_len_q=slq, seq_len_kv=slkv)
        check_valid(Float32.(Array(ov)), (20, 32), (24, 16))
    end

    # sequence-length contracts surface at trace time
    @test_throws ArgumentError Stiletto.attention(Q, K, V; seq_len_q=slq)
    @test_throws ArgumentError compile((q, k, v, l) ->
        Stiletto.attention(q, k, v; seq_len_kv=l), Q, K, V, slkv)
    @test_throws ArgumentError compile((q, k, v, lq, lkv) ->
        Stiletto.attention(q, k, v; seq_len_q=lq, seq_len_kv=lkv),
        Q, K, V, CuArray([20, 32]), CuArray([24, 16]))   # Int64: no silent convert
    @test_throws DimensionMismatch compile((q, k, v, lq, lkv) ->
        Stiletto.attention(q, k, v; seq_len_q=lq, seq_len_kv=lkv),
        Q, K, V, CuArray(Int32[20, 32, 7]), slkv)

    # shape contracts surface at trace time
    @test_throws ArgumentError compile((q, k, v) -> Stiletto.attention(q, k, v),
                                       CuArray(randn(Float16, d, h, s)), K, V)
    @test_throws DimensionMismatch compile((q, k, v) -> Stiletto.attention(q, k, v),
                                           Q, CuArray(randn(Float16, d, 3, s, b)),
                                           CuArray(randn(Float16, d, 3, s, b)))
    @test_throws DimensionMismatch compile((q, k, v) -> Stiletto.attention(q, k, v),
                                           Q, CuArray(randn(Float16, d, h, 16, b)), V)
end

@testset "convolution" begin
    xh, wh = randn(Float32, 16, 14, 3, 2), randn(Float32, 3, 3, 3, 8)
    x, w = CuArray(xh), CuArray(wh)
    c = compile((x, w) -> Stiletto.conv(x, w), x, w)
    @test Array(c(x, w)) ≈ ref_conv(xh, wh) rtol=1e-2 atol=1e-2

    # stride, dilation, and asymmetric padding
    kw = (; stride=(2, 1), dilation=(1, 2), pre_padding=(1, 0), post_padding=(2, 1))
    c = compile((x, w) -> Stiletto.conv(x, w; kw...), x, w)
    @test Array(c(x, w)) ≈ ref_conv(xh, wh; kw...) rtol=1e-2 atol=1e-2

    # grouped: 4 input channels through 2 groups onto 6 outputs
    xgh, wgh = randn(Float32, 10, 9, 4, 2), randn(Float32, 3, 3, 2, 6)
    xg, wg = CuArray(xgh), CuArray(wgh)
    c = compile((x, w) -> Stiletto.conv(x, w), xg, wg)
    @test Array(c(xg, wg)) ≈ ref_conv(xgh, wgh) rtol=1e-2 atol=1e-2

    # depthwise causal conv1d: one spatial dim, (time, channels, batch)
    T, D, B, K = 32, 8, 2, 4
    xsh, wdh = randn(Float32, T, D, B), randn(Float32, K, 1, D)
    xs, wd = CuArray(xsh), CuArray(wdh)
    causal = (; pre_padding=K - 1, post_padding=0)
    refc = ref_conv(xsh, wdh; causal...)
    c = compile((x, w) -> Stiletto.conv(x, w; causal...), xs, wd)
    @test size(c(xs, wd)) == (T, D, B)
    @test Array(c(xs, wd)) ≈ refc rtol=1e-2 atol=1e-2

    # channels-first storage presented channel-innermost in-trace: the
    # sequence layout of Vallmo's causal_conv1d, permutation free on inputs
    xcf = CuArray(permutedims(xsh, (2, 1, 3)))
    ccf = compile((x, w) -> Stiletto.conv(permutedims(x, (2, 1, 3)), w; causal...),
                  xcf, wd)
    @test Array(ccf(xcf, wd)) ≈ refc rtol=1e-2 atol=1e-2

    # conv+pointwise fusion is engine inventory (absent on sm_121/9.24):
    # skip rather than claim
    bh = randn(Float32, 1, D, 1)
    ep = try
        compile((x, w, b) -> swish.(Stiletto.conv(x, w; causal...) .+ b),
                xs, wd, CuArray(bh))
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if ep === nothing
        @test_skip false
    else
        @test Array(ep(xs, wd, CuArray(bh))) ≈
              Float32.(swish.(refc .+ bh)) rtol=1e-2 atol=1e-2
    end

    # eager tiers, cached per signature
    @test Array(Stiletto.conv(xs, wd; causal...)) ≈ refc rtol=1e-2 atol=1e-2
    y = CuArray(zeros(Float32, T, D, B))
    @test Stiletto.conv!(y, xs, wd; causal...) === y
    @test Array(y) ≈ refc rtol=1e-2 atol=1e-2
    nplans = length(cuDNN.handle().plans)
    Stiletto.conv!(y, xs, wd; causal...)
    @test length(cuDNN.handle().plans) == nplans

    # shape contracts surface at trace time
    @test_throws DimensionMismatch compile((x, w) -> Stiletto.conv(x, w), xs,
                                           CuArray(randn(Float32, K, 1, 1, D)))
    @test_throws DimensionMismatch compile((x, w) -> Stiletto.conv(x, w), xs,
                                           CuArray(randn(Float32, K, 3, D)))
    @test_throws DimensionMismatch compile((x, w) -> Stiletto.conv(x, w; stride=(1, 2)),
                                           xs, wd)
end

@testset "pooling" begin
    xh = rand(Float32, 8, 9, 3, 2)
    x = CuArray(xh)
    # shape contracts surface at trace time, engine-independent
    @test_throws DimensionMismatch compile(x -> Stiletto.maxpool(x, (2, 3, 2)), x)

    # resample engines are absent on sm_121/cuDNN 9.24 (probed across dtypes
    # and layouts); numerics activate where cuDNN provides them
    cp = try
        compile(x -> Stiletto.maxpool(x, (2, 3)), x)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if cp === nothing
        @test_skip false
    else
        @test Array(cp(x)) ≈ ref_pool(:maxpool, xh, (2, 3)) rtol=1e-6
        # overlapping windows and padding; mean over contributors vs full window
        @test Array(Stiletto.meanpool(x, 3; stride=2, pre_padding=1)) ≈
              ref_pool(:exclude, xh, 3; stride=2, pre_padding=1) rtol=1e-5
        @test Array(Stiletto.meanpool(x, 3; stride=2, pre_padding=1,
                                      include_padding=true)) ≈
              ref_pool(:include, xh, 3; stride=2, pre_padding=1) rtol=1e-5
        y = CuArray(zeros(Float32, 4, 3, 3, 2))
        @test Stiletto.maxpool!(y, x, (2, 3)) === y
        @test Array(y) ≈ ref_pool(:maxpool, xh, (2, 3)) rtol=1e-6
    end
end

@testset "reshape" begin
    v = CuArray(rand(Float32, 16))
    B = CuArray(rand(Float32, 4, 8))
    # reshaped inputs feed ops at their new shape; binding reshapes the buffer
    c = compile((v, b) -> reshape(v, 4, :) * b, v, B)
    @test Array(c(v, B)) ≈ reshape(Array(v), 4, 4) * Array(B) rtol=1e-2 atol=1e-2
    # reshape composes with itself and elides when shapes already match
    c = compile((v, b) -> reshape(reshape(v, 2, 8), 4, 4) * b, v, B)
    @test Array(c(v, B)) ≈ reshape(Array(v), 4, 4) * Array(B) rtol=1e-2 atol=1e-2
    # computed values have a fixed layout
    @test_throws ArgumentError compile((v, b) -> reshape(reshape(v, 4, :) * b, :), v, B)
    @test_throws DimensionMismatch compile((v, b) -> reshape(v, 5, :) * b, v, B)
end

@testset "permuted outputs" begin
    # cuDNN has no permute op, but an output tensor may be declared with
    # permuted strides: the producing op's epilogue write IS the permute
    A, B = CuArray(rand(Float32, 24, 48)), CuArray(rand(Float32, 48, 16))
    c = compile((a, b) -> permutedims(a * b), A, B)
    out = c(A, B)
    @test size(out) == (16, 24)
    @test Array(out) ≈ permutedims(Array(A) * Array(B)) rtol=1e-2 atol=1e-2

    # transpose/adjoint of a computed matrix are the same boundary
    c2 = compile((a, b) -> (a * b)', A, B)
    @test Array(c2(A, B)) ≈ (Array(A) * Array(B))' rtol=1e-2 atol=1e-2

    # batched: transposing the matrix dims keeps the batch outermost
    A3, B3 = CuArray(rand(Float32, 8, 16, 6)), CuArray(rand(Float32, 16, 12, 6))
    C3 = stack([Array(A3)[:, :, i] * Array(B3)[:, :, i] for i in 1:6])
    c3 = compile((a, b) -> permutedims(a ⊠ b, (2, 1, 3)), A3, B3)
    out3 = c3(A3, B3)
    @test size(out3) == (12, 8, 6)
    @test Array(out3) ≈ permutedims(C3, (2, 1, 3)) rtol=1e-2 atol=1e-2

    # batch-innermost output layouts are a silent-corruption hazard: the
    # heuristic accepts them but the kernel writes garbage (probed 2026-08,
    # sm_121/9.24) — encoded broken until an engine computes them correctly
    cb = compile((a, b) -> permutedims(a ⊠ b, (3, 1, 2)), A3, B3)
    @test_broken Array(cb(A3, B3)) ≈ permutedims(C3, (3, 1, 2)) rtol=1e-2 atol=1e-2

    # permuted in-place destination: the caller's buffer takes the layout
    O = CuArray(zeros(Float32, 16, 24))
    c4 = compile((o, a, b) -> (o .= permutedims(a * b); nothing), O, A, B)
    c4(O, A, B)
    @test Array(O) ≈ permutedims(Array(A) * Array(B)) rtol=1e-2 atol=1e-2

    # a pointwise epilogue may produce the permuted output; engines decide
    c5 = try
        compile((a, b) -> permutedims(relu.(a * b)), A, B)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if c5 === nothing
        @test_skip false
    else
        @test Array(c5(A, B)) ≈ permutedims(max.(Array(A) * Array(B), 0f0)) rtol=1e-2 atol=1e-2
    end

    # boundary only: a permuted computed value cannot feed further ops
    @test_throws ArgumentError compile((a, b) -> permutedims(a * b) .+ 1f0, A, B)
    # ... and must be the value's only use
    @test_throws ArgumentError compile((a, b) -> (y = a * b; (y, permutedims(y))), A, B)
    @test_throws ArgumentError compile((a, b) -> (y = a * b; (permutedims(y), y)), A, B)
    # a value written in place cannot also be returned permuted
    @test_throws ArgumentError compile((c, a, b) -> permutedims(mul!(c, a, b)),
                                       CuArray(zeros(Float32, 24, 16)), A, B)
    # reshaped inputs and constants have no producing op to re-lay them out
    @test_throws ArgumentError compile((v, b) -> permutedims(reshape(v, 4, :)),
                                       CuArray(rand(Float32, 16)), B)
end

@testset "view inputs" begin
    # views of inputs bind by pointer offset with the view's strides — no
    # copy, no extra op; the vcat'd gate/up split fuses into one graph
    GU = CuArray(rand(Float32, 32, 24))
    B  = CuArray(rand(Float32, 24, 8))
    gu = Array(GU)

    c = compile(x -> tanh.(view(x, 1:16, :)) .* view(x, 17:32, :), GU)
    @test Array(c(GU)) ≈ tanh.(gu[1:16, :]) .* gu[17:32, :] rtol=1e-5

    # a view feeding matmul
    c2 = compile((x, b) -> view(x, 1:16, :) * b, GU, B)
    @test Array(c2(GU, B)) ≈ gu[1:16, :] * Array(B) rtol=1e-2 atol=1e-2

    # column views are pointer offsets too, and views compose
    c3 = compile(x -> view(view(x, :, 3:20), 5:12, :) .* 2f0, GU)
    @test Array(c3(GU)) ≈ gu[5:12, 3:20] .* 2 rtol=1e-5

    # odd row offset: alignment is declared from the example and revalidated
    c5 = compile(x -> view(x, 2:17, :) .* 2f0, GU)
    @test Array(c5(GU)) ≈ gu[2:17, :] .* 2 rtol=1e-5

    # rebinding new arrays of the same shapes
    GU2 = CuArray(rand(Float32, 32, 24))
    @test Array(c(GU2)) ≈ tanh.(Array(GU2)[1:16, :]) .* Array(GU2)[17:32, :] rtol=1e-5

    # view arguments at the call boundary use the same binding path
    V = view(GU, 1:16, :)
    c6 = compile((v, b) -> v * b, V, B)
    @test Array(c6(V, B)) ≈ gu[1:16, :] * Array(B) rtol=1e-2 atol=1e-2

    # computed values have a fixed layout; integer indices drop dimensions;
    # stepped ranges would leave no packed dimension (engines fault on it)
    @test_throws ArgumentError compile((x, b) -> view(x * b, 1:8, :) .* 2f0, GU, B)
    @test_throws ArgumentError compile(x -> view(x, 1, :) .* 2f0, GU)
    @test_throws ArgumentError compile(x -> view(x, 1:2:31, :) .* 2f0, GU)
    @test_throws BoundsError compile(x -> view(x, 0:16, :) .* 2f0, GU)
end

@testset "generic dispatch" begin
    # ::AbstractArray-constrained code accepts traced values directly
    f(x::AbstractArray, y::AbstractMatrix) = x * y .+ one(eltype(x))
    A, B = CuArray(rand(Float32, 16, 16)), CuArray(rand(Float32, 16, 16))
    c = compile(f, A, B)
    @test Array(c(A, B)) ≈ f(Array(A), Array(B)) rtol=1e-2 atol=1e-2

    # Base's array +, unary -, and scalar scaling generics land in the tracer
    g(x::AbstractArray) = -(2f0 * x + x / 2f0)
    c2 = compile(g, A)
    @test Array(c2(A)) ≈ g(Array(A)) rtol=1e-5

    # interface promises a symbolic value cannot keep fail clearly
    @test_throws ArgumentError compile(x -> (x[1]; x), A)
end

@testset "in-place" begin
    A, B = CuArray(rand(Float32, 32, 16)), CuArray(rand(Float32, 16, 24))
    C = CuArray(zeros(Float32, 32, 24))

    # pure mutation: nothing returned, the destination buffer is written
    c = compile((c, a, b) -> (mul!(c, a, b); nothing), C, A, B)
    @test c(C, A, B) === nothing
    @test Array(C) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2

    # returning the assigned value hands back the destination array itself
    c2 = compile((c, a, b) -> mul!(c, a, b), C, A, B)
    @test c2(C, A, B) === C

    # 5-arg mul! fuses α·A·B + β·C as a matmul epilogue. β = 0 must not read
    # the destination — NaN contents prove it stays unread
    fill!(C, NaN32)
    c5 = compile((c, a, b) -> (mul!(c, a, b, 2f0, 0f0); nothing), C, A, B)
    c5(C, A, B)
    @test Array(C) ≈ 2 .* (Array(A) * Array(B)) rtol=1e-2 atol=1e-2

    # β ≠ 0 reads the pre-write contents of the aliased buffer (β = 1 is the
    # gradient-accumulation idiom); engine tolerance for the aliasing is
    # measured, not contractual, so skip where cuDNN declines
    c6 = try
        compile((c, a, b) -> (mul!(c, a, b, 2.5f0, -0.5f0); nothing), C, A, B)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if c6 === nothing
        @test_skip false
    else
        C0 = copy(Array(C))
        c6(C, A, B)
        @test Array(C) ≈ 2.5f0 .* (Array(A) * Array(B)) .- 0.5f0 .* C0 rtol=1e-2 atol=1e-2
        C0 = copy(Array(C))
        c7 = compile((c, a, b) -> (mul!(c, a, b, 1f0, 1f0); nothing), C, A, B)
        c7(C, A, B)
        @test Array(C) ≈ Array(A) * Array(B) .+ C0 rtol=1e-2 atol=1e-2
    end

    # in-place broadcast assignment into a caller buffer
    Y = CuArray(zeros(Float32, 16, 16))
    X = CuArray(rand(Float32, 16, 16))
    c3 = compile((y, x) -> (y .= max.(x .- 0.5f0, 0f0); nothing), Y, X)
    c3(Y, X)
    @test Array(Y) ≈ max.(Array(X) .- 0.5f0, 0f0)

    # y .+= x reads and writes the same buffer; engine support is cuDNN's call
    c4 = try
        compile((y, x) -> (y .+= x; nothing), Y, X)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end
    if c4 === nothing
        @test_skip false
    else
        before = Array(Y)
        c4(Y, X)
        @test Array(Y) ≈ before .+ Array(X) rtol=1e-5
    end

    # Base's mutating idiom: returning the destination argument hands back
    # the caller's buffer, resolved to the value written into it
    C8 = CuArray(zeros(Float32, 32, 24))
    c8 = compile((c, a, b) -> (mul!(c, a, b); c), C8, A, B)
    @test c8(C8, A, B) === C8
    @test Array(C8) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2
    c9 = compile((y, x) -> (y .= max.(x, 0f0); y), Y, X)
    @test c9(Y, X) === Y
    @test Array(Y) ≈ max.(Array(X), 0f0)
    # returning an input nothing was written to stays refused
    @test_throws ArgumentError compile((c, a, b) -> (mul!(c, a, b); a), C8, A, B)

    # graph execution is dataflow: reading a buffer after writing it is refused
    @test_throws ArgumentError compile((y, x) -> (y .= x .* 2f0; y .* x), Y, X)
    # destinations must be trace inputs
    @test_throws ArgumentError compile((y, x) -> (t = x .* 2f0; t .= x; nothing), Y, X)
    # shape mismatch at trace time
    @test_throws DimensionMismatch compile((c, a, b) -> (mul!(c, a, b); nothing),
                                           CuArray(zeros(Float32, 8, 8)), A, B)
end

@testset "Stiletto.mul!" begin
    A, B = CuArray(rand(Float32, 32, 16)), CuArray(rand(Float32, 16, 24))
    C = CuArray(zeros(Float32, 32, 24))

    # eager tier jits the graph and returns the destination
    @test Stiletto.mul!(C, A, B) === C
    @test Array(C) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2

    # β = 0 leaves the destination unread (NaN contents prove it)
    fill!(C, NaN32)
    Stiletto.mul!(C, A, B, 2f0, 0f0)
    @test Array(C) ≈ 2 .* (Array(A) * Array(B)) rtol=1e-2 atol=1e-2

    # traced tier: the qualified verb and LinearAlgebra.mul! trace identically
    fill!(C, 0f0)
    cm = compile((c, a, b) -> (Stiletto.mul!(c, a, b); nothing), C, A, B)
    cm(C, A, B)
    @test Array(C) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2

    # a traced destination claims plain operands: both get captured
    fill!(C, 0f0)
    cc = compile(c -> (mul!(c, A, B); nothing), C)
    cc(C)
    @test Array(C) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2

    # gemv: the (K,) operand and (M,) product declare as K×1 and M×1
    x = CuArray(rand(Float32, 16))
    y = CuArray(zeros(Float32, 32))
    @test Stiletto.mul!(y, A, x) === y
    @test Array(y) ≈ Array(A) * Array(x) rtol=1e-2 atol=1e-2

    # traced gemv through the LinearAlgebra spelling, β = 0 NaN-proof
    fill!(y, NaN32)
    cv = compile((y, a, x) -> (mul!(y, a, x, 2f0, 0f0); nothing), y, A, x)
    cv(y, A, x)
    @test Array(y) ≈ 2 .* (Array(A) * Array(x)) rtol=1e-2 atol=1e-2

    @test_throws DimensionMismatch Stiletto.mul!(y, A, CuArray(rand(Float32, 17)))

    # Base's shapes only; batching is spelled batched_mul!
    @test_throws MethodError Stiletto.mul!(CuArray(zeros(Float32, 8, 12, 3)),
        CuArray(rand(Float32, 8, 16, 3)), CuArray(rand(Float32, 16, 12, 3)))
end

@testset "jit" begin
    fwd(a, b) = max.(a * b, 0f0)
    A, B = CuArray(rand(Float32, 32, 16)), CuArray(rand(Float32, 16, 8))
    n0 = length(cuDNN.handle().plans)
    @test Array(jit(fwd, A, B)) ≈ max.(Array(A) * Array(B), 0f0) rtol=1e-2 atol=1e-2
    @test length(cuDNN.handle().plans) == n0 + 1
    jit(fwd, A, B)                              # same signature: cache hit
    @test length(cuDNN.handle().plans) == n0 + 1
    jit(fwd, CuArray(rand(Float32, 8, 16)), CuArray(rand(Float32, 16, 8)))
    @test length(cuDNN.handle().plans) == n0 + 2  # new shapes: new entry

    # arguments are runtime inputs, captures are trace-time constants:
    # data passed as arguments shares one graph across arrays...
    matvec(w, x) = w * x
    W1, W2 = CuArray(rand(Float32, 16, 16)), CuArray(rand(Float32, 16, 16))
    X = CuArray(rand(Float32, 16, 4))
    nl = length(cuDNN.handle().plans)
    @test Array(jit(matvec, W1, X)) ≈ Array(W1) * Array(X) rtol=1e-2 atol=1e-2
    @test Array(jit(matvec, W2, X)) ≈ Array(W2) * Array(X) rtol=1e-2 atol=1e-2
    @test length(cuDNN.handle().plans) == nl + 1

    # ...while captured arrays are baked by identity: each closure (and each
    # functor instance) compiles its own graph, and repeat calls hit
    layer(w) = x -> w * x
    @test Array(jit(layer(W1), X)) ≈ Array(W1) * Array(X) rtol=1e-2 atol=1e-2
    @test Array(jit(layer(W2), X)) ≈ Array(W2) * Array(X) rtol=1e-2 atol=1e-2
    @test length(cuDNN.handle().plans) == nl + 3
    d1 = Dense(W1)
    @test Array(jit(d1, X)) ≈ Array(W1) * Array(X) rtol=1e-2 atol=1e-2
    jit(d1, X)                                  # same instance: cache hit
    @test length(cuDNN.handle().plans) == nl + 4

    # trace-time values derived from captures bake correctly per closure
    layer2(w1, w2) = x -> (w1 * w2) * x
    @test Array(jit(layer2(W1, W2), X)) ≈ Array((W1 * W2) * X) rtol=1e-2 atol=1e-2
    @test Array(jit(layer2(W2, W1), X)) ≈ Array((W2 * W1) * X) rtol=1e-2 atol=1e-2

    # scalar arguments are runtime inputs: one graph, value rebound per call
    ns = length(cuDNN.handle().plans)
    scale(x, s) = s .* max.(x, 0f0)
    @test Array(jit(scale, X, 2f0)) ≈ 2f0 .* max.(Array(X), 0f0) rtol=1e-5
    @test Array(jit(scale, X, -3f0)) ≈ -3f0 .* max.(Array(X), 0f0) rtol=1e-5
    @test length(cuDNN.handle().plans) == ns + 1

    # branching on a captured runtime value resolves at trace time; the value
    # is part of the cache key, so each branch gets its own specialized graph
    apply(flag) = x -> flag ? max.(x, 0f0) : x .- 1f0
    n1 = length(cuDNN.handle().plans)
    @test Array(jit(apply(true), X)) ≈ max.(Array(X), 0f0)
    @test Array(jit(apply(false), X)) ≈ Array(X) .- 1f0
    @test length(cuDNN.handle().plans) == n1 + 2

    # a mutating jit call replays under CUDA graph capture
    Y = CuArray(zeros(Float32, 16, 4))
    step!(y, w, x) = (mul!(y, w, x); nothing)
    jit(step!, Y, W1, X)                        # warm the plan cache
    graph = CUDACore.capture(throw_error=false) do
        jit(step!, Y, W1, X)
    end
    if graph === nothing
        @test_skip false
    else
        exec = CUDACore.instantiate(graph)
        fill!(Y, 0f0)
        CUDACore.launch(exec)
        CUDACore.synchronize()
        @test Array(Y) ≈ Array(W1) * Array(X) rtol=1e-2 atol=1e-2
    end
end

@testset "jit invalidation" begin
    # the cache keys on the CodeInstance of the traced call, so redefining
    # the function — or anything it calls — retraces instead of replaying
    # the stale graph. invokelatest because testset bodies freeze world age.
    X = CuArray(rand(Float32, 8, 8))
    @eval stale_f(x) = x .+ 1f0
    @test Array(Base.invokelatest(jit, stale_f, X)) ≈ Array(X) .+ 1 rtol=1e-5
    @eval stale_f(x) = x .+ 2f0
    @test Array(Base.invokelatest(jit, stale_f, X)) ≈ Array(X) .+ 2 rtol=1e-5

    @eval stale_inner(x) = x .+ 1f0
    @eval stale_outer(x) = stale_inner(x) .* 2f0
    @test Array(Base.invokelatest(jit, stale_outer, X)) ≈ (Array(X) .+ 1) .* 2 rtol=1e-5
    @eval stale_inner(x) = x .+ 3f0
    @test Array(Base.invokelatest(jit, stale_outer, X)) ≈ (Array(X) .+ 3) .* 2 rtol=1e-5

    # an unchanged world replays the cached plan
    nr = length(cuDNN.handle().plans)
    Base.invokelatest(jit, stale_outer, X)
    @test length(cuDNN.handle().plans) == nr
end

@testset "graph and bindings accessors" begin
    # Stiletto.graph + Stiletto.bindings split c(args...) into its halves:
    # binding preparation (the argument→tensor mapping the trace declared)
    # and execution, so callers can schedule execute! themselves
    A, B = CuArray(rand(Float32, 32, 16)), CuArray(rand(Float32, 16, 8))
    c = compile((a, b) -> max.(a * b, 0f0), A, B)
    g = Stiletto.graph(c)
    @test g isa cuDNN.Graph
    binds, outs = Stiletto.bindings(c, A, B)
    @test length(outs) == 1
    cuDNN.execute!(g, binds)
    @test Array(outs[1]) ≈ max.(Array(A) * Array(B), 0f0) rtol=1e-2 atol=1e-2

    # rebinding: fresh arguments, same graph
    A2 = CuArray(rand(Float32, 32, 16))
    binds2, outs2 = Stiletto.bindings(c, A2, B)
    cuDNN.execute!(g, binds2)
    @test Array(outs2[1]) ≈ max.(Array(A2) * Array(B), 0f0) rtol=1e-2 atol=1e-2

    # in-place destinations appear in the outputs, backed by the caller buffer
    C = CuArray(zeros(Float32, 32, 8))
    cm = compile((c, a, b) -> mul!(c, a, b), C, A, B)
    bm, om = Stiletto.bindings(cm, C, A, B)
    @test om[1] === C
    cuDNN.execute!(Stiletto.graph(cm), bm)
    @test Array(C) ≈ Array(A) * Array(B) rtol=1e-2 atol=1e-2
end

@testset "macros" begin
    f(a, b) = a * b .+ 1f0
    A, B = CuArray(rand(Float32, 16, 16)), CuArray(rand(Float32, 16, 16))
    want = Array(A) * Array(B) .+ 1f0

    g = @compile f(A, B)
    @test Array(g(A, B)) ≈ want rtol=1e-2 atol=1e-2
    @test Array(@jit f(A, B)) ≈ want rtol=1e-2 atol=1e-2

    # call keywords ride a closure into the traced function: literals bake,
    # variables become value-keyed captures
    h(a, b; s=1f0) = s .* (a * b)
    @test Array(@jit h(A, B; s=2f0)) ≈ 2f0 .* (Array(A) * Array(B)) rtol=1e-2 atol=1e-2
    t = -1f0
    @test Array(@jit h(A, B; s=t)) ≈ -(Array(A) * Array(B)) rtol=1e-2 atol=1e-2

    # splatting passes through
    args = (A, B)
    @test Array(@jit f(args...)) ≈ want rtol=1e-2 atol=1e-2

    # only function calls are accepted
    @test_throws LoadError @eval @jit A
end

@testset "errors" begin
    A, B = CuArray(rand(Float32, 8, 8)), CuArray(rand(Float32, 8, 8))
    # a permuted computed value materializes at the output boundary only
    @test_throws ArgumentError compile((a, b) -> (a * b)' * a, A, B)
    # returning an input is not a graph
    @test_throws ArgumentError compile((a, b) -> a, A, B)
    # unmapped function
    @test_throws ArgumentError compile((a, b) -> hypot.(a, b), A, B)
    # shape mismatch surfaces at trace time
    @test_throws DimensionMismatch compile((a, b) -> a * b,
                                           CuArray(rand(Float32, 8, 4)), CuArray(rand(Float32, 8, 4)))
    # array + does not broadcast dims, matching Julia; .+ does
    @test_throws DimensionMismatch compile((a, b) -> a + b,
                                           CuArray(rand(Float32, 8, 4)), CuArray(rand(Float32, 8, 1)))
end
include("corpus.jl")
include("microscaling.jl")


end
