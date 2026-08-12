# A corpus of traceable patterns modeled on cudnn-frontend's samples
# (samples/cpp/misc/pointwise.cpp, samples/cpp/matmul, boolean fusions, the
# fp8 amax epilogue, softmax subgraphs). Every entry is plain Julia, so the
# reference is the same function applied to CPU arrays; patterns without an
# engine on the current stack are skipped rather than failed.

compile_tolerant(f, args...; kwargs...) =
    try
        compile(f, args...; kwargs...)
    catch e
        e isa cuDNN.UnsupportedGraphError || rethrow()
        nothing
    end

function check(name, f, args...; rtol=1e-4, atol=1e-4, io_dtype=Float32, ref=f)
    @testset "$name" begin
        c = compile_tolerant(f, args...; io_dtype)
        if c === nothing
            @test_skip false
        else
            got = c(args...)
            want = ref(map(Array, args)...)
            @test Float32.(Array(got)) ≈ Float32.(want) rtol=rtol atol=atol
        end
    end
end

@testset "corpus" begin
    M, N, K = 32, 32, 32
    x32  = CuArray(randn(Float32, M, N))
    a32  = CuArray(randn(Float32, M, K) / 8)
    b32  = CuArray(randn(Float32, K, N) / 8)
    c32  = CuArray(randn(Float32, N, M) / 8)
    col  = CuArray(randn(Float32, M, 1))
    x16  = CuArray(randn(Float16, M, N))

    # misc/pointwise.cpp "Fused scalar": half io, scalar constant folded in
    check("fused scalar", x -> tanh.(x .+ 5f0), x16; rtol=1e-2, atol=1e-2)

    # misc/pointwise.cpp "Reduction": full reduction to a single element
    # (a reduction-only graph gets no engine here; fused epilogue reductions do)
    check("full reduction", x -> maximum(x; dims=(1, 2)), x32)

    # matmul samples: bias + activation epilogue
    check("matmul bias relu", (a, b, c) -> max.(a * b .+ c, 0f0), a32, b32, col; rtol=1e-2, atol=1e-2)

    # fp8 matmul pattern: amax of the result, fused as an epilogue reduction
    check("matmul amax", (a, b) -> maximum(abs.(a * b); dims=(1, 2)), a32, b32; rtol=1e-2, atol=1e-2)

    # sdpa-style row softmax from primitives — a reduction result feeding
    # further computation exceeds runtime fusion (reductions fuse only as
    # terminal nodes), which is why softmax exists only inside the SDPA op
    check("softmax", x -> (e = exp.(x .- maximum(x; dims=1)); e ./ sum(e; dims=1)), x32;
          rtol=1e-3, atol=1e-3)

    # membound/boolean_fusion.cpp: comparisons and select
    check("select relu", x -> ifelse.(x .> 0f0, x, 0f0), x32)
    check("banded select", x -> ifelse.((x .> 0f0) .& (x .< 0.5f0), x, 0f0), x32)

    # back-to-back matmul: runtime fusion takes one matmul per graph outside
    # the SDPA engines, so this stays a two-kernel job
    check("b2b matmul", (a, b, c) -> (a * b) * c, a32, b32, c32; rtol=1e-3, atol=1e-3)

    # reductions broadcast back against their input (same terminal-reduction
    # limit as softmax)
    check("mean center square", x -> (x .- mean(x; dims=2)) .^ 2, x32; rtol=1e-3, atol=1e-3)
    check("sum scale", x -> x ./ (sum(x .^ 2; dims=1) .+ 1f0), x32; rtol=1e-3, atol=1e-3)

    # pointwise mode coverage: trig, floor/ceil, atan2, reciprocal, pow
    check("trig", x -> sin.(x) .* cos.(x), x32)
    # floor/ceil are discontinuous: use a grid whose arguments stay away from
    # integers, so GPU and CPU rounding cannot land on different sides
    grid = CuArray(reshape(Float32.(mod.(0:M*N-1, 7)) .+ 0.25f0, M, N))
    check("floor ceil", x -> floor.(0.5f0 .* x) .- ceil.(0.25f0 .* x), grid)
    check("atan2", (a, b) -> atan.(a, 1f0 .+ b .^ 2), a32, CuArray(randn(Float32, M, K)))
    check("rational", x -> inv.(1f0 .+ x .^ 2), x32)

    # NNlib activations as matmul epilogues (NNlibExt)
    for (name, act) in ["relu" => relu, "sigmoid" => sigmoid, "gelu" => gelu,
                        "swish" => swish, "softplus" => softplus, "elu" => elu]
        check("$name epilogue", (a, b) -> act.(a * b), a32, b32; rtol=1e-3, atol=1e-3)
    end

    # SpecialFunctions (SpecialFunctionsExt)
    check("erf", x -> erf.(x), x32)

    # norm/rmsnorm.cpp, norm/layernorm.cpp: inference-phase per-sample norms;
    # the AbstractArray fallbacks are the CPU reference
    sc = CuArray(rand(Float32, M) .+ 0.5f0)
    bs = CuArray(randn(Float32, M))
    check("rms norm", (x, s) -> Stiletto.rmsnorm(x, s), x32, sc;
          ref=(x, s) -> ref_rmsnorm(x, s))
    check("layer norm", (x, s, b) -> Stiletto.layernorm(x, s, b), x32, sc, bs;
          ref=(x, s, b) -> ref_layernorm(x, s, b))
    check("rms norm epilogue", (x, s) -> max.(Stiletto.rmsnorm(x, s), 0f0), x32, sc;
          ref=(x, s) -> max.(ref_rmsnorm(x, s), 0f0))
end
