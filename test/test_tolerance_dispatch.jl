const TOL_META = "TolDispatch"

# --- Sampling helpers -----------------------------------------------------
#
# Per-config analytical worst-case points layered on top of a dense uniform
# grid. The uniform grid is the floor for every method so that even if a
# closed-form "worst point" estimate is wrong, we still scan the domain.
# For PWL of convex x² (sawtooth, SOS2): worst error sits at segment midpoints.
# For epigraph (max of tangents): same, midpoint between adjacent tangents.
# For NMDT/DNMDT: no clean closed form, the dense grid alone is the sample set.

_uniform_grid(delta::Float64, n::Int) = collect(range(0.0, delta; length = n))

function _sawtooth_worst_points(depth::Int, delta::Float64)
    n_seg = 2^depth
    h = delta / n_seg
    return [(k + 0.5) * h for k in 0:(n_seg - 1)]
end

function _sos2_worst_points(segments::Int, delta::Float64)
    h = delta / segments
    return [(k + 0.5) * h for k in 0:(segments - 1)]
end

# Epigraph tangent lines at breakpoints a + k·Δ/2^L; the tangent envelope is
# tightest at the breakpoints and loosest at the midpoints between adjacent
# breakpoints (same locations as the sawtooth's worst points).
_epigraph_worst_points(depth::Int, delta::Float64) = _sawtooth_worst_points(depth, delta)

_quadratic_samples(::Type{IOM.SawtoothQuadConfig}, depth::Int, delta::Float64) =
    unique!(sort!(vcat(_uniform_grid(delta, 41), _sawtooth_worst_points(depth, delta))))

_quadratic_samples(::Type{IOM.EpigraphQuadConfig}, depth::Int, delta::Float64) =
    unique!(sort!(vcat(_uniform_grid(delta, 41), _epigraph_worst_points(depth, delta))))

_quadratic_samples(
    ::Type{<:Union{IOM.SolverSOS2QuadConfig, IOM.ManualSOS2QuadConfig}},
    depth::Int,
    delta::Float64,
) = unique!(sort!(vcat(_uniform_grid(delta, 41), _sos2_worst_points(depth, delta))))

_quadratic_samples(
    ::Type{<:Union{IOM.NMDTQuadConfig, IOM.DNMDTQuadConfig}},
    _::Int,
    delta::Float64,
) = _uniform_grid(delta, 81)

function _bilinear_samples(
    ::Type{Q},
    depth::Int,
    delta_x::Float64,
    delta_y::Float64,
) where {Q}
    # Per-axis samples plus the four boundary corners. Boundary fixings are
    # representative of real production use, so any solver pathology at corners
    # should surface here rather than be hidden by an inset.
    xs = _quadratic_samples(Q, depth, delta_x)
    ys = _quadratic_samples(Q, depth, delta_y)
    pts = [(x0, y0) for x0 in xs, y0 in ys]
    corners = [(0.0, 0.0), (0.0, delta_y), (delta_x, 0.0), (delta_x, delta_y)]
    return unique!(vcat(vec(pts), corners))
end

# --- Evaluation helpers ---------------------------------------------------
#
# Each returns the max observed |approx − true| over the sample set, plus the
# ratio `max_gap / tolerance` for logging.

function _eval_quadratic_overestimator(
    cfg,
    sample_points::Vector{Float64},
    delta::Float64,
    tolerance::Float64;
    expr_type = IOM.QuadraticExpression,
)
    gaps = Float64[]
    for x0 in sample_points
        setup = _setup_qa_test(["g"], 1:1)
        JuMP.fix(setup.var_container["g", 1], x0; force = true)
        IOM._add_quadratic_approx!(
            cfg,
            setup.container,
            MockThermalGen,
            ["g"],
            1:1,
            setup.var_container,
            [(min = 0.0, max = delta)],
            TOL_META,
        )
        expr = IOM.get_expression(setup.container, expr_type, MockThermalGen, TOL_META)
        JuMP.@objective(setup.jump_model, Min, expr["g", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)
        push!(gaps, abs(x0^2 - JuMP.objective_value(setup.jump_model)))
    end
    return (max_gap = maximum(gaps), ratio = maximum(gaps) / tolerance)
end

function _eval_quadratic_underestimator(
    cfg,
    sample_points::Vector{Float64},
    delta::Float64,
    tolerance::Float64,
)
    return _eval_quadratic_overestimator(
        cfg, sample_points, delta, tolerance;
        expr_type = IOM.EpigraphExpression,
    )
end

function _eval_bilinear(
    cfg,
    sample_points::Vector{Tuple{Float64, Float64}},
    delta_x::Float64,
    delta_y::Float64,
    tolerance::Float64,
)
    gaps = Float64[]
    for (x0, y0) in sample_points, sense in (JuMP.MIN_SENSE, JuMP.MAX_SENSE)
        setup = _setup_bilinear_test(["d"], 1:1)
        JuMP.fix(setup.x_var_container["d", 1], x0; force = true)
        JuMP.fix(setup.y_var_container["d", 1], y0; force = true)
        IOM._add_bilinear_approx!(
            cfg,
            setup.container,
            MockThermalGen,
            ["d"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            [(min = 0.0, max = delta_x)],
            [(min = 0.0, max = delta_y)],
            TOL_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.BilinearProductExpression, MockThermalGen, TOL_META,
        )
        JuMP.@objective(setup.jump_model, sense, expr["d", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)
        push!(gaps, abs(x0 * y0 - JuMP.objective_value(setup.jump_model)))
    end
    return (max_gap = maximum(gaps), ratio = maximum(gaps) / tolerance)
end

# --- Tolerance harness ----------------------------------------------------

function _run_quadratic_tol_case(
    ::Type{Q},
    make_cfg::F,
    tols::Tuple,
    delta::Float64;
    underestimator::Bool = false,
    extra_label::String = "",
) where {Q, F}
    for tol in tols
        d = IOM.tolerance_depth(Q; tolerance = tol, max_delta = delta)
        cfg = make_cfg(d)
        samples = _quadratic_samples(Q, d, delta)
        result = if underestimator
            _eval_quadratic_underestimator(cfg, samples, delta, tol)
        else
            _eval_quadratic_overestimator(cfg, samples, delta, tol)
        end
        @info "$(nameof(Q))$(isempty(extra_label) ? "" : " ($extra_label)")" tolerance = tol depth =
            d max_gap = result.max_gap achieved_over_tol = result.ratio
        @test result.max_gap <= tol + 1e-10
        @test result.ratio <= 1 + 1e-10
    end
end

# --- Quadratic configs ----------------------------------------------------

@testset "Tolerance Dispatch" begin
    @testset "Quadratic configs" begin
        @testset "SawtoothQuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.SawtoothQuadConfig,
                d -> IOM.SawtoothQuadConfig(; depth = d),
                (1e-1, 1e-2, 1e-3),
                1.0,
            )
        end

        @testset "SawtoothQuadConfig (Δ = 4)" begin
            _run_quadratic_tol_case(
                IOM.SawtoothQuadConfig,
                d -> IOM.SawtoothQuadConfig(; depth = d),
                (1e-1, 1e-2),
                4.0,
            )
        end

        @testset "EpigraphQuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.EpigraphQuadConfig,
                d -> IOM.EpigraphQuadConfig(; depth = d),
                (1e-1, 1e-2, 1e-3),
                1.0;
                underestimator = true,
            )
        end

        @testset "EpigraphQuadConfig (Δ = 4)" begin
            _run_quadratic_tol_case(
                IOM.EpigraphQuadConfig,
                d -> IOM.EpigraphQuadConfig(; depth = d),
                (1e-1, 1e-2),
                4.0;
                underestimator = true,
            )
        end

        # PWMCC only affects LP relaxation; left at the constructor default
        # since it doesn't influence the tolerance bound this test verifies.
        @testset "SolverSOS2QuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.SolverSOS2QuadConfig,
                d -> IOM.SolverSOS2QuadConfig(; depth = d),
                (1e-1, 1e-2, 1e-3),
                1.0,
            )
        end

        @testset "SolverSOS2QuadConfig (Δ = 7)" begin
            _run_quadratic_tol_case(
                IOM.SolverSOS2QuadConfig,
                d -> IOM.SolverSOS2QuadConfig(; depth = d),
                (1e-1, 1e-2),
                7.0,
            )
        end

        @testset "ManualSOS2QuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.ManualSOS2QuadConfig,
                d -> IOM.ManualSOS2QuadConfig(; depth = d),
                (1e-1, 1e-2, 1e-3),
                1.0,
            )
        end

        @testset "NMDTQuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.NMDTQuadConfig,
                d -> IOM.NMDTQuadConfig(; depth = d, epigraph_depth = 0),
                (1e-1, 1e-2),
                1.0,
            )
        end

        @testset "DNMDTQuadConfig (Δ = 1)" begin
            _run_quadratic_tol_case(
                IOM.DNMDTQuadConfig,
                d -> IOM.DNMDTQuadConfig(; depth = d, epigraph_depth = 0),
                (1e-1, 1e-2),
                1.0,
            )
        end
    end

    # --- Bilinear NMDT configs ----------------------------------------------

    @testset "Bilinear NMDT configs" begin
        @testset "NMDTBilinearConfig (Δx = Δy = 1)" begin
            delta_x = 1.0
            delta_y = 1.0
            for tol in (1e-1, 1e-2)
                d = IOM.tolerance_depth(IOM.NMDTBilinearConfig;
                    tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
                )
                cfg = IOM.NMDTBilinearConfig(; depth = d)
                samples = _bilinear_samples(IOM.NMDTQuadConfig, d, delta_x, delta_y)
                result = _eval_bilinear(cfg, samples, delta_x, delta_y, tol)
                @info "NMDTBilinearConfig" tolerance = tol depth = d max_gap =
                    result.max_gap achieved_over_tol = result.ratio
                @test result.max_gap <= tol + 1e-10
                @test result.ratio <= 1 + 1e-10
            end
        end

        @testset "DNMDTBilinearConfig (Δx = Δy = 1)" begin
            delta_x = 1.0
            delta_y = 1.0
            for tol in (1e-1, 1e-2)
                d = IOM.tolerance_depth(IOM.DNMDTBilinearConfig;
                    tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
                )
                cfg = IOM.DNMDTBilinearConfig(; depth = d)
                samples = _bilinear_samples(IOM.DNMDTQuadConfig, d, delta_x, delta_y)
                result = _eval_bilinear(cfg, samples, delta_x, delta_y, tol)
                @info "DNMDTBilinearConfig" tolerance = tol depth = d max_gap =
                    result.max_gap achieved_over_tol = result.ratio
                @test result.max_gap <= tol + 1e-10
                @test result.ratio <= 1 + 1e-10
            end
        end

        @testset "NMDTBilinearConfig (Δx = 2, Δy = 3)" begin
            tol = 1e-2
            delta_x = 2.0
            delta_y = 3.0
            d_unit = IOM.tolerance_depth(IOM.NMDTBilinearConfig;
                tolerance = tol, max_delta_x = 1.0, max_delta_y = 1.0,
            )
            d_big = IOM.tolerance_depth(IOM.NMDTBilinearConfig;
                tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
            )
            @test d_big > d_unit
            cfg = IOM.NMDTBilinearConfig(; depth = d_big)
            samples = _bilinear_samples(IOM.NMDTQuadConfig, d_big, delta_x, delta_y)
            result = _eval_bilinear(cfg, samples, delta_x, delta_y, tol)
            @info "NMDTBilinearConfig (non-unit)" tolerance = tol depth = d_big max_gap =
                result.max_gap achieved_over_tol = result.ratio
            @test result.max_gap <= tol + 1e-10
            @test result.ratio <= 1 + 1e-10
        end
    end

    # --- Bin2Config{Q} -------------------------------------------------------

    @testset "Bin2Config{Q}" begin
        # NMDT/DNMDT inner Qs must have epigraph_depth = 0 (see bin2.jl docstring
        # caveat — with epigraph tightening the inner result is no longer
        # one-sided over and the derivation breaks).
        bin2_cases = [
            (IOM.SawtoothQuadConfig, d -> IOM.SawtoothQuadConfig(; depth = d)),
            (IOM.SolverSOS2QuadConfig, d -> IOM.SolverSOS2QuadConfig(; depth = d)),
            (IOM.ManualSOS2QuadConfig, d -> IOM.ManualSOS2QuadConfig(; depth = d)),
            (
                IOM.NMDTQuadConfig,
                d -> IOM.NMDTQuadConfig(; depth = d, epigraph_depth = 0),
            ),
            (
                IOM.DNMDTQuadConfig,
                d -> IOM.DNMDTQuadConfig(; depth = d, epigraph_depth = 0),
            ),
        ]
        for (Q, make_inner) in bin2_cases
            @testset "Bin2Config{$(nameof(Q))}" begin
                delta_x = 1.0
                delta_y = 1.0
                for tol in (1e-1, 1e-2)
                    d = IOM.tolerance_depth(IOM.Bin2Config{Q};
                        tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
                    )
                    cfg = IOM.Bin2Config(make_inner(d); add_mccormick = false)
                    samples = _bilinear_samples(Q, d, delta_x, delta_y)
                    result = _eval_bilinear(cfg, samples, delta_x, delta_y, tol)
                    @info "Bin2Config{$(nameof(Q))}" tolerance = tol depth = d max_gap =
                        result.max_gap achieved_over_tol = result.ratio
                    @test result.max_gap <= tol + 1e-10
                    @test result.ratio <= 1 + 1e-10
                end
            end
        end
    end

    # --- HybSConfig{Q} -------------------------------------------------------

    @testset "HybSConfig{Q}" begin
        hybs_cases = [
            (IOM.SawtoothQuadConfig, d -> IOM.SawtoothQuadConfig(; depth = d)),
            (IOM.SolverSOS2QuadConfig, d -> IOM.SolverSOS2QuadConfig(; depth = d)),
            (IOM.ManualSOS2QuadConfig, d -> IOM.ManualSOS2QuadConfig(; depth = d)),
        ]
        for (Q, make_inner) in hybs_cases
            @testset "HybSConfig{$(nameof(Q))}" begin
                delta_x = 1.0
                delta_y = 1.0
                for tol in (1e-1, 1e-2)
                    di = IOM.tolerance_depth(IOM.HybSConfig{Q};
                        tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
                    )
                    de = IOM.tolerance_epigraph_depth(IOM.HybSConfig{Q};
                        tolerance = tol, max_delta_x = delta_x, max_delta_y = delta_y,
                    )
                    cfg = IOM.HybSConfig(make_inner(di);
                        epigraph_depth = de, add_mccormick = false,
                    )
                    samples = _bilinear_samples(Q, di, delta_x, delta_y)
                    result = _eval_bilinear(cfg, samples, delta_x, delta_y, tol)
                    @info "HybSConfig{$(nameof(Q))}" tolerance = tol inner_depth = di epi_depth =
                        de max_gap = result.max_gap achieved_over_tol = result.ratio
                    @test result.max_gap <= tol + 1e-10
                    @test result.ratio <= 1 + 1e-10
                end
            end
        end
    end
end
