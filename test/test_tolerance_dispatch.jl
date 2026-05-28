const TOL_META = "TolDispatch"

# Sweep one univariate quadratic config over a few tolerances on [0, Δ] and
# assert the achieved approximation gap is within the requested tolerance.
function _sweep_quadratic_tolerance(make_config, Δ, tols)
    for ε in tols
        cfg = make_config(ε, Δ)
        gaps = Float64[]
        for x0 in range(0.0, Δ; length = 11)
            setup = _setup_qa_test(["g"], 1:1)
            JuMP.fix(setup.var_container["g", 1], x0; force = true)
            IOM._add_quadratic_approx!(
                cfg,
                setup.container,
                MockThermalGen,
                ["g"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = Δ)],
                TOL_META,
            )
            expr = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TOL_META,
            )
            JuMP.@objective(setup.jump_model, Min, expr["g", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            push!(gaps, abs(x0^2 - JuMP.objective_value(setup.jump_model)))
        end
        @test maximum(gaps) <= ε + 1e-6
    end
end

# Same idea for an epigraph-style lower bound: the achieved gap is
# (x0^2 - approximation value) since the approximation lower-bounds x^2.
function _sweep_epigraph_tolerance(make_config, Δ, tols)
    for ε in tols
        cfg = make_config(ε, Δ)
        gaps = Float64[]
        for x0 in range(0.0, Δ; length = 11)
            setup = _setup_qa_test(["g"], 1:1)
            JuMP.fix(setup.var_container["g", 1], x0; force = true)
            IOM._add_quadratic_approx!(
                cfg,
                setup.container,
                MockThermalGen,
                ["g"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = Δ)],
                TOL_META,
            )
            expr = IOM.get_expression(
                setup.container,
                IOM.EpigraphExpression,
                MockThermalGen,
                TOL_META,
            )
            # Epigraph gives a lower bound on x²; minimize z to see the
            # closest the tangent envelope reaches the true value from below.
            JuMP.@objective(setup.jump_model, Min, expr["g", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            push!(gaps, abs(x0^2 - JuMP.objective_value(setup.jump_model)))
        end
        @test maximum(gaps) <= ε + 1e-6
    end
end

function _sweep_bilinear_tolerance(make_config, Δx, Δy, tols)
    for ε in tols
        cfg = make_config(ε, Δx, Δy)
        gaps = Float64[]
        for x0 in range(0.05 * Δx, 0.95 * Δx; length = 5),
            y0 in range(0.05 * Δy, 0.95 * Δy; length = 5),
            sense in (JuMP.MIN_SENSE, JuMP.MAX_SENSE)

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
                [(min = 0.0, max = Δx)],
                [(min = 0.0, max = Δy)],
                TOL_META,
            )
            expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                TOL_META,
            )
            JuMP.@objective(setup.jump_model, sense, expr["d", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            push!(gaps, abs(x0 * y0 - JuMP.objective_value(setup.jump_model)))
        end
        @test maximum(gaps) <= ε + 1e-6
    end
end

@testset "Tolerance Dispatch" begin
    # SOS2 methods accept `pwmcc_segments=0` to skip the optional piecewise
    # McCormick concave cuts (the cuts interact poorly with a fixed primal,
    # mirroring how the existing SOS2 tests in test_quadratic_approximations.jl
    # all pass pwmcc_segments=0). The tolerance contract bounds the PWL error,
    # which the cuts don't affect.
    @testset "SolverSOS2QuadConfig" begin
        _sweep_quadratic_tolerance(
            (ε, Δ) -> IOM.SolverSOS2QuadConfig(;
                tolerance = ε,
                max_delta = Δ,
                pwmcc_segments = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "ManualSOS2QuadConfig" begin
        _sweep_quadratic_tolerance(
            (ε, Δ) -> IOM.ManualSOS2QuadConfig(;
                tolerance = ε,
                max_delta = Δ,
                pwmcc_segments = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "SawtoothQuadConfig" begin
        _sweep_quadratic_tolerance(
            (ε, Δ) -> IOM.SawtoothQuadConfig(; tolerance = ε, max_delta = Δ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "EpigraphQuadConfig" begin
        _sweep_epigraph_tolerance(
            (ε, Δ) -> IOM.EpigraphQuadConfig(; tolerance = ε, max_delta = Δ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "NMDTQuadConfig" begin
        _sweep_quadratic_tolerance(
            (ε, Δ) ->
                IOM.NMDTQuadConfig(; tolerance = ε, max_delta = Δ, epigraph_depth = 0),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "DNMDTQuadConfig" begin
        _sweep_quadratic_tolerance(
            (ε, Δ) ->
                IOM.DNMDTQuadConfig(; tolerance = ε, max_delta = Δ, epigraph_depth = 0),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "NMDTBilinearConfig" begin
        _sweep_bilinear_tolerance(
            (ε, Δx, Δy) -> IOM.NMDTBilinearConfig(;
                tolerance = ε,
                max_delta_x = Δx,
                max_delta_y = Δy,
            ),
            1.0,
            1.0,
            (1e-1, 1e-2),
        )
    end

    @testset "DNMDTBilinearConfig" begin
        _sweep_bilinear_tolerance(
            (ε, Δx, Δy) -> IOM.DNMDTBilinearConfig(;
                tolerance = ε,
                max_delta_x = Δx,
                max_delta_y = Δy,
            ),
            1.0,
            1.0,
            (1e-1, 1e-2),
        )
    end

    @testset "Non-unit domain Δ scales correctly" begin
        # Δ = 4 → sawtooth depth must grow vs Δ = 1 for the same ε.
        cfg1 = IOM.SawtoothQuadConfig(; tolerance = 1e-2, max_delta = 1.0)
        cfg4 = IOM.SawtoothQuadConfig(; tolerance = 1e-2, max_delta = 4.0)
        @test cfg4.depth > cfg1.depth
        _sweep_quadratic_tolerance(
            (ε, Δ) -> IOM.SawtoothQuadConfig(; tolerance = ε, max_delta = Δ),
            4.0,
            (1e-1, 1e-2),
        )
    end

    @testset "Stub configs error informatively" begin
        # Bin2 / HybS have hand-rolled kw stubs (their structs have fields, so
        # there's no collision with the auto-generated positional constructor).
        @test_throws ErrorException IOM.Bin2Config(; tolerance = 1e-3)
        @test_throws ErrorException IOM.HybSConfig(; tolerance = 1e-3)
        # No-op configs are empty structs, so `(; tolerance=…)` falls through
        # to Julia's standard MethodError (a hand-rolled stub would clobber
        # the auto-generated no-arg constructor).
        @test_throws MethodError IOM.NoQuadApproxConfig(; tolerance = 1e-3)
        @test_throws MethodError IOM.NoBilinearApproxConfig(; tolerance = 1e-3)
    end
end
