const TOL_META = "TolDispatch"

# Sweep one univariate quadratic config over a few tolerances on [0, delta] and
# assert the achieved approximation gap is within the requested tolerance.
function _sweep_quadratic_tolerance(make_config, delta, tols)
    for tol in tols
        cfg = make_config(tol, delta)
        gaps = Float64[]
        for x0 in range(0.0, delta; length = 11)
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
        # The math (ceiling on depth) guarantees gap ≤ tol in exact arithmetic;
        # the tiny absolute guard covers float roundoff at exact-integer boundaries
        # (e.g. SOS2 depth = delta / (2·sqrt(tol)) lands on an integer when tol = 1e-2).
        @test maximum(gaps) <= tol + 1e-10
    end
end

# Same idea for an epigraph-style lower bound: the achieved gap is
# (x0^2 - approximation value) since the approximation lower-bounds x^2.
function _sweep_epigraph_tolerance(make_config, delta, tols)
    for tol in tols
        cfg = make_config(tol, delta)
        gaps = Float64[]
        for x0 in range(0.0, delta; length = 11)
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
        # The math (ceiling on depth) guarantees gap ≤ tol in exact arithmetic;
        # the tiny absolute guard covers float roundoff at exact-integer boundaries
        # (e.g. SOS2 depth = delta / (2·sqrt(tol)) lands on an integer when tol = 1e-2).
        @test maximum(gaps) <= tol + 1e-10
    end
end

function _sweep_bilinear_tolerance(make_config, delta_x, delta_y, tols)
    for tol in tols
        cfg = make_config(tol, delta_x, delta_y)
        gaps = Float64[]
        for x0 in range(0.05 * delta_x, 0.95 * delta_x; length = 5),
            y0 in range(0.05 * delta_y, 0.95 * delta_y; length = 5),
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
                [(min = 0.0, max = delta_x)],
                [(min = 0.0, max = delta_y)],
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
        # The math (ceiling on depth) guarantees gap ≤ tol in exact arithmetic;
        # the tiny absolute guard covers float roundoff at exact-integer boundaries
        # (e.g. SOS2 depth = delta / (2·sqrt(tol)) lands on an integer when tol = 1e-2).
        @test maximum(gaps) <= tol + 1e-10
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
            (tol, delta) -> IOM.SolverSOS2QuadConfig(;
                tolerance = tol,
                max_delta = delta,
                pwmcc_segments = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "ManualSOS2QuadConfig" begin
        _sweep_quadratic_tolerance(
            (tol, delta) -> IOM.ManualSOS2QuadConfig(;
                tolerance = tol,
                max_delta = delta,
                pwmcc_segments = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "SawtoothQuadConfig" begin
        _sweep_quadratic_tolerance(
            (tol, delta) -> IOM.SawtoothQuadConfig(; tolerance = tol, max_delta = delta),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "EpigraphQuadConfig" begin
        _sweep_epigraph_tolerance(
            (tol, delta) -> IOM.EpigraphQuadConfig(; tolerance = tol, max_delta = delta),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "NMDTQuadConfig" begin
        _sweep_quadratic_tolerance(
            (tol, delta) -> IOM.NMDTQuadConfig(;
                tolerance = tol,
                max_delta = delta,
                epigraph_depth = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "DNMDTQuadConfig" begin
        _sweep_quadratic_tolerance(
            (tol, delta) -> IOM.DNMDTQuadConfig(;
                tolerance = tol,
                max_delta = delta,
                epigraph_depth = 0,
            ),
            1.0,
            (1e-1, 1e-2, 1e-3),
        )
    end

    @testset "NMDTBilinearConfig" begin
        _sweep_bilinear_tolerance(
            (tol, delta_x, delta_y) -> IOM.NMDTBilinearConfig(;
                tolerance = tol,
                max_delta_x = delta_x,
                max_delta_y = delta_y,
            ),
            1.0,
            1.0,
            (1e-1, 1e-2),
        )
    end

    @testset "DNMDTBilinearConfig" begin
        _sweep_bilinear_tolerance(
            (tol, delta_x, delta_y) -> IOM.DNMDTBilinearConfig(;
                tolerance = tol,
                max_delta_x = delta_x,
                max_delta_y = delta_y,
            ),
            1.0,
            1.0,
            (1e-1, 1e-2),
        )
    end

    # Generalized domain-scaling check: for every config used above, the
    # computed depth must grow when the domain grows for fixed tolerance,
    # and the gap sweep at the larger domain must still meet the tolerance.
    @testset "Non-unit domain delta scales correctly (univariate)" begin
        quad_factories = [
            (
                "SolverSOS2",
                (tol, delta) -> IOM.SolverSOS2QuadConfig(;
                    tolerance = tol,
                    max_delta = delta,
                    pwmcc_segments = 0,
                ),
            ),
            (
                "ManualSOS2",
                (tol, delta) -> IOM.ManualSOS2QuadConfig(;
                    tolerance = tol,
                    max_delta = delta,
                    pwmcc_segments = 0,
                ),
            ),
            (
                "Sawtooth",
                (tol, delta) ->
                    IOM.SawtoothQuadConfig(; tolerance = tol, max_delta = delta),
            ),
            (
                "NMDT",
                (tol, delta) -> IOM.NMDTQuadConfig(;
                    tolerance = tol,
                    max_delta = delta,
                    epigraph_depth = 0,
                ),
            ),
            (
                "DNMDT",
                (tol, delta) -> IOM.DNMDTQuadConfig(;
                    tolerance = tol,
                    max_delta = delta,
                    epigraph_depth = 0,
                ),
            ),
        ]
        for (name, make) in quad_factories
            @testset "$name" begin
                @test make(1e-2, 4.0).depth > make(1e-2, 1.0).depth
                _sweep_quadratic_tolerance(make, 4.0, (1e-1, 1e-2))
            end
        end

        @testset "Epigraph" begin
            make =
                (tol, delta) ->
                    IOM.EpigraphQuadConfig(; tolerance = tol, max_delta = delta)
            @test make(1e-2, 4.0).depth > make(1e-2, 1.0).depth
            _sweep_epigraph_tolerance(make, 4.0, (1e-1, 1e-2))
        end
    end

    @testset "Non-unit domain delta scales correctly (bilinear)" begin
        bilinear_factories = [
            (
                "NMDTBilinear",
                (tol, delta_x, delta_y) -> IOM.NMDTBilinearConfig(;
                    tolerance = tol,
                    max_delta_x = delta_x,
                    max_delta_y = delta_y,
                ),
            ),
            (
                "DNMDTBilinear",
                (tol, delta_x, delta_y) -> IOM.DNMDTBilinearConfig(;
                    tolerance = tol,
                    max_delta_x = delta_x,
                    max_delta_y = delta_y,
                ),
            ),
        ]
        for (name, make) in bilinear_factories
            @testset "$name" begin
                @test make(1e-2, 2.0, 2.0).depth > make(1e-2, 1.0, 1.0).depth
                _sweep_bilinear_tolerance(make, 2.0, 2.0, (1e-1, 1e-2))
            end
        end
    end
end
