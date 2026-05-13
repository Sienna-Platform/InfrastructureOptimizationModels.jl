# Pure-JuMP tests for the approximation layer.
#
# These tests exercise `build_quadratic_approx` and `build_bilinear_approx`
# directly against a bare `JuMP.Model`. No OptimizationContainer is involved
# anywhere — these tests would pass if `OptimizationContainer` were removed
# from the package.
#
# The point of having this layer separately testable is that mathematical
# properties of an approximation method (lower-boundedness, exactness at
# breakpoints, McCormick envelope feasibility, etc.) can be checked without
# any of the IOM container scaffolding getting in the way.

function _pure_jump_scalar_var(model::JuMP.Model, lb::Float64, ub::Float64; name::String)
    # The DenseAxisArray is indexed by a single device named "dev1" and a single
    # time step. `name` is only the JuMP base_name (for readability in logs).
    x = JuMP.@variable(model, base_name = name, lower_bound = lb, upper_bound = ub)
    return JuMP.Containers.DenseAxisArray(reshape([x], 1, 1), ["dev1"], 1:1)
end

@testset "Pure-JuMP Approximations" begin
    @testset "no_approx_quadratic returns exact x²" begin
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 4.0; name = "x")
        bounds = [(min = 0.0, max = 4.0)]
        result = IOM.build_quadratic_approx(IOM.NoQuadApproxConfig(), model, x, bounds)
        @test IOM.get_approximation(result) === result.approximation
        # The expression should be x*x, a QuadExpr
        expr = result.approximation["dev1", 1]
        @test expr isa JuMP.QuadExpr
    end

    @testset "solver_sos2 is exact at breakpoints" begin
        # With depth=4 breakpoints are at {0, 1, 2, 3, 4}, so x² is exact at x=2.
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 4.0; name = "x")
        bounds = [(min = 0.0, max = 4.0)]
        result = IOM.build_quadratic_approx(
            IOM.SolverSOS2QuadConfig(4, 0), model, x, bounds,
        )
        JuMP.fix(x["dev1", 1], 2.0; force = true)
        JuMP.@objective(model, Min, result.approximation["dev1", 1])
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.value(result.approximation["dev1", 1]) ≈ 4.0 atol = 1e-6
    end

    @testset "solver_sos2 minimizes x² − 4x correctly" begin
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 4.0; name = "x")
        bounds = [(min = 0.0, max = 4.0)]
        result = IOM.build_quadratic_approx(
            IOM.SolverSOS2QuadConfig(4, 0), model, x, bounds,
        )
        JuMP.@objective(model, Min, result.approximation["dev1", 1] - 4.0 * x["dev1", 1])
        JuMP.optimize!(model)
        @test JuMP.value(x["dev1", 1]) ≈ 2.0 atol = 1e-6
        @test JuMP.objective_value(model) ≈ -4.0 atol = 1e-6
    end

    @testset "epigraph lower-bounds x² uniformly" begin
        # The epigraph relaxation is a pure-LP lower bound on x².
        # Sample x at five interior points, minimize z = approximation,
        # and verify z(x) ≤ x² + tiny tolerance.
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
        bounds = [(min = 0.0, max = 1.0)]
        result = IOM.build_quadratic_approx(IOM.EpigraphQuadConfig(4), model, x, bounds)
        for sample in [0.1, 0.3, 0.5, 0.7, 0.9]
            JuMP.fix(x["dev1", 1], sample; force = true)
            JuMP.@objective(model, Min, result.approximation["dev1", 1])
            JuMP.optimize!(model)
            @test JuMP.termination_status(model) == JuMP.OPTIMAL
            @test JuMP.value(result.approximation["dev1", 1]) <= sample^2 + 1e-8
        end
    end

    @testset "epigraph quality improves monotonically with depth" begin
        # Pure-JuMP version of the test that previously had to be done via the
        # container layer. The error at x=0.35 should shrink as depth grows.
        errors = Float64[]
        for depth in 1:6
            model = JuMP.Model(HiGHS.Optimizer)
            JuMP.set_silent(model)
            x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
            bounds = [(min = 0.0, max = 1.0)]
            result = IOM.build_quadratic_approx(
                IOM.EpigraphQuadConfig(depth), model, x, bounds,
            )
            JuMP.fix(x["dev1", 1], 0.35; force = true)
            JuMP.@objective(model, Min, result.approximation["dev1", 1])
            JuMP.optimize!(model)
            push!(errors, abs(JuMP.objective_value(model) - 0.35^2))
        end
        for i in 2:length(errors)
            @test errors[i] <= errors[i - 1] + 1e-10
        end
    end

    @testset "sawtooth is exact at breakpoints" begin
        # With depth=3 breakpoints are at the dyadic rationals on [0,1].
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
        bounds = [(min = 0.0, max = 1.0)]
        result = IOM.build_quadratic_approx(IOM.SawtoothQuadConfig(3, 0), model, x, bounds)
        # x = 0.5 is at a breakpoint, x² = 0.25 exactly.
        JuMP.fix(x["dev1", 1], 0.5; force = true)
        JuMP.@objective(model, Min, result.approximation["dev1", 1])
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.value(result.approximation["dev1", 1]) ≈ 0.25 atol = 1e-6
    end

    @testset "NMDT discretizes xh correctly" begin
        # Stand-alone test of the shared NMDT discretization step.
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 4.0; name = "x")
        bounds = [(min = 0.0, max = 4.0)]
        disc = IOM.build_discretization(model, x, bounds, 3)
        # At x=2 (xh=0.5), expect β_1 = 1, β_2 = 0, β_3 = 0, δ = 0.
        JuMP.fix(x["dev1", 1], 2.0; force = true)
        JuMP.@objective(model, Min, disc.delta_var["dev1", 1])
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        # The discretization must reproduce xh = 0.5:
        b1 = JuMP.value(disc.beta_var["dev1", 1, 1])
        b2 = JuMP.value(disc.beta_var["dev1", 2, 1])
        b3 = JuMP.value(disc.beta_var["dev1", 3, 1])
        d = JuMP.value(disc.delta_var["dev1", 1])
        @test 0.5 * b1 + 0.25 * b2 + 0.125 * b3 + d ≈ 0.5 atol = 1e-6
    end

    @testset "no_approx_bilinear returns exact x·y" begin
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 2.0; name = "x")
        y = _pure_jump_scalar_var(model, 0.0, 2.0; name = "y")
        x_bounds = [(min = 0.0, max = 2.0)]
        y_bounds = [(min = 0.0, max = 2.0)]
        result = IOM.build_bilinear_approx(
            IOM.NoBilinearApproxConfig(), model, x, y, x_bounds, y_bounds,
        )
        expr = result.approximation["dev1", 1]
        @test expr isa JuMP.QuadExpr
    end

    @testset "McCormick envelope bracketing on x·y" begin
        # Standard McCormick envelope on [0,1]² brackets the true x·y at corners.
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
        y = _pure_jump_scalar_var(model, 0.0, 1.0; name = "y")
        z = JuMP.@variable(model, base_name = "z")
        z_arr = JuMP.Containers.DenseAxisArray(reshape([z], 1, 1), ["dev1"], 1:1)
        IOM.build_mccormick_envelope(
            model, x, y, z_arr,
            [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)],
        )
        # At x=y=1, the only feasible z is 1.
        JuMP.fix(x["dev1", 1], 1.0; force = true)
        JuMP.fix(y["dev1", 1], 1.0; force = true)
        JuMP.@objective(model, Min, z)
        JuMP.optimize!(model)
        @test JuMP.value(z) ≈ 1.0 atol = 1e-6
    end

    @testset "Bin2 z = ½(p² − x² − y²) identity" begin
        # When the underlying quadratic method is exact at the queried point,
        # Bin2 reproduces x·y exactly.
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
        y = _pure_jump_scalar_var(model, 0.0, 1.0; name = "y")
        x_bounds = [(min = 0.0, max = 1.0)]
        y_bounds = [(min = 0.0, max = 1.0)]
        # depth=2 places breakpoints at {0, 0.5, 1.0}; pick a point at the corners.
        result = IOM.build_bilinear_approx(
            IOM.Bin2Config(IOM.SolverSOS2QuadConfig(2, 0), false),
            model, x, y, x_bounds, y_bounds,
        )
        JuMP.fix(x["dev1", 1], 1.0; force = true)
        JuMP.fix(y["dev1", 1], 1.0; force = true)
        JuMP.@objective(model, Min, result.approximation["dev1", 1])
        JuMP.optimize!(model)
        @test JuMP.value(result.approximation["dev1", 1]) ≈ 1.0 atol = 1e-6
    end

    @testset "get_approximation returns the approximation field" begin
        model = JuMP.Model()
        x = _pure_jump_scalar_var(model, 0.0, 1.0; name = "x")
        bounds = [(min = 0.0, max = 1.0)]
        result = IOM.build_quadratic_approx(IOM.EpigraphQuadConfig(2), model, x, bounds)
        @test IOM.get_approximation(result) === result.approximation
    end
end
