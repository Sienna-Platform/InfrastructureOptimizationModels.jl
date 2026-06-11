const MOI = JuMP.MOI
const TEST_META = "TestVar"

@testset "Quadratic Approximations" begin
    @testset "PWMCC segment validation" begin
        # PWMCC chord cuts are valid only when every sub-segment boundary coincides
        # with a PWL breakpoint, i.e. the McCormickTightener partitions evenly divide
        # depth. A misaligned grid such as (depth=3, partitions=2) cuts off
        # MIP-feasible points, so the constructor must reject it.
        for T in
            (IOM.SOS2QuadConfig{IOM.SolverBackend}, IOM.SOS2QuadConfig{IOM.ManualBackend})
            # Non-dividing: rejected.
            @test_throws ArgumentError T(; depth = 3, tightener = IOM.McCormickTightener(2))
            # Dividing: accepted.
            @test T(;
                depth = 4,
                tightener = IOM.McCormickTightener(2),
            ).tightener.partitions ==
                  2
            # NoTightener disables PWMCC entirely and is always accepted (also the default).
            @test T(; depth = 3, tightener = IOM.NoTightener()).tightener isa
                  IOM.NoTightener
            @test T(; depth = 3).tightener isa IOM.NoTightener
        end
    end

    @testset "Solver SOS2" begin
        @testset "Solve min x^2 - 4x" begin
            # Analytic minimum of x^2 - 4x at x=2, value = -4
            # With breakpoints at 0,1,2,3,4 the approximation is exact at breakpoints
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM.add_quadratic_approx!(
                IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 4),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # Objective: x^2 - 4x
            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM.add_quadratic_approx!(
                IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 4),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # x^2 + y = 10 → with x=3, x^2=9, y=1
            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end

        @testset "Approximation quality improves with more segments" begin
            # min (√2)x² - (√3)x on [0, 6], analytic minimum at x=√3/8
            analytic_min = sqrt(2) * (sqrt(3 / 8)^2) - sqrt(3) * sqrt(3 / 8)
            errors = Float64[]
            for num_segments in 2 .^ (1:6)
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 6.0)

                IOM.add_quadratic_approx!(
                    IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = num_segments),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.var_container,
                    [(min = 0.0, max = 6.0)],
                    TEST_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.QuadraticExpression,
                    MockThermalGen,
                    TEST_META,
                )
                x_sq = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - analytic_min))
            end
            # Each doubling of segments should reduce error (or maintain if already exact)
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Manual SOS2" begin
        @testset "Solve min x^2 - 4x" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM.add_quadratic_approx!(
                IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 4),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM.add_quadratic_approx!(
                IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 4),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end
    end

    @testset "Sawtooth" begin
        @testset "Solve min x^2 - 4x" begin
            # depth=2 → breakpoints at 0,1,2,3,4 → exact at x=2
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM.add_quadratic_approx!(
                IOM.SawtoothQuadConfig(; depth = 2),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM.add_quadratic_approx!(
                IOM.SawtoothQuadConfig(; depth = 2),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 4.0)],
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression,
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end

        @testset "Approximation quality improves with depth" begin
            # min (√2)x² - (√3)x on [0, 6], analytic minimum at x=√3/8
            analytic_min = sqrt(2) * (sqrt(3 / 8)^2) - sqrt(3) * sqrt(3 / 8)
            errors = Float64[]
            for depth in 1:6
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 6.0)

                IOM.add_quadratic_approx!(
                    IOM.SawtoothQuadConfig(; depth = depth),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.var_container,
                    [(min = 0.0, max = 6.0)],
                    TEST_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.QuadraticExpression,
                    MockThermalGen,
                    TEST_META,
                )
                x_sq = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - analytic_min))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end

        @testset "Agrees with SOS2 at aligned breakpoints" begin
            # SOS2 and Sawtooth should agree when n_sos2_segments = 2^(sawtooth_depth)
            for depth in 1:6
                sos2_value = nothing
                for method in [:sos2, :sawtooth]
                    setup = _setup_qa_test(["dev1"], 1:1)
                    x_var = setup.var_container["dev1", 1]
                    JuMP.set_lower_bound(x_var, 0.0)
                    JuMP.set_upper_bound(x_var, 4.0)

                    if method == :sos2
                        IOM.add_quadratic_approx!(
                            IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 2^depth),
                            setup.container,
                            MockThermalGen,
                            ["dev1"],
                            1:1,
                            setup.var_container,
                            [(min = 0.0, max = 4.0)],
                            TEST_META,
                        )
                    else
                        IOM.add_quadratic_approx!(
                            IOM.SawtoothQuadConfig(; depth = depth),
                            setup.container,
                            MockThermalGen,
                            ["dev1"],
                            1:1,
                            setup.var_container,
                            [(min = 0.0, max = 4.0)],
                            TEST_META,
                        )
                    end
                    expr_container = IOM.get_expression(
                        setup.container,
                        IOM.QuadraticExpression,
                        MockThermalGen,
                        TEST_META,
                    )
                    x_sq = expr_container["dev1", 1]

                    JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                    JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                    JuMP.set_silent(setup.jump_model)
                    JuMP.optimize!(setup.jump_model)

                    if method == :sos2
                        sos2_value = JuMP.objective_value(setup.jump_model)
                    else
                        sawtooth_value = JuMP.objective_value(setup.jump_model)
                        @test sos2_value ≈ sawtooth_value atol = 1e-6
                    end
                end
            end
        end
    end
end
