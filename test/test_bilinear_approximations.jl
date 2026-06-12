const BILINEAR_META = "BilinearTest"

@testset "Bilinear Approximations" begin
    @testset "Bin2 + Solver SOS2" begin
        @testset "Reformulated McCormick tightens LP relaxation" begin
            # At interior point x=2.5, y=1.5, true product = 3.75
            # Compare max z with and without reformulated McCormick
            true_product = 2.5 * 1.5
            gaps = Float64[]
            for add_mc in [false, true]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], 2.5; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], 1.5; force = true)

                IOM.add_bilinear_approx!(
                    IOM.Bin2Config(
                        IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 4);
                        tightener = add_mc ? IOM.McCormickTightener() : IOM.NoTightener(),
                    ),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 4.0)],
                    [(min = 0.0, max = 4.0)],
                    BILINEAR_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression,
                    MockThermalGen,
                    BILINEAR_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(gaps, abs(JuMP.objective_value(setup.jump_model) - true_product))
            end
            # With McCormick (gaps[2]) should be <= without (gaps[1])
            @test gaps[2] <= gaps[1] + 1e-10
        end

        @testset "Fixed-variable correctness" begin
            # Fix x=3, y ∈ [0,4]: min xy should give z≈0 at y=0
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4

            # Fix x=2, y=6: z should be exactly 6
            setup2 = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup2.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup2.y_var_container["dev1", 1], 3.0; force = true)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 8)),
                setup2.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup2.x_var_container,
                setup2.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container2 = IOM.get_expression(
                setup2.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr2 = expr_container2["dev1", 1]

            JuMP.@objective(setup2.jump_model, Max, z_expr2)
            JuMP.set_optimizer(setup2.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup2.jump_model)
            JuMP.optimize!(setup2.jump_model)

            @test JuMP.termination_status(setup2.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup2.jump_model) ≈ 6.0 atol = 1e-6
        end

        @testset "Constraint usage: x·y + w = 10 with x=2" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.fix(x_var, 2.0; force = true)
            JuMP.fix(y_var, 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            # x·y + w = 10 → 2·3 + w = 10 → w = 4
            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-4
        end

        @testset "Vertex optimum" begin
            # min x·y on [0,4]×[0,4]; minimum is 0 at a corner
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4
        end

        @testset "Approximation quality improves with segments" begin
            # Fix x=2.5, y=1.5: true product = 3.75
            # Sweep segments, verify gap shrinks
            true_product = 2.5 * 1.5
            errors = Float64[]
            for num_segments in 2 .^ (1:5)
                setup = _setup_bilinear_test(["dev1"], 1:1)
                x_var = setup.x_var_container["dev1", 1]
                y_var = setup.y_var_container["dev1", 1]
                JuMP.fix(x_var, 2.5; force = true)
                JuMP.fix(y_var, 1.5; force = true)

                IOM.add_bilinear_approx!(
                    IOM.Bin2Config(
                        IOM.SOS2QuadConfig{IOM.SolverBackend}(;
                            depth = num_segments,
                        ),
                    ),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 4.0)],
                    [(min = 0.0, max = 4.0)],
                    BILINEAR_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression,
                    MockThermalGen,
                    BILINEAR_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Bin2 + Manual SOS2" begin
        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-4
        end

        @testset "Constraint usage: x·y + w = 10 with x=2, y=3" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-4
        end

        @testset "Vertex optimum" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 8)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4
        end

        @testset "Approximation quality improves with segments" begin
            true_product = 2.5 * 1.5
            errors = Float64[]
            for num_segments in 2 .^ (1:5)
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], 2.5; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], 1.5; force = true)

                IOM.add_bilinear_approx!(
                    IOM.Bin2Config(
                        IOM.SOS2QuadConfig{IOM.ManualBackend}(;
                            depth = num_segments,
                        ),
                    ),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 4.0)],
                    [(min = 0.0, max = 4.0)],
                    BILINEAR_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression,
                    MockThermalGen,
                    BILINEAR_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Bin2 + Sawtooth" begin
        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SawtoothQuadConfig(; depth = 3)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-3
        end

        @testset "Constraint usage: x·y + w = 10 with x=2, y=3" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SawtoothQuadConfig(; depth = 3)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-3
        end

        @testset "Vertex optimum" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            IOM.add_bilinear_approx!(
                IOM.Bin2Config(IOM.SawtoothQuadConfig(; depth = 3)),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 4.0)],
                [(min = 0.0, max = 4.0)],
                BILINEAR_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-3
        end

        @testset "Approximation quality improves with depth" begin
            true_product = 2.5 * 1.5
            errors = Float64[]
            for depth in 1:5
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], 2.5; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], 1.5; force = true)

                IOM.add_bilinear_approx!(
                    IOM.Bin2Config(IOM.SawtoothQuadConfig(; depth = depth)),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 4.0)],
                    [(min = 0.0, max = 4.0)],
                    BILINEAR_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression,
                    MockThermalGen,
                    BILINEAR_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "McCormick-only bilinear config" begin
        @testset "Exact at box corners (max over the envelope)" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x = setup.x_var_container["dev1", 1]
            y = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x, 0.0)
            JuMP.set_upper_bound(x, 4.0)
            JuMP.set_lower_bound(y, 0.0)
            JuMP.set_upper_bound(y, 4.0)

            z_expr =
                IOM.add_bilinear_approx!(
                    IOM.McCormickBilinearConfig(),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 4.0)],
                    [(min = 0.0, max = 4.0)],
                    BILINEAR_META,
                )[
                    "dev1",
                    1,
                ]

            # Result is published in a BilinearProductExpression container.
            @test IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                BILINEAR_META,
            )[
                "dev1",
                1,
            ] == z_expr

            # The McCormick envelope's maximum is attained at the corner x=y=4 where z = x·y = 16.
            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 16.0 atol = 1e-6
        end
    end

    @testset "Binary×continuous McCormick exactness" begin
        # `_add_binary_continuous_mccormick!` drops the lower inequality made redundant by the
        # auxiliary variable's lower bound. With β ∈ {0,1} the linearization must remain exact:
        # z = β·x at both the min and max of z, for every fixed (β, x).
        for beta_fixed in (0.0, 1.0), x_fixed in (0.0, 1.5, 4.0)
            setup = _setup_bilinear_test(["d"], 1:1)
            x = setup.x_var_container["d", 1]
            beta = setup.y_var_container["d", 1]
            JuMP.fix(x, x_fixed; force = true)
            JuMP.fix(beta, beta_fixed; force = true)
            z = JuMP.@variable(setup.jump_model, lower_bound = 0.0, upper_bound = 4.0)
            cons = IOM.add_constraints_container!(
                setup.container,
                IOM.McCormickConstraint,
                MockThermalGen,
                ["d"],
                1:4,
                1:1;
                sparse = true,
                meta = "bc",
            )
            IOM._add_binary_continuous_mccormick!(
                setup.jump_model, cons, ("d", 1), x, beta, z, 0.0, 4.0,
            )
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            expected = beta_fixed * x_fixed
            for sense in (JuMP.MAX_SENSE, JuMP.MIN_SENSE)
                JuMP.@objective(setup.jump_model, sense, z)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                @test JuMP.objective_value(setup.jump_model) ≈ expected atol = 1e-6
            end
        end
    end
end
