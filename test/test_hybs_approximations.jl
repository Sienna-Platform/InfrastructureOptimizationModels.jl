const HYBS_META = "HybSTest"
const HYBS_BILINEAR_META = "BilinearTest"

@testset "Epigraph Quadratic Approximation" begin
    @testset "Lower-bounds x^2 on [0,1]" begin
        setup = _setup_qa_test(["dev1"], 1:1)
        x_var = setup.var_container["dev1", 1]
        JuMP.fix(x_var, 0.35; force = true)

        IOM._add_quadratic_approx!(
            IOM.EpigraphQuadConfig(; depth = 4),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.var_container,
            [(min = 0.0, max = 1.0)],
            HYBS_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.EpigraphExpression,
            MockThermalGen,
            HYBS_META,
        )
        z_epi = expr_container["dev1", 1]

        JuMP.@objective(setup.jump_model, Min, z_epi)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) <= 0.1225 + 1e-8
        @test JuMP.objective_value(setup.jump_model) >= 0.1225 - 0.02
    end

    @testset "Lower-bounds x^2 on non-unit interval [0,2]" begin
        setup = _setup_qa_test(["dev1"], 1:1)
        x_var = setup.var_container["dev1", 1]
        JuMP.fix(x_var, 1.3; force = true)

        IOM._add_quadratic_approx!(
            IOM.EpigraphQuadConfig(; depth = 4),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.var_container,
            [(min = 0.0, max = 2.0)],
            HYBS_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.EpigraphExpression,
            MockThermalGen,
            HYBS_META,
        )
        z_epi = expr_container["dev1", 1]

        JuMP.@objective(setup.jump_model, Min, z_epi)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) <= 1.69 + 1e-8
        @test JuMP.objective_value(setup.jump_model) >= 1.69 - 0.1
    end

    @testset "Approximation quality improves with depth" begin
        errors = Float64[]
        true_val = 0.35^2
        for depth in 1:6
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 0.35; force = true)

            IOM._add_quadratic_approx!(
                IOM.EpigraphQuadConfig(; depth = depth),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                [(min = 0.0, max = 1.0)],
                HYBS_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.EpigraphExpression,
                MockThermalGen,
                HYBS_META,
            )
            z_epi = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_epi)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            obj_val = JuMP.objective_value(setup.jump_model)
            push!(errors, abs(obj_val - true_val))
        end
        for i in 2:length(errors)
            @test errors[i] <= errors[i - 1] + 1e-10
        end
    end
end

@testset "HybS Bilinear Approximation" begin
    @testset "Brackets true product at interior points" begin
        test_points = [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.HybSConfig(IOM.SawtoothQuadConfig(; depth = 2); epigraph_depth = 2),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    [(min = 0.0, max = 1.0)],
                    [(min = 0.0, max = 1.0)],
                    HYBS_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression,
                    MockThermalGen,
                    HYBS_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, sense, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0 * y0
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end

    @testset "Fixed-variable correctness" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        x_var = setup.x_var_container["dev1", 1]
        y_var = setup.y_var_container["dev1", 1]
        JuMP.fix(x_var, 2.0; force = true)
        JuMP.fix(y_var, 3.0; force = true)

        IOM._add_bilinear_approx!(
            IOM.HybSConfig(IOM.SawtoothQuadConfig(; depth = 3); epigraph_depth = 3),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            [(min = 0.0, max = 4.0)],
            [(min = 0.0, max = 4.0)],
            HYBS_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.BilinearProductExpression,
            MockThermalGen,
            HYBS_META,
        )
        z_expr = expr_container["dev1", 1]

        JuMP.@objective(setup.jump_model, Max, z_expr)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
    end

    @testset "Constraint usage: x*y + w = 10 with x=2, y=3" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        x_var = setup.x_var_container["dev1", 1]
        y_var = setup.y_var_container["dev1", 1]
        JuMP.fix(x_var, 2.0; force = true)
        JuMP.fix(y_var, 3.0; force = true)

        w = JuMP.@variable(setup.jump_model, base_name = "w")

        IOM._add_bilinear_approx!(
            IOM.HybSConfig(IOM.SawtoothQuadConfig(; depth = 3); epigraph_depth = 3),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            [(min = 0.0, max = 4.0)],
            [(min = 0.0, max = 4.0)],
            HYBS_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.BilinearProductExpression,
            MockThermalGen,
            HYBS_META,
        )
        z_expr = expr_container["dev1", 1]

        JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
        JuMP.@objective(setup.jump_model, Min, w)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.value(w) ≈ 4.0 atol = 0.5
    end

    @testset "Approximation quality improves with depth" begin
        true_product = 0.4 * 0.7
        errors = Float64[]
        for depth in 1:5
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_bilinear_approx!(
                IOM.HybSConfig(
                    IOM.SawtoothQuadConfig(; depth = depth);
                    epigraph_depth = depth,
                ),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 0.0, max = 1.0)],
                [(min = 0.0, max = 1.0)],
                HYBS_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                HYBS_META,
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

    @testset "Non-unit bounds" begin
        true_product = 3.5 * 2.1
        z_vals = Float64[]
        for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 3.5; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 2.1; force = true)

            IOM._add_bilinear_approx!(
                IOM.HybSConfig(IOM.SawtoothQuadConfig(; depth = 3); epigraph_depth = 3),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                [(min = 2.0, max = 5.0)],
                [(min = 1.0, max = 3.0)],
                HYBS_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression,
                MockThermalGen,
                HYBS_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, sense, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            push!(z_vals, JuMP.objective_value(setup.jump_model))
        end
        @test z_vals[1] <= true_product + 1e-6
        @test z_vals[2] >= true_product - 1e-6
    end

    @testset "Vertex optimum" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        x_var = setup.x_var_container["dev1", 1]
        y_var = setup.y_var_container["dev1", 1]
        JuMP.set_lower_bound(x_var, 0.0)
        JuMP.set_upper_bound(x_var, 4.0)
        JuMP.set_lower_bound(y_var, 0.0)
        JuMP.set_upper_bound(y_var, 4.0)

        IOM._add_bilinear_approx!(
            IOM.HybSConfig(IOM.SawtoothQuadConfig(; depth = 2); epigraph_depth = 2),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            [(min = 0.0, max = 4.0)],
            [(min = 0.0, max = 4.0)],
            HYBS_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.BilinearProductExpression,
            MockThermalGen,
            HYBS_META,
        )
        z_expr = expr_container["dev1", 1]

        # min x*y on [0,4]^2 is 0 at corner (0,*) or (*,0)
        JuMP.@objective(setup.jump_model, Min, z_expr)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4
    end

    @testset "HybS uses fewer binaries than Bin2" begin
        for depth in [1, 2, 4]
            # HybS
            setup_h = _setup_bilinear_test(["dev1"], 1:1)
            IOM._add_bilinear_approx!(
                IOM.HybSConfig(
                    IOM.SawtoothQuadConfig(; depth = depth);
                    epigraph_depth = depth,
                ),
                setup_h.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup_h.x_var_container,
                setup_h.y_var_container,
                [(min = 0.0, max = 1.0)],
                [(min = 0.0, max = 1.0)],
                HYBS_META,
            )
            n_bin_hybs =
                count(JuMP.is_binary, JuMP.all_variables(setup_h.jump_model))

            # Bin2 (sawtooth)
            setup_b = _setup_bilinear_test(["dev1"], 1:1)
            IOM._add_bilinear_approx!(
                IOM.Bin2Config(IOM.SawtoothQuadConfig(; depth = depth)),
                setup_b.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup_b.x_var_container,
                setup_b.y_var_container,
                [(min = 0.0, max = 1.0)],
                [(min = 0.0, max = 1.0)],
                HYBS_BILINEAR_META,
            )
            n_bin_bin2 =
                count(JuMP.is_binary, JuMP.all_variables(setup_b.jump_model))

            @test n_bin_hybs == 2 * depth
            @test n_bin_bin2 == 3 * depth
            @test n_bin_hybs < n_bin_bin2
        end
    end
end
