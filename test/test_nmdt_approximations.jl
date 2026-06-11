const DNMDT_META = "DNMDTTest"
const DNMDT_HYBS_META = "HybSTest"
const NMDT_META = "NMDTTest"
const NMDT_BILINEAR_META = "NMDTBilinearTest"

@testset "D-NMDT Univariate Approximation" begin
    @testset "Binary expansion correctness" begin
        names = ["gen1"]
        ts = 1:1
        setup = _setup_qa_test(names, ts)
        JuMP.set_lower_bound(setup.var_container["gen1", 1], 0.0)
        JuMP.set_upper_bound(setup.var_container["gen1", 1], 1.0)
        JuMP.fix(setup.var_container["gen1", 1], 0.6; force = true)

        IOM._add_quadratic_approx!(
            IOM.DNMDTQuadConfig(; depth = 4, epigraph_depth = 0),
            setup.container, MockThermalGen, names, ts,
            setup.var_container, [(min = 0.0, max = 1.0)], DNMDT_META,
        )

        JuMP.@objective(setup.jump_model, Min, 0)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

        beta = IOM.get_variable(
            setup.container, IOM.NMDTBinaryVariable, MockThermalGen, DNMDT_META,
        )
        dx = IOM.get_variable(
            setup.container, IOM.NMDTResidualVariable, MockThermalGen, DNMDT_META,
        )

        reconstructed =
            sum(2.0^(-j) * JuMP.value(beta["gen1", j, 1]) for j in 1:4) +
            JuMP.value(dx["gen1", 1])
        @test reconstructed ≈ 0.6 atol = 1e-8
    end

    @testset "Relaxation validity (D-NMDT)" begin
        test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        for x0 in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                IOM._add_quadratic_approx!(
                    IOM.DNMDTQuadConfig(; depth = 3, epigraph_depth = 0),
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, [(min = 0.0, max = 1.0)], DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression,
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0^2
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end
end

@testset "T-D-NMDT Tightening" begin
    @testset "T-D-NMDT lower bound >= D-NMDT lower bound" begin
        for x0 in [0.15, 0.35, 0.65, 0.85]
            lb_dnmdt = NaN
            lb_tdnmdt = NaN
            for tighten in [false, true]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                IOM._add_quadratic_approx!(
                    (
                        if tighten
                            IOM.DNMDTQuadConfig(; depth = 2)
                        else
                            IOM.DNMDTQuadConfig(; depth = 2, epigraph_depth = 0)
                        end
                    ),
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, [(min = 0.0, max = 1.0)], DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression,
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, Min, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

                if !tighten
                    lb_dnmdt = JuMP.objective_value(setup.jump_model)
                else
                    lb_tdnmdt = JuMP.objective_value(setup.jump_model)
                end
            end
            # T-D-NMDT should be at least as tight
            @test lb_tdnmdt >= lb_dnmdt - 1e-6
            # Both should be valid lower bounds
            @test lb_dnmdt <= x0^2 + 1e-6
            @test lb_tdnmdt <= x0^2 + 1e-6
        end
    end

    @testset "Convergence with depth" begin
        true_val = 0.35^2
        errors = Float64[]
        for L in 1:4
            setup = _setup_qa_test(["gen1"], 1:1)
            JuMP.fix(setup.var_container["gen1", 1], 0.35; force = true)

            IOM._add_quadratic_approx!(
                IOM.DNMDTQuadConfig(; depth = L),
                setup.container, MockThermalGen, ["gen1"], 1:1,
                setup.var_container, [(min = 0.0, max = 1.0)], DNMDT_META,
            )
            expr = IOM.get_expression(
                setup.container, IOM.QuadraticExpression,
                MockThermalGen, DNMDT_META,
            )

            JuMP.@objective(setup.jump_model, Max, expr["gen1", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

            push!(errors, abs(JuMP.objective_value(setup.jump_model) - true_val))
        end
        for i in 2:length(errors)
            @test errors[i] <= errors[i - 1] + 1e-10
        end
    end
end

@testset "D-NMDT Bivariate Approximation" begin
    @testset "Relaxation validity" begin
        test_points = [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.DNMDTBilinearConfig(; depth = 2),
                    setup.container, MockThermalGen, ["dev1"], 1:1,
                    setup.x_var_container, setup.y_var_container,
                    [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.BilinearProductExpression,
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
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

    # Closed-form D-NMDT bilinear gap bound is now exercised by the
    # tolerance-dispatch tests in test_tolerance_dispatch.jl.

    @testset "General bounds (non-unit intervals)" begin
        x_min, x_max = 0.2, 0.8
        y_min, y_max = -0.3, 1.5
        test_points = [(0.5, 0.6), (0.3, 1.0), (0.7, -0.1)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.DNMDTBilinearConfig(; depth = 8),
                    setup.container, MockThermalGen, ["dev1"], 1:1,
                    setup.x_var_container, setup.y_var_container,
                    [(min = x_min, max = x_max)], [(min = y_min, max = y_max)], DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.BilinearProductExpression,
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
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

    # @testset "McCormick toggle" begin
    #     setup = _setup_bilinear_test(["dev1"], 1:1)

    #     IOM._add_bilinear_approx!(
    #         setup.container, MockThermalGen, ["dev1"], 1:1,
    #         setup.x_var_container, setup.y_var_container,
    #         0.0, 1.0, 0.0, 1.0, 2, DNMDT_META;
    #         add_mccormick = false,
    #     )

    #     @test !IOM.has_container_key(
    #         setup.container, IOM.McCormickConstraint, MockThermalGen, DNMDT_META,
    #     )
    # end

    @testset "Fixed-variable correctness" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
        JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

        IOM._add_bilinear_approx!(
            IOM.DNMDTBilinearConfig(; depth = 3),
            setup.container, MockThermalGen, ["dev1"], 1:1,
            setup.y_var_container, setup.x_var_container,
            [(min = 0.0, max = 4.0)], [(min = 0.0, max = 4.0)], DNMDT_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.BilinearProductExpression,
            MockThermalGen, DNMDT_META,
        )

        JuMP.@objective(setup.jump_model, Max, expr["dev1", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
    end

    @testset "D-NMDT vs HybS comparison" begin
        true_product = 0.4 * 0.7
        for depth in [2, 3]
            # D-NMDT
            setup_d = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_d.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_d.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_bilinear_approx!(
                IOM.DNMDTBilinearConfig(; depth = depth),
                setup_d.container, MockThermalGen, ["dev1"], 1:1,
                setup_d.x_var_container, setup_d.y_var_container,
                [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], DNMDT_META,
            )
            expr_d = IOM.get_expression(
                setup_d.container, IOM.BilinearProductExpression,
                MockThermalGen, DNMDT_META,
            )

            JuMP.@objective(setup_d.jump_model, Max, expr_d["dev1", 1])
            JuMP.set_optimizer(setup_d.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_d.jump_model)
            JuMP.optimize!(setup_d.jump_model)
            dnmdt_gap = abs(JuMP.objective_value(setup_d.jump_model) - true_product)

            # HybS
            setup_h = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_h.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_h.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_bilinear_approx!(
                IOM.HybSConfig(
                    IOM.SawtoothQuadConfig(; depth = depth);
                    epigraph_depth = depth,
                ),
                setup_h.container, MockThermalGen, ["dev1"], 1:1,
                setup_h.x_var_container, setup_h.y_var_container,
                [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], DNMDT_HYBS_META,
            )
            expr_h = IOM.get_expression(
                setup_h.container, IOM.BilinearProductExpression,
                MockThermalGen, DNMDT_HYBS_META,
            )

            JuMP.@objective(setup_h.jump_model, Max, expr_h["dev1", 1])
            JuMP.set_optimizer(setup_h.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_h.jump_model)
            JuMP.optimize!(setup_h.jump_model)
            hybs_gap = abs(JuMP.objective_value(setup_h.jump_model) - true_product)

            # D-NMDT should be at least as tight (same binary budget)
            @test dnmdt_gap <= hybs_gap + 1e-6

            # Both use same number of binaries: 2L
            n_bin_d = count(JuMP.is_binary, JuMP.all_variables(setup_d.jump_model))
            n_bin_h = count(JuMP.is_binary, JuMP.all_variables(setup_h.jump_model))
            @test n_bin_d == n_bin_h
        end
    end
end

@testset "NMDT Univariate Approximation" begin
    @testset "Binary expansion correctness" begin
        names = ["gen1"]
        ts = 1:1
        setup = _setup_qa_test(names, ts)
        JuMP.set_lower_bound(setup.var_container["gen1", 1], 0.0)
        JuMP.set_upper_bound(setup.var_container["gen1", 1], 1.0)
        JuMP.fix(setup.var_container["gen1", 1], 0.6; force = true)

        IOM._add_quadratic_approx!(
            IOM.NMDTQuadConfig(; depth = 4, epigraph_depth = 0),
            setup.container, MockThermalGen, names, ts,
            setup.var_container, [(min = 0.0, max = 1.0)], NMDT_META,
        )

        JuMP.@objective(setup.jump_model, Min, 0)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

        beta = IOM.get_variable(
            setup.container, IOM.NMDTBinaryVariable, MockThermalGen, NMDT_META,
        )
        dx = IOM.get_variable(
            setup.container, IOM.NMDTResidualVariable, MockThermalGen, NMDT_META,
        )

        reconstructed =
            sum(2.0^(-j) * JuMP.value(beta["gen1", j, 1]) for j in 1:4) +
            JuMP.value(dx["gen1", 1])
        @test reconstructed ≈ 0.6 atol = 1e-8
    end

    @testset "Relaxation validity" begin
        test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        for x0 in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                IOM._add_quadratic_approx!(
                    IOM.NMDTQuadConfig(; depth = 3, epigraph_depth = 0),
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, [(min = 0.0, max = 1.0)], NMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression,
                    MockThermalGen, NMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0^2
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end

    # Closed-form NMDT gap bound is now exercised by the tolerance-dispatch
    # tests in test_tolerance_dispatch.jl.

    # Helper for the tightening tests: solve MIN over the NMDT expression at fixed
    # x0 and return the MIP-optimal value (the lower envelope of result_expr).
    function _nmdt_min(config, x0)
        setup = _setup_qa_test(["gen1"], 1:1)
        JuMP.fix(setup.var_container["gen1", 1], x0; force = true)
        IOM._add_quadratic_approx!(
            config,
            setup.container, MockThermalGen, ["gen1"], 1:1,
            setup.var_container, [(min = 0.0, max = 1.0)], NMDT_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.QuadraticExpression,
            MockThermalGen, NMDT_META,
        )
        JuMP.@objective(setup.jump_model, Min, expr["gen1", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)
        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        return JuMP.objective_value(setup.jump_model)
    end

    # When epigraph_depth = 0, result_expr is the bare NMDT approximation; MIN
    # of result_expr is the NMDT value itself. When epigraph_depth = depth, the
    # tolerance contract holds and result_expr ≥ epigraph(x), so MIN tightens
    # toward x² without exceeding it.
    @testset "NMDT without tightening (epigraph_depth = 0)" begin
        for x0 in [0.15, 0.35, 0.65, 0.85]
            lb_nmdt = _nmdt_min(IOM.NMDTQuadConfig(; depth = 2, epigraph_depth = 0), x0)
            @test lb_nmdt <= x0^2 + 1e-6
        end
    end

    @testset "NMDT with tightening (epigraph_depth = depth)" begin
        for x0 in [0.15, 0.35, 0.65, 0.85]
            depth = 2
            lb_nmdt = _nmdt_min(IOM.NMDTQuadConfig(; depth, epigraph_depth = 0), x0)
            lb_tnmdt =
                _nmdt_min(IOM.NMDTQuadConfig(; depth, epigraph_depth = depth), x0)
            @test lb_tnmdt >= lb_nmdt - 1e-6   # tightening can only raise the MIN
            @test lb_tnmdt <= x0^2 + 1e-6      # but still ≤ x² (overestimator side)
        end
    end
end

@testset "NMDT vs D-NMDT Univariate Comparison" begin
    @testset "D-NMDT tighter than NMDT at same binary count" begin
        # Both use L binary variables for x²; D-NMDT has bound 2^(-2L-2)
        # while NMDT has the looser bound 2^(-L-2).
        # epigraph_depth = 0 on both sides isolates the comparison to the bare
        # NMDT/DNMDT approximation — tightening would muddy the comparison
        # because epigraph(x) is the same for both configs.
        for L in [2, 3]
            gaps_nmdt = Float64[]
            gaps_dnmdt = Float64[]
            for x0 in range(0.05, 0.95; length = 9)
                for (config_fn, tag) in [
                    (L -> IOM.NMDTQuadConfig(; depth = L, epigraph_depth = 0), :nmdt),
                    (L -> IOM.DNMDTQuadConfig(; depth = L, epigraph_depth = 0), :dnmdt),
                ]
                    z_vals = Float64[]
                    for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                        setup = _setup_qa_test(["gen1"], 1:1)
                        JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                        IOM._add_quadratic_approx!(
                            config_fn(L),
                            setup.container, MockThermalGen, ["gen1"], 1:1,
                            setup.var_container, [(min = 0.0, max = 1.0)], NMDT_META,
                        )
                        expr = IOM.get_expression(
                            setup.container, IOM.QuadraticExpression,
                            MockThermalGen, NMDT_META,
                        )

                        JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                        JuMP.set_silent(setup.jump_model)
                        JuMP.optimize!(setup.jump_model)
                        gap = abs(x0^2 - JuMP.objective_value(setup.jump_model))
                        if tag == :nmdt
                            push!(gaps_nmdt, gap)
                        else
                            push!(gaps_dnmdt, gap)
                        end
                    end
                end
            end
            # Both are valid relaxations within their respective bounds
            @test maximum(gaps_nmdt) <= 2.0^(-L - 2) + 1e-6
            @test maximum(gaps_dnmdt) <= 2.0^(-2 * L - 2) + 1e-6
            # D-NMDT is at least as tight as NMDT
            @test maximum(gaps_dnmdt) <= maximum(gaps_nmdt) + 1e-6
        end
    end

    @testset "Same binary count at same depth" begin
        # Both D-NMDT and NMDT univariate use L binary variables for depth=L
        depth = 4
        setup_n = _setup_qa_test(["gen1"], 1:1)
        IOM._add_quadratic_approx!(
            IOM.NMDTQuadConfig(; depth = depth, epigraph_depth = 0),
            setup_n.container, MockThermalGen, ["gen1"], 1:1,
            setup_n.var_container, [(min = 0.0, max = 1.0)], NMDT_META,
        )
        n_bin_nmdt = count(JuMP.is_binary, JuMP.all_variables(setup_n.jump_model))

        setup_d = _setup_qa_test(["gen1"], 1:1)
        IOM._add_quadratic_approx!(
            IOM.DNMDTQuadConfig(; depth = depth, epigraph_depth = 0),
            setup_d.container, MockThermalGen, ["gen1"], 1:1,
            setup_d.var_container, [(min = 0.0, max = 1.0)], DNMDT_META,
        )
        n_bin_dnmdt = count(JuMP.is_binary, JuMP.all_variables(setup_d.jump_model))

        @test n_bin_nmdt == depth
        @test n_bin_dnmdt == depth
        @test n_bin_nmdt == n_bin_dnmdt
    end
end

# NOTE: _add_bilinear_approx! discretizes only x (L binary variables), leaving
# y normalized but continuous. This halves the binary count vs D-NMDT bilinear (2L).
# The trade-off: NMDT bilinear error bound is 2^(-L-2) vs D-NMDT's 2^(-2L-2).
@testset "NMDT Bilinear Approximation" begin
    @testset "Relaxation validity" begin
        test_points = [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.NMDTBilinearConfig(; depth = 3),
                    setup.container, MockThermalGen, ["dev1"], 1:1,
                    setup.x_var_container, setup.y_var_container,
                    [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], NMDT_BILINEAR_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.BilinearProductExpression,
                    MockThermalGen, NMDT_BILINEAR_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
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

    # Closed-form NMDT bilinear gap bound is now exercised by the
    # tolerance-dispatch tests in test_tolerance_dispatch.jl.

    @testset "Fixed-variable correctness" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        JuMP.fix(setup.x_var_container["dev1", 1], 0.75; force = true)
        JuMP.fix(setup.y_var_container["dev1", 1], 0.5; force = true)

        IOM._add_bilinear_approx!(
            IOM.NMDTBilinearConfig(; depth = 4),
            setup.container, MockThermalGen, ["dev1"], 1:1,
            setup.x_var_container, setup.y_var_container,
            [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], NMDT_BILINEAR_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.BilinearProductExpression,
            MockThermalGen, NMDT_BILINEAR_META,
        )

        JuMP.@objective(setup.jump_model, Max, expr["dev1", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        true_val = 0.75 * 0.5
        @test JuMP.objective_value(setup.jump_model) ≈ true_val atol = 2.0^(-4 - 2)
    end
end

@testset "NMDT vs D-NMDT Bilinear: binary efficiency" begin
    @testset "NMDT uses L binaries, D-NMDT uses 2L" begin
        L = 3
        setup_n = _setup_bilinear_test(["dev1"], 1:1)
        IOM._add_bilinear_approx!(
            IOM.NMDTBilinearConfig(; depth = L),
            setup_n.container, MockThermalGen, ["dev1"], 1:1,
            setup_n.x_var_container, setup_n.y_var_container,
            [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], NMDT_BILINEAR_META,
        )
        n_bin_nmdt = count(JuMP.is_binary, JuMP.all_variables(setup_n.jump_model))

        setup_d = _setup_bilinear_test(["dev1"], 1:1)
        IOM._add_bilinear_approx!(
            IOM.DNMDTBilinearConfig(; depth = L),
            setup_d.container, MockThermalGen, ["dev1"], 1:1,
            setup_d.x_var_container, setup_d.y_var_container,
            [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], DNMDT_META,
        )
        n_bin_dnmdt = count(JuMP.is_binary, JuMP.all_variables(setup_d.jump_model))

        # NMDT discretizes only x (L bins); D-NMDT discretizes x and y (2L bins)
        @test n_bin_nmdt == L
        @test n_bin_dnmdt == 2 * L
        @test n_bin_nmdt == n_bin_dnmdt ÷ 2
    end

    @testset "D-NMDT tighter than NMDT at same L" begin
        true_product = 0.4 * 0.7
        for L in [2, 3]
            setup_n = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_n.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_n.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_bilinear_approx!(
                IOM.NMDTBilinearConfig(; depth = L),
                setup_n.container, MockThermalGen, ["dev1"], 1:1,
                setup_n.x_var_container, setup_n.y_var_container,
                [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], NMDT_BILINEAR_META,
            )
            expr_n = IOM.get_expression(
                setup_n.container, IOM.BilinearProductExpression,
                MockThermalGen, NMDT_BILINEAR_META,
            )

            JuMP.@objective(setup_n.jump_model, Max, expr_n["dev1", 1])
            JuMP.set_optimizer(setup_n.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_n.jump_model)
            JuMP.optimize!(setup_n.jump_model)
            nmdt_gap = abs(JuMP.objective_value(setup_n.jump_model) - true_product)

            setup_d = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_d.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_d.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_bilinear_approx!(
                IOM.DNMDTBilinearConfig(; depth = L),
                setup_d.container, MockThermalGen, ["dev1"], 1:1,
                setup_d.x_var_container, setup_d.y_var_container,
                [(min = 0.0, max = 1.0)], [(min = 0.0, max = 1.0)], DNMDT_META,
            )
            expr_d = IOM.get_expression(
                setup_d.container, IOM.BilinearProductExpression,
                MockThermalGen, DNMDT_META,
            )

            JuMP.@objective(setup_d.jump_model, Max, expr_d["dev1", 1])
            JuMP.set_optimizer(setup_d.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_d.jump_model)
            JuMP.optimize!(setup_d.jump_model)
            dnmdt_gap = abs(JuMP.objective_value(setup_d.jump_model) - true_product)

            # NMDT bound: 2^(-L-2); D-NMDT bound: 2^(-2L-2) — D-NMDT is tighter at same L
            @test nmdt_gap <= 2.0^(-L - 2) + 1e-6
            @test dnmdt_gap <= 2.0^(-2 * L - 2) + 1e-6
            @test dnmdt_gap <= nmdt_gap + 1e-6
        end
    end
end

@testset "NMDT/DNMDT epigraph tightening on non-[0,1] domains" begin
    # Regression: `_tighten_lower_bounds!` previously applied the normalized [0,1]
    # epigraph cut directly to the unnormalized `result_expr`. On any domain other
    # than exactly [0,1] this is invalid — on [-1,1] at x=0 it makes the model
    # infeasible, and on [0,2] it collapses the lower envelope (massively
    # under-estimating x²). Tightening is on by default (epigraph_depth = 3*depth).
    for (config_fn, tag) in [
        (depth -> IOM.NMDTQuadConfig(; depth), "NMDT"),
        (depth -> IOM.DNMDTQuadConfig(; depth), "DNMDT"),
    ]
        @testset "$tag domain [-1, 1] feasible at x = 0" begin
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], 0.0; force = true)
                IOM._add_quadratic_approx!(
                    config_fn(3),
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, [(min = -1.0, max = 1.0)], NMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression, MockThermalGen, NMDT_META,
                )
                JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                # Pre-fix: the misframed cut made this INFEASIBLE at x = 0.
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                # x² = 0 at x = 0; the relaxation must bracket it.
                if sense == JuMP.MIN_SENSE
                    @test JuMP.objective_value(setup.jump_model) <= 1e-6
                else
                    @test JuMP.objective_value(setup.jump_model) >= -1e-6
                end
            end
        end

        @testset "$tag domain [0, 2] lower envelope not collapsed at x = 1" begin
            setup = _setup_qa_test(["gen1"], 1:1)
            JuMP.fix(setup.var_container["gen1", 1], 1.0; force = true)
            IOM._add_quadratic_approx!(
                config_fn(3),
                setup.container, MockThermalGen, ["gen1"], 1:1,
                setup.var_container, [(min = 0.0, max = 2.0)], NMDT_META,
            )
            expr = IOM.get_expression(
                setup.container, IOM.QuadraticExpression, MockThermalGen, NMDT_META,
            )
            JuMP.@objective(setup.jump_model, Min, expr["gen1", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            lb = JuMP.objective_value(setup.jump_model)
            # Valid lower bound on x² = 1 ...
            @test lb <= 1.0 + 1e-6
            # ... and with default tightening (McCormick lower bounds dropped) the
            # unnormalized epigraph cut must hold the envelope near x², not collapse it.
            @test lb >= 1.0 - 0.1
        end
    end
end
