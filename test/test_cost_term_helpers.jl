"""
Unit tests for cost term helpers in src/objective_function/cost_term_helpers.jl.
Tests the generic building blocks for adding costs to expressions and objectives.

Uses common helpers from test_utils/objective_function_helpers.jl:
- make_test_container, add_test_variable!, add_test_expression!, add_test_parameter!
Test types defined in test_utils/test_types.jl.
"""

@testset "Cost Term Helpers" begin
    @testset "add_cost_term_invariant!" begin
        @testset "adds cost to invariant objective" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            rate = 10.0
            cost = IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify cost expression is var * rate
            @test cost == var * rate

            # Verify it was added to invariant objective
            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.coefficient(invariant, var) ≈ rate

            # Verify variant is empty
            variant = IOM.get_variant_terms(obj)
            @test JuMP.coefficient(variant, var) ≈ 0.0
        end

        @testset "adds cost to expression if present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)
            add_test_expression!(
                container,
                TestCostExpression,
                MockThermalGen,
                ["gen1"],
                1:3,
            )

            rate = 15.0
            IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify expression was updated
            expr_container =
                IOM.get_expression(container, TestCostExpression, MockThermalGen)
            expr = expr_container["gen1", 1]
            @test JuMP.coefficient(expr, var) ≈ rate
        end

        @testset "skips expression if not present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)
            # Don't add expression container

            rate = 10.0
            # Should not error even without expression container
            cost = IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost == var * rate
        end

        @testset "handles zero rate" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            cost = IOM.add_cost_term_invariant!(
                container, var, 0.0, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost == 0.0
        end

        @testset "handles scalar quantity (Float64)" begin
            container = make_test_container(1:3)
            add_test_expression!(
                container,
                TestCostExpression,
                MockThermalGen,
                ["gen1"],
                1:3,
            )

            quantity = 5.0
            rate = 10.0
            cost = IOM.add_cost_term_invariant!(
                container, quantity, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost ≈ 50.0

            # Verify constant was added to objective
            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.constant(invariant) ≈ 50.0
        end
    end

    @testset "add_cost_term_variant! (parameter overload)" begin
        @testset "adds cost to variant objective using parameter rate" begin
            container = make_test_container(1:3)
            var = add_test_variable!(
                container, TestCostVariable, MockThermalGen, "gen1", 1,
            )

            # Parameter value = 20.0, multiplier defaults to 1.0
            # So rate = param * mult = 20.0, cost = var * 20.0
            param_values = [20.0 20.0 20.0]
            add_test_parameter!(
                container, TestCostParameter, MockThermalGen,
                ["gen1"], 1:3, param_values,
            )

            cost = IOM.add_cost_term_variant!(
                container, var, TestCostParameter,
                TestCostExpression, MockThermalGen, "gen1", 1,
            )

            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)
            @test JuMP.coefficient(variant, var) ≈ 20.0

            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.coefficient(invariant, var) ≈ 0.0
        end

        @testset "adds cost to expression if present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(
                container, TestCostVariable, MockThermalGen, "gen1", 1,
            )
            add_test_expression!(
                container, TestCostExpression, MockThermalGen, ["gen1"], 1:3,
            )
            param_values = [15.0 15.0 15.0]
            add_test_parameter!(
                container, TestCostParameter, MockThermalGen,
                ["gen1"], 1:3, param_values,
            )

            IOM.add_cost_term_variant!(
                container, var, TestCostParameter,
                TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify expression was updated
            expr_container =
                IOM.get_expression(container, TestCostExpression, MockThermalGen)
            expr = expr_container["gen1", 1]
            @test JuMP.coefficient(expr, var) ≈ 15.0
        end
    end

    @testset "add_cost_term_variant! (rate overload)" begin
        @testset "adds cost to variant objective with explicit rate" begin
            container = make_test_container(1:3)
            var = add_test_variable!(
                container, TestCostVariable, MockThermalGen, "gen1", 1,
            )

            rate = 7.5
            cost = IOM.add_cost_term_variant!(
                container, var, rate,
                TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # cost = var * 7.5
            @test cost == var * rate
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)
            @test JuMP.coefficient(variant, var) ≈ rate
        end

        @testset "adds cost to expression if present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(
                container, TestCostVariable, MockThermalGen, "gen1", 1,
            )
            add_test_expression!(
                container, TestCostExpression, MockThermalGen, ["gen1"], 1:3,
            )

            rate = 12.0
            IOM.add_cost_term_variant!(
                container, var, rate,
                TestCostExpression, MockThermalGen, "gen1", 1,
            )

            expr_container =
                IOM.get_expression(container, TestCostExpression, MockThermalGen)
            expr = expr_container["gen1", 1]
            @test JuMP.coefficient(expr, var) ≈ rate
        end
    end

    @testset "add_cost_to_expression!" begin
        @testset "adds cost to expression when container key exists" begin
            time_steps = 1:3
            container = make_test_container(time_steps)

            # Use FuelConsumptionExpression (accepts IS.InfrastructureSystemsComponent)
            add_test_expression!(
                container,
                IOM.FuelConsumptionExpression,
                MockThermalGen,
                ["gen1"],
                time_steps,
            )

            # Add a cost term at t=2
            cost_value = 42.5
            IOM.add_cost_to_expression!(
                container,
                IOM.FuelConsumptionExpression,
                cost_value,
                MockThermalGen,
                "gen1",
                2,
            )

            expr_container = IOM.get_expression(
                container, IOM.FuelConsumptionExpression, MockThermalGen,
            )
            @test JuMP.constant(expr_container["gen1", 2]) ≈ cost_value
            # Other time steps should be unaffected
            @test JuMP.constant(expr_container["gen1", 1]) ≈ 0.0
            @test JuMP.constant(expr_container["gen1", 3]) ≈ 0.0
        end

        @testset "adds JuMP variable expression to expression container" begin
            time_steps = 1:2
            container = make_test_container(time_steps)
            var = add_test_variable!(
                container, TestCostVariable, MockThermalGen, "gen1", 1,
            )

            add_test_expression!(
                container,
                IOM.FuelConsumptionExpression,
                MockThermalGen,
                ["gen1"],
                time_steps,
            )

            # Add var * 3.0 as cost expression
            cost_expr = 3.0 * var
            IOM.add_cost_to_expression!(
                container,
                IOM.FuelConsumptionExpression,
                cost_expr,
                MockThermalGen,
                "gen1",
                1,
            )

            expr_container = IOM.get_expression(
                container, IOM.FuelConsumptionExpression, MockThermalGen,
            )
            @test JuMP.coefficient(expr_container["gen1", 1], var) ≈ 3.0
        end

        @testset "no-op when container key does not exist" begin
            container = make_test_container(1:2)
            # Don't register FuelConsumptionExpression — should silently return
            IOM.add_cost_to_expression!(
                container,
                IOM.FuelConsumptionExpression,
                10.0,
                MockThermalGen,
                "gen1",
                1,
            )
            @test !IOM.has_container_key(
                container, IOM.FuelConsumptionExpression, MockThermalGen,
            )
        end
    end

    @testset "_add_time_varying_fuel_variable_cost!" begin
        @testset "adds fuel cost to variant objective for each time step" begin
            time_steps = 1:3
            device = make_mock_thermal("gen1"; base_power = 100.0)
            container = make_test_container(time_steps)

            # Create power variable
            power_var_container = IOM.add_variable_container!(
                container, TestCostVariable, MockThermalGen, ["gen1"], time_steps,
            )
            jump_model = IOM.get_jump_model(container)
            for t in time_steps
                power_var_container["gen1", t] = JuMP.@variable(
                    jump_model, base_name = "P_gen1_$(t)",
                )
            end

            # Pre-populate FuelConsumptionExpression:
            # In production, POM's thermalgeneration_constructor.jl populates this
            # via add_expressions!(container, FuelConsumptionExpression, ...).
            # We set it to proportional_term * power_var to simulate that.
            fuel_expr_container = IOM.add_expression_container!(
                container,
                IOM.FuelConsumptionExpression,
                MockThermalGen,
                ["gen1"],
                time_steps,
            )
            proportional_term = 6.0  # MMBTU/p.u.h
            for t in time_steps
                JuMP.add_to_expression!(
                    fuel_expr_container["gen1", t],
                    proportional_term,
                    power_var_container["gen1", t],
                )
            end

            # Set up FuelCostParameter with time-varying fuel prices
            fuel_prices = [4.0, 8.0, 2.0]
            add_test_parameter!(
                container,
                IOM.FuelCostParameter,
                MockThermalGen,
                ["gen1"],
                time_steps,
                reshape(fuel_prices, 1, :),
            )

            # Dispatch requires an IS.TimeSeriesKey argument (unused in the body)
            ts_key = IS.StaticTimeSeriesKey(
                IS.SingleTimeSeries,
                "fuel_cost",
                Dates.DateTime(2024, 1, 1),
                Dates.Hour(1),
                3,
                Dict{String, Any}(),
            )

            IOM._add_time_varying_fuel_variable_cost!(
                container,
                TestCostVariable,
                device,
                ts_key,
            )

            # Variant objective should contain:
            #   fuel_consumption[name, t] * fuel_price[t]
            # = proportional_term * power_var * fuel_price
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)
            for t in time_steps
                var = power_var_container["gen1", t]
                expected = proportional_term * fuel_prices[t]
                @test JuMP.coefficient(variant, var) ≈ expected atol = 1e-10
            end
        end
    end

    @testset "PWL Helpers" begin
        @testset "add_pwl_variables_delta! creates bounded variables" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables_delta!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 4,
            )

            @test length(pwl_vars) == 4
            for (i, var) in enumerate(pwl_vars)
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 1.0
                # Check name contains key elements (type names may be fully qualified)
                # FIXME variable names depend on from where the function is called. ick.
                var_name = JuMP.name(var)
                @test occursin("TestPWLVariable", var_name)
                @test occursin("gen1", var_name)
                @test occursin("pwl_$(i)", var_name)
            end

            # Verify stored in container
            var_container = IOM.get_variable(container, TestPWLVariable, MockThermalGen)
            for i in 1:4
                @test var_container["gen1", i, 1] === pwl_vars[i]
            end
        end

        @testset "add_pwl_variables_delta! with custom upper bound" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables_delta!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 3;
                upper_bound = 100.0,
            )

            @test length(pwl_vars) == 3
            for var in pwl_vars
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 100.0
            end
        end

        @testset "add_pwl_variables_delta! with no upper bound (Inf)" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables_delta!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 3;
                upper_bound = Inf,
            )

            @test length(pwl_vars) == 3
            for var in pwl_vars
                @test JuMP.lower_bound(var) == 0.0
                @test !JuMP.has_upper_bound(var)
            end
        end

        @testset "add_pwl_linking_constraint! creates correct constraint" begin
            container = make_test_container(1:3)

            # Create power variable
            power_var =
                add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            # Create PWL delta variables manually
            jump_model = IOM.get_jump_model(container)
            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            breakpoints = [0.0, 50.0, 100.0]

            IOM.add_pwl_linking_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                power_var, pwl_vars, breakpoints,
            )

            # Get the constraint
            con_container =
                IOM.get_constraint(container, TestCostConstraint, MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint: power_var == sum(pwl_vars .* breakpoints)
            # In normalized form: power_var - 0*δ1 - 50*δ2 - 100*δ3 == 0
            con_func = JuMP.constraint_object(con).func
            @test JuMP.coefficient(con_func, power_var) ≈ 1.0
            @test JuMP.coefficient(con_func, pwl_vars[1]) ≈ -0.0
            @test JuMP.coefficient(con_func, pwl_vars[2]) ≈ -50.0
            @test JuMP.coefficient(con_func, pwl_vars[3]) ≈ -100.0
        end

        @testset "add_pwl_normalization_constraint! creates correct constraint" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            on_status = 1.0

            IOM.add_pwl_normalization_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                pwl_vars, on_status,
            )

            con_container =
                IOM.get_constraint(container, TestCostConstraint, MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint: sum(pwl_vars) == on_status
            # In normalized form: δ1 + δ2 + δ3 == 1
            con_func = JuMP.constraint_object(con).func
            for var in pwl_vars
                @test JuMP.coefficient(con_func, var) ≈ 1.0
            end
            @test JuMP.normalized_rhs(con) ≈ 1.0
        end

        @testset "add_pwl_normalization_constraint! with variable on_status" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            on_var = JuMP.@variable(jump_model, base_name = "on_status", binary = true)

            IOM.add_pwl_normalization_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                pwl_vars, on_var,
            )

            con_container =
                IOM.get_constraint(container, TestCostConstraint, MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint includes the on_status variable
            con_func = JuMP.constraint_object(con).func
            @test JuMP.coefficient(con_func, on_var) ≈ -1.0
        end

        @testset "add_pwl_sos2_constraint! creates SOS2 constraint" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:4]

            IOM.add_pwl_sos2_constraint!(
                container, MockThermalGen, "gen1", 1, pwl_vars,
            )

            # Verify SOS2 constraint was added to the model
            # JuMP stores SOS constraints separately
            @test JuMP.num_constraints(
                jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            ) == 1
        end

        @testset "get_pwl_cost_expression_delta computes correct expression" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            slopes = [10.0, 20.0, 30.0]
            multiplier = 2.0

            cost_expr = IOM.get_pwl_cost_expression_delta(pwl_vars, slopes, multiplier)

            # Verify: cost = Σ δ[i] * slope[i] * multiplier
            # = δ1 * 10 * 2 + δ2 * 20 * 2 + δ3 * 30 * 2
            # = 20*δ1 + 40*δ2 + 60*δ3
            @test JuMP.coefficient(cost_expr, pwl_vars[1]) ≈ 20.0
            @test JuMP.coefficient(cost_expr, pwl_vars[2]) ≈ 40.0
            @test JuMP.coefficient(cost_expr, pwl_vars[3]) ≈ 60.0
            @test JuMP.constant(cost_expr) ≈ 0.0
        end

        @testset "get_pwl_cost_expression_delta with multiplier = 1" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:2]
            slopes = [5.0, 15.0]

            cost_expr = IOM.get_pwl_cost_expression_delta(pwl_vars, slopes, 1.0)

            @test JuMP.coefficient(cost_expr, pwl_vars[1]) ≈ 5.0
            @test JuMP.coefficient(cost_expr, pwl_vars[2]) ≈ 15.0
        end
    end

    # Helpers for the ProductionCostExpression propagation testset
    function _setup_prop_test_container(time_steps, names = ["gen1"])
        container = make_test_container(time_steps)
        var_container = IOM.add_variable_container!(
            container, TestCostVariable, MockThermalGen, names, time_steps,
        )
        jump_model = IOM.get_jump_model(container)
        for name in names, t in time_steps
            var_container[name, t] =
                JuMP.@variable(jump_model, base_name = "v_$(name)_$(t)")
        end
        return container, var_container
    end

    _expr_coef(container, ::Type{E}, var, name, t) where {E} = JuMP.coefficient(
        IOM.get_expression(container, E, MockThermalGen)[name, t], var,
    )
    _inv_coef(container, var) = JuMP.coefficient(
        IOM.get_invariant_terms(IOM.get_objective_expression(container)), var,
    )
    _var_coef(container, var) = JuMP.coefficient(
        IOM.get_variant_terms(IOM.get_objective_expression(container)), var,
    )

    @testset "ProductionCostExpression propagation" begin
        time_steps = 1:1

        @testset "add_cost_term_to_expression! propagates constituent → ProductionCost" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            r = 11.0
            IOM.add_cost_term_to_expression!(
                container, vars["gen1", 1], r,
                IOM.FuelCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _expr_coef(
                container, IOM.ProductionCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            # No objective hook on this entry point.
            @test _inv_coef(container, vars["gen1", 1]) ≈ 0.0
            @test _var_coef(container, vars["gen1", 1]) ≈ 0.0
        end

        @testset "add_cost_term_invariant! → constituent, ProductionCost, and invariant obj" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            r = 13.0
            IOM.add_cost_term_invariant!(
                container, vars["gen1", 1], r,
                IOM.FuelCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _expr_coef(
                container, IOM.ProductionCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _inv_coef(container, vars["gen1", 1]) ≈ r
            @test _var_coef(container, vars["gen1", 1]) ≈ 0.0
        end

        @testset "add_cost_term_invariant! direct ProductionCost write does not recurse" begin
            container, vars = _setup_prop_test_container(time_steps)
            add_test_expression!(
                container, IOM.ProductionCostExpression, MockThermalGen,
                ["gen1"], time_steps,
            )
            r = 17.0
            IOM.add_cost_term_invariant!(
                container, vars["gen1", 1], r,
                IOM.ProductionCostExpression, MockThermalGen, "gen1", 1,
            )
            # ProductionCostExpression ⊀ ConstituentCostExpression, so the propagation
            # hook is a no-op — exactly one write to ProductionCostExpression.
            @test _expr_coef(
                container, IOM.ProductionCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _inv_coef(container, vars["gen1", 1]) ≈ r
        end

        @testset "add_cost_term_invariant! to non-constituent reaches obj, not ProductionCost" begin
            container, vars = _setup_prop_test_container(time_steps)
            add_test_expression!(
                container, TestCostExpression, MockThermalGen, ["gen1"], time_steps,
            )
            add_test_expression!(
                container, IOM.ProductionCostExpression, MockThermalGen,
                ["gen1"], time_steps,
            )
            r = 19.0
            IOM.add_cost_term_invariant!(
                container, vars["gen1", 1], r,
                TestCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, TestCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            # TestCostExpression is not a ConstituentCostExpression — ProductionCost untouched.
            @test _expr_coef(
                container, IOM.ProductionCostExpression,
                vars["gen1", 1], "gen1", 1) ≈ 0.0
            @test _inv_coef(container, vars["gen1", 1]) ≈ r
        end

        @testset "add_cost_term_variant! (param-rate) propagates and hits variant obj" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            param_rate = 23.0
            add_test_parameter!(
                container, TestCostParameter, MockThermalGen,
                ["gen1"], time_steps, fill(param_rate, 1, length(time_steps)),
            )
            IOM.add_cost_term_variant!(
                container, vars["gen1", 1], TestCostParameter,
                IOM.FuelCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ param_rate
            @test _expr_coef(
                container, IOM.ProductionCostExpression,
                vars["gen1", 1], "gen1", 1) ≈ param_rate
            @test _var_coef(container, vars["gen1", 1]) ≈ param_rate
            @test _inv_coef(container, vars["gen1", 1]) ≈ 0.0
        end

        @testset "add_cost_term_variant! (explicit-rate) propagates and hits variant obj" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            r = 29.0
            IOM.add_cost_term_variant!(
                container, vars["gen1", 1], r,
                IOM.FuelCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _expr_coef(
                container, IOM.ProductionCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _var_coef(container, vars["gen1", 1]) ≈ r
            @test _inv_coef(container, vars["gen1", 1]) ≈ 0.0
        end

        @testset "add_cost_to_expression! propagates constituent → ProductionCost" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            r = 31.0
            IOM.add_cost_to_expression!(
                container, IOM.FuelCostExpression, r * vars["gen1", 1],
                MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test _expr_coef(
                container, IOM.ProductionCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            # No objective hook on this entry point.
            @test _inv_coef(container, vars["gen1", 1]) ≈ 0.0
            @test _var_coef(container, vars["gen1", 1]) ≈ 0.0
        end

        @testset "add_proportional_cost_invariant! propagates per-time-step" begin
            ts = 1:3
            container, vars = _setup_prop_test_container(ts)
            for E in (IOM.FuelCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], ts)
            end
            device = make_mock_thermal("gen1"; base_power = 100.0)
            cost_term = 7.0
            multiplier = 1.0
            # SYSTEM_BASE → no normalization; dt = 1 hour → rate = cost_term * multiplier.
            IOM.add_proportional_cost_invariant!(
                container, TestCostVariable, device, cost_term,
                IS.UnitSystem.SYSTEM_BASE, multiplier, IOM.FuelCostExpression,
            )
            expected = cost_term * multiplier
            for t in ts
                @test _expr_coef(
                    container, IOM.FuelCostExpression,
                    vars["gen1", t], "gen1", t) ≈ expected
                @test _expr_coef(
                    container, IOM.ProductionCostExpression,
                    vars["gen1", t], "gen1", t) ≈ expected
                @test _inv_coef(container, vars["gen1", t]) ≈ expected
            end
        end

        @testset "multiple constituents sum into ProductionCost without double-count" begin
            container, vars = _setup_prop_test_container(time_steps)
            for E in (IOM.FuelCostExpression, IOM.VOMCostExpression,
                IOM.StartUpCostExpression, IOM.ProductionCostExpression)
                add_test_expression!(container, E, MockThermalGen, ["gen1"], time_steps)
            end
            r_fuel, r_vom, r_su = 3.0, 5.0, 7.0
            for (E, r) in ((IOM.FuelCostExpression, r_fuel),
                (IOM.VOMCostExpression, r_vom),
                (IOM.StartUpCostExpression, r_su))
                IOM.add_cost_term_invariant!(
                    container, vars["gen1", 1], r, E, MockThermalGen, "gen1", 1,
                )
            end
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r_fuel
            @test _expr_coef(
                container, IOM.VOMCostExpression, vars["gen1", 1], "gen1", 1) ≈ r_vom
            @test _expr_coef(
                container, IOM.StartUpCostExpression, vars["gen1", 1], "gen1", 1) ≈ r_su
            # Each constituent contributes once; no double-count, no missing terms.
            @test _expr_coef(
                container, IOM.ProductionCostExpression,
                vars["gen1", 1], "gen1", 1) ≈ r_fuel + r_vom + r_su
            @test _inv_coef(container, vars["gen1", 1]) ≈ r_fuel + r_vom + r_su
        end

        @testset "ProductionCost not registered: constituent write is a no-op for prod" begin
            container, vars = _setup_prop_test_container(time_steps)
            # Only the constituent container is registered.
            add_test_expression!(
                container, IOM.FuelCostExpression, MockThermalGen, ["gen1"], time_steps,
            )
            r = 41.0
            IOM.add_cost_term_invariant!(
                container, vars["gen1", 1], r,
                IOM.FuelCostExpression, MockThermalGen, "gen1", 1,
            )
            @test _expr_coef(
                container, IOM.FuelCostExpression, vars["gen1", 1], "gen1", 1) ≈ r
            @test !IOM.has_container_key(
                container, IOM.ProductionCostExpression, MockThermalGen)
            @test _inv_coef(container, vars["gen1", 1]) ≈ r
        end
    end
end
