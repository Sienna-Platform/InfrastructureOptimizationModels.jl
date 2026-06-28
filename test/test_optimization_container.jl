"""
Unit tests for OptimizationContainer using mock components.
Tests container machinery without requiring real PowerSystems data or solvers.
"""

using InfrastructureSystems

# Define aliases if not already defined by test harness
if !@isdefined(PSI)
    const PSI = InfrastructureOptimizationModels
end
const ISOPT = InfrastructureSystems.Optimization

# Mock constraint/expression types for testing container machinery
struct MockConstraintType <: ISOPT.ConstraintType end
struct MockExpressionType <: ISOPT.ExpressionType end

@testset "OptimizationContainer with MockSystem" begin
    @testset "Container creation" begin
        # Create mock system
        mock_sys = MockSystem(100.0)

        # Create settings with mock system
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )

        # Create container - uses duck-typed system and mock time series type
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )

        @test PSI.get_model_base_power(container) == 100.0
        @test isempty(PSI.get_variables(container))
        @test isempty(PSI.get_constraints(container))
        @test isempty(PSI.get_expressions(container))
    end

    @testset "add_variable_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )

        # Set time steps (normally done by init_optimization_container!)
        PSI.set_time_steps!(container, 1:24)

        # Add a variable container using mock component type as the key
        # (the container just needs a type - doesn't need actual component instances)
        device_names = ["gen1", "gen2", "gen3"]
        time_steps = PSI.get_time_steps(container)

        var_container = PSI.add_variable_container!(
            container,
            PSI.ActivePowerVariable,
            MockComponentType,
            device_names,
            time_steps,
        )

        # Verify the container was created
        @test !isempty(PSI.get_variables(container))

        # Verify we can retrieve it
        var_key = PSI.VariableKey(PSI.ActivePowerVariable, MockComponentType)
        @test haskey(PSI.get_variables(container), var_key)

        # Verify dimensions
        retrieved =
            PSI.get_variable(container, PSI.ActivePowerVariable, MockComponentType)
        @test size(retrieved) == (length(device_names), length(time_steps))
    end

    @testset "add_constraints_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = PSI.get_time_steps(container)

        cons_container = PSI.add_constraints_container!(
            container,
            MockConstraintType,
            MockComponentType,
            device_names,
            time_steps,
        )

        @test !isempty(PSI.get_constraints(container))

        cons_key = PSI.ConstraintKey(MockConstraintType, MockComponentType)
        @test haskey(PSI.get_constraints(container), cons_key)
    end

    @testset "add_expression_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = PSI.get_time_steps(container)

        expr_container = PSI.add_expression_container!(
            container,
            MockExpressionType,
            MockComponentType,
            device_names,
            time_steps,
        )

        @test !isempty(PSI.get_expressions(container))

        expr_key = PSI.ExpressionKey(MockExpressionType, MockComponentType)
        @test haskey(PSI.get_expressions(container), expr_key)
    end

    @testset "add_variable_container! - prebuilt SparseAxisArray" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(1),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            JuMP.Model(),
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:1)
        model = PSI.get_jump_model(container)

        # Genuinely irregular per-outage keys (not a cartesian product), built
        # incrementally the way the post-contingency models do.
        idx_keys = [("outage1", "b1", 1), ("outage1", "b2", 1), ("outage2", "b1", 1)]
        sa = SparseAxisArray(Dict(k => JuMP.@variable(model) for k in idx_keys))

        result = PSI.add_variable_container!(
            container, TestVariableType, MockComponentType, sa; meta = "svc",
        )

        @test result === sa
        var_key = PSI.VariableKey(TestVariableType, MockComponentType, "svc")
        @test haskey(PSI.get_variables(container), var_key)
        @test PSI.get_variables(container)[var_key] === sa
    end

    @testset "add_constraints_container! - prebuilt SparseAxisArray" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(1),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            JuMP.Model(),
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:1)
        model = PSI.get_jump_model(container)
        x = JuMP.@variable(model)

        idx_keys = [("outage1", "b1", 1), ("outage2", "b1", 1)]
        sa = SparseAxisArray(Dict(k => JuMP.@constraint(model, x <= 1.0) for k in idx_keys))

        result = PSI.add_constraints_container!(
            container, MockConstraintType, MockComponentType, sa; meta = "lb",
        )

        @test result === sa
        cons_key = PSI.ConstraintKey(MockConstraintType, MockComponentType, "lb")
        @test haskey(PSI.get_constraints(container), cons_key)
        @test PSI.get_constraints(container)[cons_key] === sa
    end

    @testset "Parameter multiplier applied exactly once" begin
        # Regression for the double-multiply bug: the generic get_parameter_values
        # previously multiplied by the multiplier, and calculate_parameter_values
        # multiplied again, yielding param .* mult.^2. With the common -1.0 RHS
        # multiplier this silently flipped the sign of written parameter outputs.
        param_array = DenseAxisArray([5.0], ["dev1"])
        multiplier_array = DenseAxisArray([-1.0], ["dev1"])
        container = PSI.ParameterContainer(param_array, multiplier_array)

        # get_parameter_values returns raw values (no multiplier applied)
        @test PSI.get_parameter_values(container)["dev1"] == 5.0
        # calculate_parameter_values applies the multiplier exactly once
        @test PSI.calculate_parameter_values(container)["dev1"] == -5.0
    end

    @testset "process_duals preserves general-integer variables (Task 1.8)" begin
        # Regression for the dual-processing restore loop, which previously
        # re-declared every integer variable as binary and crashed on fixed
        # integers (UndefVarError on the closed-loop variable `v`).
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(1),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        jump_model = JuMP.Model(HiGHS_optimizer)
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            jump_model,
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:1)
        model = PSI.get_jump_model(container)

        names = ["g1"]
        var = PSI.add_variable_container!(
            container,
            TestVariableType,
            MockComponentType,
            names,
            1:1,
        )
        v = JuMP.@variable(model, integer = true, lower_bound = 0.0, upper_bound = 5.0)
        var["g1", 1] = v

        cons = PSI.add_constraints_container!(
            container,
            TestConstraintType,
            MockComponentType,
            names,
            1:1,
        )
        cons["g1", 1] = JuMP.@constraint(model, v <= 3.0)

        PSI.add_dual_container!(
            container,
            TestConstraintType,
            MockComponentType,
            names,
            1:1,
        )

        JuMP.@objective(model, Max, v)
        JuMP.optimize!(model)
        @test JuMP.value(v) ≈ 3.0

        status = PSI.process_duals(container, HiGHS_optimizer)
        @test status == PSI.RunStatus.SUCCESSFULLY_FINALIZED

        # The general-integer variable must remain integer (not silently binary),
        # be unfixed, and have its bounds restored.
        @test JuMP.is_integer(v)
        @test !JuMP.is_binary(v)
        @test !JuMP.is_fixed(v)
        @test JuMP.has_lower_bound(v) && JuMP.lower_bound(v) == 0.0
        @test JuMP.has_upper_bound(v) && JuMP.upper_bound(v) == 5.0
    end

    @testset "Key-based InitialCondition constructor (Task 2.4)" begin
        # Previously instantiated InitialCondition{T, U} with U the component type,
        # violating the value-type bound → TypeError on every call.
        ic_key = PSI.InitialConditionKey(MockInitialCondition, MockComponentType)
        component = MockComponentType()
        ic = PSI.InitialCondition(ic_key, component, 1.0)
        @test PSI.get_value(ic) == 1.0
        @test PSI.get_ic_type(ic) === MockInitialCondition
    end

    @testset "get_objective_expression is idempotent (Task 2.17)" begin
        model = JuMP.Model()
        JuMP.@variable(model, x)
        obj = PSI.ObjectiveFunction()
        JuMP.add_to_expression!(PSI.get_invariant_terms(obj), 2.0, x)
        JuMP.add_to_expression!(PSI.get_variant_terms(obj), 3.0, x)
        expr1 = PSI.get_objective_expression(obj)
        expr2 = PSI.get_objective_expression(obj)
        # Calling twice must not double-count the invariant terms.
        @test expr1 == expr2
        @test JuMP.coefficient(expr2, x) == 5.0
    end
end
