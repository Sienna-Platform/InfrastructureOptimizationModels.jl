"""
Unit tests for OptimizationContainer using mock components.
Tests container machinery without requiring real PowerSystems data or solvers.
"""

using InfrastructureSystems

const ISOPT = InfrastructureSystems.Optimization

# Mock constraint/expression types for testing container machinery
struct MockConstraintType <: ISOPT.ConstraintType end
struct MockExpressionType <: ISOPT.ExpressionType end

@testset "OptimizationContainer with MockSystem" begin
    @testset "Container creation" begin
        # Create mock system
        mock_sys = MockSystem(100.0)

        # Create settings with mock system
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )

        # Create container - uses duck-typed system and mock time series type
        container = IOM.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )

        @test IOM.get_model_base_power(container) == 100.0
        @test isempty(IOM.get_variables(container))
        @test isempty(IOM.get_constraints(container))
        @test isempty(IOM.get_expressions(container))
    end

    @testset "add_variable_container!" begin
        mock_sys = MockSystem(100.0)
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = IOM.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )

        # Set time steps (normally done by init_optimization_container!)
        IOM.set_time_steps!(container, 1:24)

        # Add a variable container using mock component type as the key
        # (the container just needs a type - doesn't need actual component instances)
        device_names = ["gen1", "gen2", "gen3"]
        time_steps = IOM.get_time_steps(container)

        var_container = IOM.add_variable_container!(
            container,
            IOM.ActivePowerVariable,
            MockComponentType,
            device_names,
            time_steps,
        )

        # Verify the container was created
        @test !isempty(IOM.get_variables(container))

        # Verify we can retrieve it
        var_key = IOM.VariableKey(IOM.ActivePowerVariable, MockComponentType)
        @test haskey(IOM.get_variables(container), var_key)

        # Verify dimensions
        retrieved =
            IOM.get_variable(container, IOM.ActivePowerVariable, MockComponentType)
        @test size(retrieved) == (length(device_names), length(time_steps))
    end

    @testset "add_constraints_container!" begin
        mock_sys = MockSystem(100.0)
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = IOM.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )
        IOM.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = IOM.get_time_steps(container)

        cons_container = IOM.add_constraints_container!(
            container,
            MockConstraintType,
            MockComponentType,
            device_names,
            time_steps,
        )

        @test !isempty(IOM.get_constraints(container))

        cons_key = IOM.ConstraintKey(MockConstraintType, MockComponentType)
        @test haskey(IOM.get_constraints(container), cons_key)
    end

    @testset "add_expression_container!" begin
        mock_sys = MockSystem(100.0)
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = IOM.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )
        IOM.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = IOM.get_time_steps(container)

        expr_container = IOM.add_expression_container!(
            container,
            MockExpressionType,
            MockComponentType,
            device_names,
            time_steps,
        )

        @test !isempty(IOM.get_expressions(container))

        expr_key = IOM.ExpressionKey(MockExpressionType, MockComponentType)
        @test haskey(IOM.get_expressions(container), expr_key)
    end

    @testset "Parameter multiplier applied exactly once" begin
        # Regression for the double-multiply bug: the generic get_parameter_values
        # previously multiplied by the multiplier, and calculate_parameter_values
        # multiplied again, yielding param .* mult.^2. With the common -1.0 RHS
        # multiplier this silently flipped the sign of written parameter outputs.
        param_array = DenseAxisArray([5.0], ["dev1"])
        multiplier_array = DenseAxisArray([-1.0], ["dev1"])
        container = IOM.ParameterContainer(param_array, multiplier_array)

        # get_parameter_values returns raw values (no multiplier applied)
        @test IOM.get_parameter_values(container)["dev1"] == 5.0
        # calculate_parameter_values applies the multiplier exactly once
        @test IOM.calculate_parameter_values(container)["dev1"] == -5.0
    end

    @testset "process_duals preserves general-integer variables (Task 1.8)" begin
        # Regression for the dual-processing restore loop, which previously
        # re-declared every integer variable as binary and crashed on fixed
        # integers (UndefVarError on the closed-loop variable `v`).
        mock_sys = MockSystem(100.0)
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(1),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        jump_model = JuMP.Model(HiGHS_optimizer)
        container = IOM.OptimizationContainer(
            mock_sys,
            settings,
            jump_model,
            MockDeterministic,
        )
        IOM.set_time_steps!(container, 1:1)
        model = IOM.get_jump_model(container)

        names = ["g1"]
        var = IOM.add_variable_container!(
            container,
            TestVariableType,
            MockComponentType,
            names,
            1:1,
        )
        v = JuMP.@variable(model, integer = true, lower_bound = 0.0, upper_bound = 5.0)
        var["g1", 1] = v

        cons = IOM.add_constraints_container!(
            container,
            TestConstraintType,
            MockComponentType,
            names,
            1:1,
        )
        cons["g1", 1] = JuMP.@constraint(model, v <= 3.0)

        IOM.add_dual_container!(
            container,
            TestConstraintType,
            MockComponentType,
            names,
            1:1,
        )

        JuMP.@objective(model, Max, v)
        JuMP.optimize!(model)
        @test JuMP.value(v) ≈ 3.0

        status = IOM.process_duals(container, HiGHS_optimizer)
        @test status == IOM.RunStatus.SUCCESSFULLY_FINALIZED

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
        ic_key = IOM.InitialConditionKey(MockInitialCondition, MockComponentType)
        component = MockComponentType()
        ic = IOM.InitialCondition(ic_key, component, 1.0)
        @test IOM.get_value(ic) == 1.0
        @test IOM.get_ic_type(ic) === MockInitialCondition
    end

    @testset "get_objective_expression is idempotent (Task 2.17)" begin
        model = JuMP.Model()
        JuMP.@variable(model, x)
        obj = IOM.ObjectiveFunction()
        JuMP.add_to_expression!(IOM.get_invariant_terms(obj), 2.0, x)
        JuMP.add_to_expression!(IOM.get_variant_terms(obj), 3.0, x)
        expr1 = IOM.get_objective_expression(obj)
        expr2 = IOM.get_objective_expression(obj)
        # Calling twice must not double-count the invariant terms.
        @test expr1 == expr2
        @test JuMP.coefficient(expr2, x) == 5.0
    end
end
