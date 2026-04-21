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

    @testset "_assign_container! throws on duplicate key" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )
        PSI.set_time_steps!(container, 1:24)

        # First assignment succeeds
        PSI.add_variable_container!(
            container,
            PSI.ActivePowerVariable,
            MockComponentType,
            ["gen1"],
            1:24,
        )

        # Second assignment with same key should throw.
        # Suppress the @error log to avoid tripping the framework's zero-error assertion.
        @test_throws IS.InvalidValue Logging.with_logger(Logging.NullLogger()) do
            PSI.add_variable_container!(
                container,
                PSI.ActivePowerVariable,
                MockComponentType,
                ["gen1"],
                1:24,
            )
        end
    end

    @testset "_get_entry throws on missing key" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            MockDeterministic,
        )

        # Looking up a variable that was never added should throw
        @test_throws IS.InvalidValue PSI.get_variable(
            container,
            PSI.ActivePowerVariable,
            MockComponentType,
        )
    end
end
