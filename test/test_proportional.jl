"""
Unit tests for proportional cost objective function construction.
Tests the functions in src/objective_function/proportional.jl using mock components.
"""

# Test-specific variable type
struct TestProportionalVariable <: InfrastructureOptimizationModels.VariableType end

# Test-specific formulation
struct TestProportionalFormulation <:
       InfrastructureOptimizationModels.AbstractDeviceFormulation end

# Stub: objective_function_multiplier returns 1.0 for test types
InfrastructureOptimizationModels.objective_function_multiplier(
    ::Type{TestProportionalVariable},
    ::Type{TestProportionalFormulation},
) = 1.0

# Interface implementations for mock types

# Non-time-varying proportional_cost: return the proportional_term from MockOperationCost
InfrastructureOptimizationModels.proportional_cost(
    op_cost::MockOperationCost,
    ::Type{TestProportionalVariable},
    d::MockThermalGen,
    ::Type{TestProportionalFormulation},
) = op_cost.proportional_term

# Time-varying proportional_cost: same value for all time steps (could vary if needed)
InfrastructureOptimizationModels.proportional_cost(
    ::InfrastructureOptimizationModels.OptimizationContainer,
    op_cost::MockOperationCost,
    ::Type{TestProportionalVariable},
    d::MockThermalGen,
    ::Type{TestProportionalFormulation},
    ::Int,
) = op_cost.proportional_term

# is_time_variant_proportional: return the is_time_variant flag from MockOperationCost
InfrastructureOptimizationModels.is_time_variant_proportional(op_cost::MockOperationCost) =
    op_cost.is_time_variant

# Helper to set up container with variables for devices
function setup_proportional_test_container(
    time_steps::UnitRange{Int},
    devices::Vector{MockThermalGen};
    resolution = Dates.Hour(1),
)
    sys = MockSystem(100.0)
    settings = InfrastructureOptimizationModels.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = InfrastructureOptimizationModels.OptimizationContainer(
        sys,
        settings,
        JuMP.Model(),
        MockDeterministic,
    )
    InfrastructureOptimizationModels.set_time_steps!(container, time_steps)

    # Add variable container for all devices
    device_names = [get_name(d) for d in devices]
    var_container = InfrastructureOptimizationModels.add_variable_container!(
        container,
        TestProportionalVariable,
        MockThermalGen,
        device_names,
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = InfrastructureOptimizationModels.get_jump_model(container)
    for name in device_names, t in time_steps
        var_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "TestProportional_$(name)_$(t)",
        )
    end

    return container
end

# Create a FlattenIteratorWrapper from a vector of devices
function make_device_iterator(devices::Vector{MockThermalGen})
    return IS.FlattenIteratorWrapper(MockThermalGen, Vector[devices])
end

@testset "Proportional Cost Objective Functions" begin
    @testset "add_proportional_cost! adds to invariant objective" begin
        time_steps = 1:3
        cost_value = 15.0
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(cost_value, false),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Verify costs are in invariant expression (not variant)
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            cost_value;
            variant = false,
        )

        # Verify variant expression has no terms
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            0.0;
            variant = true,
        )
    end

    @testset "add_proportional_cost! with zero cost skips device" begin
        time_steps = 1:2
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(0.0, false),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Both invariant and variant should have zero coefficients
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            0.0;
            variant = false,
        )
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_proportional_cost! with multiple devices" begin
        time_steps = 1:2
        cost1 = 10.0
        cost2 = 20.0
        device1 = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(cost1, false),
        )
        device2 = make_mock_thermal(
            "gen2";
            operation_cost = MockOperationCost(cost2, false),
        )
        devices = [device1, device2]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Verify each device has correct coefficients
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            cost1;
            variant = false,
        )
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen2",
            cost2;
            variant = false,
        )
    end

    @testset "add_proportional_cost_maybe_time_variant! - not time variant" begin
        time_steps = 1:3
        cost_value = 25.0
        # is_time_variant = false
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(cost_value, false),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost_maybe_time_variant!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Costs should be in invariant expression
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            cost_value;
            variant = false,
        )

        # Variant expression should have zero
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            0.0;
            variant = true,
        )
    end

    @testset "add_proportional_cost_maybe_time_variant! - time variant" begin
        time_steps = 1:3
        cost_value = 30.0
        # is_time_variant = true
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(cost_value, true),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost_maybe_time_variant!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Costs should be in variant expression (not invariant)
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            cost_value;
            variant = true,
        )

        # Invariant expression should have zero
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen1",
            0.0;
            variant = false,
        )
    end

    @testset "add_proportional_cost_maybe_time_variant! - mixed devices" begin
        time_steps = 1:2
        cost_invariant = 10.0
        cost_variant = 20.0

        device_invariant = make_mock_thermal(
            "gen_inv";
            operation_cost = MockOperationCost(cost_invariant, false),
        )
        device_variant = make_mock_thermal(
            "gen_var";
            operation_cost = MockOperationCost(cost_variant, true),
        )
        devices = [device_invariant, device_variant]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost_maybe_time_variant!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # device_invariant costs in invariant expression
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen_inv",
            cost_invariant;
            variant = false,
        )
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen_inv",
            0.0;
            variant = true,
        )

        # device_variant costs in variant expression
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen_var",
            cost_variant;
            variant = true,
        )
        @test verify_objective_coefficients(
            container,
            TestProportionalVariable,
            MockThermalGen,
            "gen_var",
            0.0;
            variant = false,
        )
    end

    @testset "add_proportional_cost_maybe_time_variant! with zero cost skips" begin
        time_steps = 1:2
        # Zero cost, even if marked as time variant
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(0.0, true),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost_maybe_time_variant!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Both should be zero - device was skipped
        @test count_objective_terms(container; variant = false) == 0
        @test count_objective_terms(container; variant = true) == 0
    end

    @testset "add_proportional_cost_maybe_time_variant! with skip_proportional_cost" begin
        # Override skip_proportional_cost for MockThermalGen
        IOM.skip_proportional_cost(::MockThermalGen) = true

        time_steps = 1:2
        cost_value = 15.0
        device = make_mock_thermal(
            "gen1";
            operation_cost = MockOperationCost(cost_value, true),
        )
        devices = [device]
        container = setup_proportional_test_container(time_steps, devices)

        # Add ProductionCostExpression so add_cost_to_expression! has somewhere to write
        device_names = [get_name(d) for d in devices]
        add_test_expression!(
            container,
            IOM.ProductionCostExpression,
            MockThermalGen,
            device_names,
            time_steps,
        )

        devices_iter = make_device_iterator(devices)

        InfrastructureOptimizationModels.add_proportional_cost_maybe_time_variant!(
            container,
            TestProportionalVariable,
            devices_iter,
            TestProportionalFormulation,
        )

        # Cost should NOT be in the objective (skip_proportional_cost = true)
        @test count_objective_terms(container; variant = false) == 0
        @test count_objective_terms(container; variant = true) == 0

        # But should be in the ProductionCostExpression
        expr = IOM.get_expression(
            container, IOM.ProductionCostExpression, MockThermalGen)
        for t in time_steps
            @test JuMP.constant(expr["gen1", t]) ≈ cost_value
        end

        # Reset to default
        IOM.skip_proportional_cost(::MockThermalGen) = false
    end
end
