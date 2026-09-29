"""
Unit tests for linear curve objective function construction.
Tests the functions in src/objective_function/linear_curve.jl using mock components.
"""

# Test-specific variable type
struct TestActivePowerVariable <: InfrastructureOptimizationModels.VariableType end

# Test-specific formulation
struct TestLinearFormulation <: InfrastructureOptimizationModels.AbstractDeviceFormulation end

# Stub: objective_function_multiplier returns 1.0 for test types
InfrastructureOptimizationModels.objective_function_multiplier(
    ::Type{TestActivePowerVariable},
    ::Type{TestLinearFormulation},
) = 1.0

# Helper to set up container with variables for a device
function setup_container_with_variables(
    time_steps::UnitRange{Int},
    device::MockThermalGen;
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

    # Add variable container for the device
    device_name = get_name(device)
    var_container = InfrastructureOptimizationModels.add_variable_container!(
        container,
        TestActivePowerVariable,
        MockThermalGen,
        [device_name],
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = InfrastructureOptimizationModels.get_jump_model(container)
    for t in time_steps
        var_container[device_name, t] = JuMP.@variable(
            jump_model,
            base_name = "TestActivePower_$(device_name)_$(t)",
        )
    end

    return container
end

@testset "Linear Curve Objective Functions" begin
    @testset "add_proportional_cost_invariant! normalizes to system base" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 50.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        # Cost: 25.0 $/MWh in natural units, system base = 100 MW
        # Normalized: 25.0 * 100.0 = 2500.0 $/p.u.h
        # With dt = 1.0 and multiplier = 1.0: coefficient = 2500.0
        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            25.0,
            1.0,
            IOM.ProductionCostExpression,
        )

        expected_coef = 25.0 * 100.0 * 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "add_proportional_cost_invariant! with multiplier and sub-hourly resolution" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Minute(15),
            )

        # Cost: 0.2 $/MWh, i.e. 20.0 $/p.u.h at 100 MW system base; multiplier = 2.0
        # dt = 0.25, so coefficient = 20.0 * 2.0 * 0.25 = 10.0
        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            0.2,
            2.0,
            IOM.ProductionCostExpression,
        )

        expected_coef = 20.0 * 2.0 * 0.25
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "add_proportional_cost_invariant! skips zero cost" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            0.0,
            1.0,
            IOM.ProductionCostExpression,
        )

        # Should have zero coefficients since cost_term is zero
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            0.0,
        )
    end

    @testset "add_variable_cost_to_objective! with CostCurve{LinearCurve}" begin
        @testset "30 \$/MWh" begin
            time_steps = 1:3
            # Device with 50 MW base power, system has 100 MW base
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 30 $/MWh
            cost_curve = IS.CostCurve(IS.LinearCurve(30.0))

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestLinearFormulation,
            )

            # Cost is in $/MWh, variable is in p.u. (system base)
            # proportional_term_per_unit = 30.0 * 100.0 (system_base) = 3000.0
            # With dt = 1.0, coefficient = 3000.0
            expected_coef = 30.0 * 100.0 * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "0.3 \$/MWh" begin
            time_steps = 1:3
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 0.3 $/MWh, i.e. 30 $/p.u.h at the 100 MW system base
            cost_curve = IS.CostCurve(IS.LinearCurve(0.3))

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestLinearFormulation,
            )

            # proportional_term_per_unit = 0.3 * 100.0 = 30.0
            # With dt = 1.0, coefficient = 30.0
            expected_coef = 30.0 * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "0.6 \$/MWh" begin
            time_steps = 1:3
            # Device with 50 MW base power, system has 100 MW base
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 0.6 $/MWh, i.e. 30 $ per device p.u. (50 MW) per hour
            cost_curve = IS.CostCurve(IS.LinearCurve(0.6))

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestLinearFormulation,
            )

            # Only the system base enters: 0.6 * 100.0 = 60.0
            # With dt = 1.0, coefficient = 60.0
            expected_coef = 60.0 * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "with non-unity resolution" begin
            time_steps = 1:3
            device = make_mock_thermal("gen1"; base_power = 100.0)
            # 15-minute resolution
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Minute(15),
            )

            # Cost: 20 $/MWh
            cost_curve = IS.CostCurve(IS.LinearCurve(20.0))

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestLinearFormulation,
            )

            # 15-min resolution:
            # proportional_term_per_unit = 20.0 * 100.0 = 2000.0
            # dt = 15 minutes / 60 = 0.25 hours
            # coefficient = 2000.0 * 0.25 = 500.0
            expected_coef = 20.0 * 100.0 * 0.25
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{LinearCurve}" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 50.0)
        container = setup_container_with_variables(
            time_steps,
            device;
            resolution = Dates.Hour(1),
        )

        # FuelCurve: fuel consumption rate (MMBTU/MWh) × fuel cost ($/MMBTU)
        # Linear fuel consumption: 8.0 MMBTU/MWh
        # Fuel cost: 5.0 $/MMBTU
        # Total cost: 8.0 * 5.0 = 40.0 $/MWh
        fuel_curve = IS.FuelCurve(
            IS.LinearCurve(8.0),  # MMBTU/MWh
            5.0,  # $/MMBTU
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestActivePowerVariable,
            device,
            fuel_curve,
            TestLinearFormulation,
        )

        # fuel_curve_per_unit = 8.0 * 100.0 (system_base) = 800.0
        # Total cost coefficient = fuel_curve_per_unit * fuel_cost * dt
        #                        = 800.0 * 5.0 * 1.0 = 4000.0
        expected_coef = 8.0 * 100.0 * 5.0 * 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "FuelCurve{LinearCurve} time-variant fuel cost adds to variant objective" begin
        # When fuel_cost is a TimeSeriesKey, the cost goes to the variant objective
        # and is computed as: FuelConsumptionExpression[name, t] * FuelCostParameter[name, t]
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container = setup_container_with_variables(
            time_steps,
            device;
            resolution = Dates.Hour(1),
        )

        # Set up FuelConsumptionExpression with known values.
        # In production, this is populated by add_expressions!(container,
        # FuelConsumptionExpression, devices, device_model) in POM's
        # thermalgeneration_constructor.jl. We pre-populate here to isolate
        # the IOM objective function logic from the POM constructor flow.
        fuel_expr_container = IOM.add_expression_container!(
            container,
            IOM.FuelConsumptionExpression,
            MockThermalGen,
            ["gen1"],
            time_steps,
        )
        jump_model = IOM.get_jump_model(container)
        power_var = IOM.get_variable(
            container, TestActivePowerVariable, MockThermalGen,
        )
        # Simulate fuel consumption = proportional_term * power_var
        proportional_term = 8.0  # MMBTU/p.u.h (already normalized)
        for t in time_steps
            JuMP.add_to_expression!(
                fuel_expr_container["gen1", t],
                proportional_term,
                power_var["gen1", t],
            )
        end

        # Set up FuelCostParameter with time-varying fuel prices
        fuel_prices = [5.0, 7.0, 3.0]
        add_test_parameter!(
            container,
            IOM.FuelCostParameter,
            MockThermalGen,
            ["gen1"],
            time_steps,
            reshape(fuel_prices, 1, :),
        )

        # Create a FuelCurve with a TimeSeriesKey as fuel_cost
        # Store-minted in production; fabricated here because this path reads the price
        # from the pre-populated FuelCostParameter, never from a store.
        ts_key = IS.TimeSeriesKey{IS.SingleTimeSeries{Float64}}(1)
        # 0.08 MMBTU/MWh = `proportional_term` MMBTU/p.u.h at the 100 MW system base.
        fuel_curve = IS.FuelCurve(IS.LinearCurve(0.08), ts_key)

        IOM.add_variable_cost_to_objective!(
            container,
            TestActivePowerVariable,
            device,
            fuel_curve,
            TestLinearFormulation,
        )

        # Variant objective should contain: fuel_expr * fuel_price for each t
        # = proportional_term * power_var * fuel_price
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)
        for t in time_steps
            var = power_var["gen1", t]
            expected = proportional_term * fuel_prices[t]
            @test JuMP.coefficient(variant, var) ≈ expected atol = 1e-10
        end
    end
end
