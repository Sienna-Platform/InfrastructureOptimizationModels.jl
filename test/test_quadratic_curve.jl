"""
Unit tests for quadratic curve objective function construction.
Tests the functions in src/objective_function/quadratic_curve.jl using mock components.
"""

# Test-specific variable type (reuse from linear if already defined, otherwise define)
if !@isdefined(TestActivePowerVariable)
    struct TestActivePowerVariable <: InfrastructureOptimizationModels.VariableType end
end

# Test-specific formulation
if !@isdefined(TestQuadraticFormulation)
    struct TestQuadraticFormulation <:
           InfrastructureOptimizationModels.AbstractDeviceFormulation end
end

# Stub: objective_function_multiplier returns 1.0 for test types
InfrastructureOptimizationModels.objective_function_multiplier(
    ::Type{TestActivePowerVariable},
    ::Type{TestQuadraticFormulation},
) = 1.0

# Helper to set up container with variables for a device (same pattern as linear tests)
function setup_quadratic_test_container(
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

@testset "Quadratic Curve Objective Functions" begin
    @testset "_add_quadraticcurve_variable_term_to_model! adds quadratic cost" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_quadratic_test_container(time_steps, device; resolution = Dates.Hour(1))

        # quadratic=2.0, linear=5.0 per unit
        # With dt=1.0 (1 hour resolution), final coeffs should be same
        quadratic_term = 2.0
        linear_term = 5.0

        InfrastructureOptimizationModels._add_quadraticcurve_variable_term_to_model!(
            container,
            TestActivePowerVariable,
            device,
            linear_term,
            quadratic_term,
            1,  # time period
        )

        # Verify coefficients (dt=1.0 so same as input)
        quad_coef = get_objective_quadratic_coefficient(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            1,
        )
        lin_coef = get_objective_coefficient(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            1,
        )

        @test quad_coef ≈ quadratic_term  # dt=1.0
        @test lin_coef ≈ linear_term  # dt=1.0

        # Other time periods should have zero coefficients
        for t in 2:3
            @test get_objective_quadratic_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            ) ≈ 0.0
            @test get_objective_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            ) ≈ 0.0
        end
    end

    @testset "_add_quadraticcurve_variable_term_to_model! with dt scaling" begin
        time_steps = 1:2
        device = make_mock_thermal("gen1"; base_power = 100.0)
        # 30 minute resolution gives dt=0.5
        container = setup_quadratic_test_container(
            time_steps,
            device;
            resolution = Dates.Minute(30),
        )

        quadratic_term = 3.0
        linear_term = 7.0

        InfrastructureOptimizationModels._add_quadraticcurve_variable_term_to_model!(
            container,
            TestActivePowerVariable,
            device,
            linear_term,
            quadratic_term,
            1,
        )

        quad_coef = get_objective_quadratic_coefficient(
            container, TestActivePowerVariable, MockThermalGen, "gen1", 1,
        )
        lin_coef = get_objective_coefficient(
            container, TestActivePowerVariable, MockThermalGen, "gen1", 1,
        )

        # dt = 0.5 for 30 minute resolution
        @test quad_coef ≈ quadratic_term * 0.5
        @test lin_coef ≈ linear_term * 0.5
    end

    @testset "_add_quadraticcurve_variable_cost! scalar - same cost all time steps" begin
        time_steps = 1:4
        # Use limits that pass monotonicity check
        device =
            make_mock_thermal("gen1"; base_power = 100.0, limits = (min = 0.0, max = 100.0))
        container = setup_quadratic_test_container(
            time_steps, device; resolution = Dates.Hour(1),
        )

        # Scalar costs
        proportional_term = 10.0
        quadratic_term = 2.0

        InfrastructureOptimizationModels._add_quadraticcurve_variable_cost!(
            container,
            TestActivePowerVariable,
            TestQuadraticFormulation,
            device,
            proportional_term,
            quadratic_term,
        )

        # With 1-hour resolution, dt = 1.0
        # Coefficients are scaled by dt in _add_quadraticcurve_variable_term_to_model!
        expected_linear = proportional_term  # Note: actual scaling depends on implementation
        expected_quadratic = quadratic_term

        # Verify all time steps have same coefficients
        for t in time_steps
            quad_coef = get_objective_quadratic_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            lin_coef = get_objective_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            # dt=1.0 multiplier applied
            @test quad_coef ≈ expected_quadratic * 1.0
            @test lin_coef ≈ expected_linear * 1.0
        end
    end

    @testset "_add_quadraticcurve_variable_cost! vector - different costs per time step" begin
        time_steps = 1:3
        device =
            make_mock_thermal("gen1"; base_power = 100.0, limits = (min = 0.0, max = 100.0))
        container = setup_quadratic_test_container(
            time_steps, device; resolution = Dates.Hour(1),
        )

        # Time-varying costs
        proportional_terms = [5.0, 10.0, 15.0]
        quadratic_terms = [1.0, 2.0, 3.0]

        InfrastructureOptimizationModels._add_quadraticcurve_variable_cost!(
            container,
            TestActivePowerVariable,
            TestQuadraticFormulation,
            device,
            proportional_terms,
            quadratic_terms,
        )

        # Verify each time step has correct coefficients (with dt=1.0)
        for t in time_steps
            quad_coef = get_objective_quadratic_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            lin_coef = get_objective_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            @test quad_coef ≈ quadratic_terms[t] * 1.0
            @test lin_coef ≈ proportional_terms[t] * 1.0
        end
    end

    @testset "add_variable_cost_to_objective! with CostCurve{QuadraticCurve}" begin
        @testset "NATURAL_UNITS" begin
            time_steps = 1:3
            device = make_mock_thermal(
                "gen1";
                base_power = 50.0,
                limits = (min = 0.0, max = 100.0),
            )
            container = setup_quadratic_test_container(
                time_steps, device; resolution = Dates.Hour(1),
            )

            # Cost: quadratic=0.5, linear=20 in natural units (MW)
            cost_curve = IS.CostCurve(
                IS.QuadraticCurve(0.5, 20.0, 0.0),  # (quadratic, linear, constant)
                IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestQuadraticFormulation,
            )

            # NATURAL_UNITS conversion:
            # linear_per_unit = linear * system_base = 20.0 * 100.0 = 2000.0
            # quadratic_per_unit = quadratic * system_base^2 = 0.5 * 100.0^2 = 5000.0
            # With dt = 1.0
            expected_linear = 20.0 * 100.0 * 1.0
            expected_quadratic = 0.5 * 100.0 * 100.0 * 1.0

            @test verify_quadratic_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_linear,
                expected_quadratic,
            )
        end

        @testset "SYSTEM_BASE" begin
            time_steps = 1:3
            device = make_mock_thermal(
                "gen1";
                base_power = 50.0,
                limits = (min = 0.0, max = 100.0),
            )
            container = setup_quadratic_test_container(
                time_steps, device; resolution = Dates.Hour(1),
            )

            # Cost in system base units - no conversion needed
            cost_curve = IS.CostCurve(
                IS.QuadraticCurve(2.0, 30.0, 0.0),
                IS.UnitSystem.SYSTEM_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestQuadraticFormulation,
            )

            # SYSTEM_BASE: no conversion
            expected_linear = 30.0 * 1.0
            expected_quadratic = 2.0 * 1.0

            @test verify_quadratic_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_linear,
                expected_quadratic,
            )
        end

        @testset "DEVICE_BASE" begin
            time_steps = 1:3
            # Device with 50 MW base, system has 100 MW base
            device = make_mock_thermal(
                "gen1";
                base_power = 50.0,
                limits = (min = 0.0, max = 100.0),
            )
            container = setup_quadratic_test_container(
                time_steps, device; resolution = Dates.Hour(1),
            )

            cost_curve = IS.CostCurve(
                IS.QuadraticCurve(1.0, 20.0, 0.0),
                IS.UnitSystem.DEVICE_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestQuadraticFormulation,
            )

            # DEVICE_BASE conversion:
            # linear: cost * (system_base / device_base) = 20.0 * (100/50) = 40.0
            # quadratic: cost * (system_base / device_base)^2 = 1.0 * (100/50)^2 = 4.0
            expected_linear = 20.0 * (100.0 / 50.0) * 1.0
            expected_quadratic = 1.0 * (100.0 / 50.0)^2 * 1.0

            @test verify_quadratic_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_linear,
                expected_quadratic,
            )
        end

        @testset "with non-unity resolution" begin
            time_steps = 1:3
            device = make_mock_thermal(
                "gen1";
                base_power = 100.0,
                limits = (min = 0.0, max = 100.0),
            )
            # 15-minute resolution
            container = setup_quadratic_test_container(
                time_steps, device; resolution = Dates.Minute(15),
            )

            cost_curve = IS.CostCurve(
                IS.QuadraticCurve(2.0, 40.0, 0.0),
                IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable,
                device,
                cost_curve,
                TestQuadraticFormulation,
            )

            # NATURAL_UNITS with 15-min resolution (dt = 0.25):
            # linear_per_unit = 40.0 * 100.0 = 4000.0, then * 0.25 = 1000.0
            # quadratic_per_unit = 2.0 * 100.0^2 = 20000.0, then * 0.25 = 5000.0
            expected_linear = 40.0 * 100.0 * 0.25
            expected_quadratic = 2.0 * 100.0 * 100.0 * 0.25

            @test verify_quadratic_objective_coefficients(
                container,
                TestActivePowerVariable,
                MockThermalGen,
                "gen1",
                expected_linear,
                expected_quadratic,
            )
        end
    end

    @testset "quadratic fallback to linear when quadratic term is zero" begin
        time_steps = 1:2
        device =
            make_mock_thermal("gen1"; base_power = 100.0, limits = (min = 0.0, max = 100.0))
        container = setup_quadratic_test_container(time_steps, device)

        # Quadratic term is essentially zero - should fall back to linear
        proportional_term = 15.0
        quadratic_term = 0.0

        InfrastructureOptimizationModels._add_quadraticcurve_variable_cost!(
            container,
            TestActivePowerVariable,
            TestQuadraticFormulation,
            device,
            proportional_term,
            quadratic_term,
        )

        # Should only have linear terms, no quadratic
        for t in time_steps
            quad_coef = get_objective_quadratic_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            lin_coef = get_objective_coefficient(
                container, TestActivePowerVariable, MockThermalGen, "gen1", t,
            )
            @test quad_coef ≈ 0.0
            @test lin_coef ≈ proportional_term * 1.0  # dt = 1.0
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{QuadraticCurve}" begin
        time_steps = 1:3
        device = make_mock_thermal(
            "gen1";
            base_power = 50.0,
            limits = (min = 0.0, max = 100.0),
        )
        container = setup_quadratic_test_container(
            time_steps, device; resolution = Dates.Hour(1),
        )

        # FuelCurve with quadratic fuel consumption
        # Quadratic: 0.02 MMBTU/MW²h, Linear: 7.0 MMBTU/MWh
        # Fuel cost: 4.0 $/MMBTU
        fuel_curve = IS.FuelCurve(
            IS.QuadraticCurve(0.02, 7.0, 0.0),  # (quadratic, linear, constant)
            IS.UnitSystem.NATURAL_UNITS,
            4.0,  # $/MMBTU
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestActivePowerVariable,
            device,
            fuel_curve,
            TestQuadraticFormulation,
        )

        # NATURAL_UNITS conversion:
        # linear_per_unit = 7.0 * 100.0 (system_base) = 700.0
        # quadratic_per_unit = 0.02 * 100.0^2 = 200.0
        # With fuel_cost = 4.0 and dt = 1.0:
        # final_linear = 700.0 * 4.0 * 1.0 = 2800.0
        # final_quadratic = 200.0 * 4.0 * 1.0 = 800.0
        expected_linear = 7.0 * 100.0 * 4.0 * 1.0
        expected_quadratic = 0.02 * 100.0 * 100.0 * 4.0 * 1.0

        @test verify_quadratic_objective_coefficients(
            container,
            TestActivePowerVariable,
            MockThermalGen,
            "gen1",
            expected_linear,
            expected_quadratic,
        )
    end

    @testset "_check_quadratic_monotonicity warns for non-monotonic cost" begin
        # derivative f'(x) = 2*quad*x + linear
        # With quad=-1.0, linear=0.5: f'(0) = 0.5, f'(1) = -1.5 (negative → warning)
        @test_logs (:warn, r"not monotonically increasing") InfrastructureOptimizationModels._check_quadratic_monotonicity(
            "test_gen", -1.0, 0.5, 0.0, 1.0,
        )
        # Positive-definite case: no warning
        @test_logs InfrastructureOptimizationModels._check_quadratic_monotonicity(
            "test_gen", 1.0, 1.0, 0.0, 1.0,
        )
    end

    @testset "_add_quadraticcurve_variable_term_to_model! populates ProductionCostExpression" begin
        time_steps = 1:2
        device = make_mock_thermal("gen_expr"; base_power = 100.0)
        container =
            setup_quadratic_test_container(time_steps, device; resolution = Dates.Hour(1))

        # Add a ProductionCostExpression container so the branch is exercised
        # Use QuadExpr type since the quadratic cost produces QuadExpr terms
        InfrastructureOptimizationModels.add_expression_container!(
            container,
            InfrastructureOptimizationModels.ProductionCostExpression,
            MockThermalGen,
            ["gen_expr"],
            time_steps;
            expr_type = JuMP.QuadExpr,
        )
        expr_container = InfrastructureOptimizationModels.get_expression(
            container,
            InfrastructureOptimizationModels.ProductionCostExpression,
            MockThermalGen,
        )

        quadratic_term = 2.0
        linear_term = 5.0
        InfrastructureOptimizationModels._add_quadraticcurve_variable_term_to_model!(
            container,
            TestActivePowerVariable,
            device,
            linear_term,
            quadratic_term,
            1,
        )

        # The expression at [name, t=1] should be non-zero (cost was added)
        cost_expr = expr_container["gen_expr", 1]
        # Should be a QuadExpr with non-trivial quadratic terms
        @test cost_expr isa JuMP.GenericQuadExpr
        @test !isempty(cost_expr.terms)
    end
end
