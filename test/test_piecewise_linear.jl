"""
Unit tests for piecewise linear objective function construction.
Tests the functions in src/objective_function/piecewise_linear.jl using mock components.
"""

# Test-specific variable type
if !isdefined(InfrastructureOptimizationModelsTests, :TestPWLVariable)
    struct TestPWLVariable <: InfrastructureOptimizationModels.VariableType end
end

# Test-specific formulation
if !isdefined(InfrastructureOptimizationModelsTests, :TestPWLFormulation)
    struct TestPWLFormulation <: InfrastructureOptimizationModels.AbstractDeviceFormulation end
end

# Required stubs
InfrastructureOptimizationModels.objective_function_multiplier(
    ::Type{TestPWLVariable},
    ::Type{TestPWLFormulation},
) = 1.0

InfrastructureOptimizationModels._sos_status(
    ::Type{MockThermalGen},
    ::Type{TestPWLFormulation},
) = IOM.SOSStatusVariable.NO_VARIABLE

# Helper to set up container with variables for a device
function setup_pwl_container_with_variables(
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
        TestPWLVariable,
        MockThermalGen,
        [device_name],
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = InfrastructureOptimizationModels.get_jump_model(container)
    for t in time_steps
        var_container[device_name, t] = JuMP.@variable(
            jump_model,
            base_name = "TestPWLPower_$(device_name)_$(t)",
        )
    end

    return container
end

# Standard PWL points for testing
const CONVEX_PWL_POINTS = [
    (x = 0.0, y = 0.0),
    (x = 0.5, y = 10.0),
    (x = 1.0, y = 25.0),
]  # Slopes: 20, 30 (convex - increasing)

const NONCONVEX_PWL_POINTS = [
    (x = 0.0, y = 0.0),
    (x = 0.5, y = 20.0),
    (x = 1.0, y = 25.0),
]  # Slopes: 40, 10 (non-convex - decreasing)

"""
Helper to set up common test fixtures for PWL tests.
Returns (container, device, cost_curve, pwl_data).
"""
function setup_pwl_test(;
    time_steps = 1:2,
    device_name = "gen1",
    device_base_power = 100.0,
    resolution = Dates.Hour(1),
    points = CONVEX_PWL_POINTS,
    unit_system = IS.UnitSystem.NATURAL_UNITS,
    fuel_cost = nothing,  # If set, creates FuelCurve instead of CostCurve
)
    # When fuel_cost is provided, the device's operation_cost must also have it
    # because get_fuel_cost(device) is called to look up the cost multiplier
    if isnothing(fuel_cost)
        op_cost = MockOperationCost(0.0, false, 0.0)
    else
        op_cost = MockOperationCost(0.0, false, fuel_cost)
    end
    device = make_mock_thermal(
        device_name;
        base_power = device_base_power,
        operation_cost = op_cost,
    )
    container = setup_pwl_container_with_variables(time_steps, device; resolution)
    pwl_data = IS.PiecewiseLinearData(points)

    if isnothing(fuel_cost)
        cost_curve = IS.CostCurve(
            IS.InputOutputCurve(pwl_data),
            unit_system,
        )
    else
        cost_curve = IS.FuelCurve(
            IS.InputOutputCurve(pwl_data),
            unit_system,
            fuel_cost,
        )
    end

    return (; container, device, cost_curve, pwl_data)
end

"""
Set up a container with PWL variables already added, ready for constraint tests.
Returns (; container, device, pwl_data, break_points, power_var).
"""
function setup_pwl_constraint_test(;
    device_name = "gen1",
    time_steps = 1:1,
    points = CONVEX_PWL_POINTS,
    kw...,
)
    (; container, device, pwl_data) = setup_pwl_test(;
        time_steps, device_name, points, kw...,
    )
    for t in time_steps
        IOM.add_pwl_variables_lambda!(container, MockThermalGen, device_name, t, pwl_data)
    end
    break_points = IS.get_x_coords(pwl_data)
    power_var = IOM.get_variable(
        container, TestPWLVariable, MockThermalGen,
    )[
        device_name,
        first(time_steps),
    ]
    return (; container, device, pwl_data, break_points, power_var)
end

@testset "Piecewise Linear Objective Functions" begin
    @testset "add_pwl_variables_lambda!" begin
        (; container, pwl_data) = setup_pwl_test()

        # Add PWL variables for time period 1
        pwl_vars = InfrastructureOptimizationModels.add_pwl_variables_lambda!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        # length(PiecewiseLinearData) = number of segments; code adds +1 to get number of points
        @test length(pwl_vars) == length(pwl_data) + 1

        # Verify bounds are [0, 1]
        for var in pwl_vars
            @test JuMP.lower_bound(var) == 0.0
            @test JuMP.upper_bound(var) == 1.0
        end

        # Verify variables are stored in PiecewiseLinearCostVariable container
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        @test !isnothing(pwl_var_container)
        @test pwl_var_container["gen1", 1, 1] === pwl_vars[1]
    end

    @testset "_add_pwl_constraint_standard! creates linking and normalization constraints" begin
        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; time_steps = 1:1)

        InfrastructureOptimizationModels._add_pwl_constraint_standard!(
            container,
            device,
            break_points,
            IOM.SOSStatusVariable.NO_VARIABLE,
            1,
            power_var,
        )

        # Check that constraints were added
        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        @test JuMP.num_constraints(jump_model, JuMP.AffExpr, MOI.EqualTo{Float64}) >= 2

        # Verify constraint containers exist and contain constraint refs
        const_container = InfrastructureOptimizationModels.get_constraint(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostConstraint,
            MockThermalGen,
        )
        @test const_container["gen1", 1] isa JuMP.ConstraintRef

        norm_container = InfrastructureOptimizationModels.get_constraint(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )
        @test norm_container["gen1", 1] isa JuMP.ConstraintRef
    end

    @testset "get_pwl_cost_expression_lambda computes correct expression" begin
        (; container, pwl_data) = setup_pwl_constraint_test(; time_steps = 1:1)

        cost_expr = InfrastructureOptimizationModels.get_pwl_cost_expression_lambda(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
            1.0,
        )

        @test cost_expr isa JuMP.AffExpr

        # Verify coefficients match y_coords
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        y_coords = IS.get_y_coords(pwl_data)
        for (i, y) in enumerate(y_coords)
            var = pwl_var_container["gen1", i, 1]
            @test JuMP.coefficient(cost_expr, var) ≈ y
        end
    end

    @testset "get_pwl_cost_expression_lambda with multiplier" begin
        (; container, pwl_data) = setup_pwl_constraint_test(; time_steps = 1:1)

        multiplier = 2.5
        cost_expr = InfrastructureOptimizationModels.get_pwl_cost_expression_lambda(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
            multiplier,
        )

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        y_coords = IS.get_y_coords(pwl_data)
        for (i, y) in enumerate(y_coords)
            var = pwl_var_container["gen1", i, 1]
            @test JuMP.coefficient(cost_expr, var) ≈ y * multiplier
        end
    end

    @testset "add_pwl_sos2_constraint! adds SOS2 constraint for non-convex curves" begin
        (; container, pwl_data) =
            setup_pwl_constraint_test(; time_steps = 1:1, points = NONCONVEX_PWL_POINTS)

        @test !IS.is_convex(pwl_data)

        pwl_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearCostVariable, MockThermalGen,
        )
        pwl_vars = [pwl_var_container["gen1", i, 1] for i in 1:(length(pwl_data) + 1)]
        IOM.add_pwl_sos2_constraint!(container, MockThermalGen, "gen1", 1, pwl_vars)

        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        num_sos2 =
            JuMP.num_constraints(jump_model, Vector{JuMP.VariableRef}, MOI.SOS2{Float64})
        @test num_sos2 == 1
    end

    @testset "add_variable_cost_to_objective! with CostCurve{PiecewisePointCurve}" begin
        # Points in natural units (MW, $): (0, 0), (50, 1000), (100, 2500)
        natural_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 1000.0),
            (x = 100.0, y = 2500.0),
        ]

        @testset "NATURAL_UNITS" begin
            (; container, device, cost_curve) = setup_pwl_test(;
                device_base_power = 50.0,
                points = natural_points,
                unit_system = IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable,
                device,
                cost_curve,
                TestPWLFormulation,
            )

            # Verify PWL variables were created
            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
                MockThermalGen,
            )
            @test !isnothing(pwl_var_container)

            # NATURAL_UNITS: x_coords / base_power, y_coords unchanged
            # For 100 MW system base, points become (0, 0), (0.5, 1000), (1.0, 2500)
            # dt = 1.0
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)
            @test length(invariant.terms) > 0

            var_y0 = pwl_var_container["gen1", 1, 1]
            var_y1000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y0) ≈ 0.0 atol = 1e-10
            @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        end

        @testset "SYSTEM_BASE" begin
            # Points already in system base units (p.u., $)
            system_base_points = [
                (x = 0.0, y = 0.0),
                (x = 0.5, y = 1000.0),
                (x = 1.0, y = 2500.0),
            ]
            (; container, device, cost_curve) = setup_pwl_test(;
                device_base_power = 50.0,
                points = system_base_points,
                unit_system = IS.UnitSystem.SYSTEM_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable,
                device,
                cost_curve,
                TestPWLFormulation,
            )

            # SYSTEM_BASE: no conversion needed
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
                MockThermalGen,
            )
            var_y1000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        end

        @testset "with non-unity resolution (15 min)" begin
            linear_points = [(x = 0.0, y = 0.0), (x = 100.0, y = 2000.0)]
            (; container, device, cost_curve) = setup_pwl_test(;
                time_steps = 1:4,
                points = linear_points,
                resolution = Dates.Minute(15),
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable,
                device,
                cost_curve,
                TestPWLFormulation,
            )

            # dt = 15/60 = 0.25 hours, y=2000 * 0.25 = 500
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
                MockThermalGen,
            )
            var_y2000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y2000) ≈ 500.0 atol = 1e-10
        end

        @testset "non-convex curve adds SOS2" begin
            # Non-convex: slopes decrease (40, then 10)
            nonconvex_natural_points = [
                (x = 0.0, y = 0.0),
                (x = 50.0, y = 2000.0),
                (x = 100.0, y = 2500.0),
            ]
            (; container, device, cost_curve, pwl_data) = setup_pwl_test(;
                time_steps = 1:1,
                points = nonconvex_natural_points,
            )
            @test !IS.is_convex(pwl_data)

            @test_logs (:warn, r"not compatible with a linear PWL") InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable,
                device,
                cost_curve,
                TestPWLFormulation,
            )

            jump_model = InfrastructureOptimizationModels.get_jump_model(container)
            num_sos2 = JuMP.num_constraints(
                jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test num_sos2 == 1
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{PiecewisePointCurve}" begin
        # Fuel curve: heat rate (MMBTU/h) vs power (MW)
        # Points: (0 MW, 0), (50 MW, 400), (100 MW, 900) MMBTU/h
        fuel_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 400.0),
            (x = 100.0, y = 900.0),
        ]
        (; container, device, cost_curve) = setup_pwl_test(;
            points = fuel_points,
            fuel_cost = 5.0,  # $/MMBTU
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            cost_curve,
            TestPWLFormulation,
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        # Fuel consumption * fuel_cost * dt
        # y=400 * 5.0 * 1.0 = $2000, y=900 * 5.0 * 1.0 = $4500
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        var_y400 = pwl_var_container["gen1", 2, 1]
        var_y900 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y400) ≈ 2000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y900) ≈ 4500.0 atol = 1e-10
    end

    @testset "add_variable_cost_to_objective! with PiecewiseIncrementalCurve" begin
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container = setup_pwl_container_with_variables(1:2, device)

        # Incremental curve: marginal costs at each segment
        # x_coords: [0, 50, 100] MW, slopes: [20, 30] $/MWh
        # Converts to points: (0, 0), (50, 1000), (100, 2500)
        incremental_curve = IS.PiecewiseIncrementalCurve(
            0.0,                    # initial_input
            [0.0, 50.0, 100.0],     # x_coords
            [20.0, 30.0],           # slopes
        )
        cost_curve = IS.CostCurve(
            incremental_curve,  # Already an IncrementalCurve
            IS.UnitSystem.NATURAL_UNITS,
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            cost_curve,
            TestPWLFormulation,
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        var_y1000 = pwl_var_container["gen1", 2, 1]
        var_y2500 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y2500) ≈ 2500.0 atol = 1e-10
    end

    @testset "linear PWL (convex but not strictly convex)" begin
        # A PWL with constant slopes is convex but not strictly convex
        # This tests the edge case where is_convex returns true but slopes are equal
        # Points: (0, 0), (50, 1000), (100, 2000) - slope is constant 20 $/MWh
        linear_pwl_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 1000.0),
            (x = 100.0, y = 2000.0),
        ]
        (; container, device, cost_curve, pwl_data) = setup_pwl_test(;
            time_steps = 1:1,
            points = linear_pwl_points,
        )

        # Should be convex (equal slopes count as convex)
        @test IS.is_convex(pwl_data)

        # Should NOT add SOS2 constraint (no warning expected)
        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            cost_curve,
            TestPWLFormulation,
        )

        # Verify no SOS2 constraint was added
        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        num_sos2 =
            JuMP.num_constraints(jump_model, Vector{JuMP.VariableRef}, MOI.SOS2{Float64})
        @test num_sos2 == 0

        # Verify costs are correct
        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        var_y1000 = pwl_var_container["gen1", 2, 1]
        var_y2000 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y2000) ≈ 2000.0 atol = 1e-10
    end

    @testset "zero cost PWL is handled correctly" begin
        zero_cost_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 0.0),
            (x = 100.0, y = 0.0),
        ]
        (; container, device, cost_curve) = setup_pwl_test(; points = zero_cost_points)

        # Should return early without adding to objective
        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            cost_curve,
            TestPWLFormulation,
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)
        @test length(invariant.terms) == 0
    end

    @testset "_add_pwl_constraint_compact! linking constraint includes P_min offset" begin
        # Compact form: power_var + P_min * bin == Σ δ_i * breakpoint_i
        # With NO_VARIABLE status, bin = 1.0
        # So: power_var + P_min == Σ δ_i * breakpoint_i
        P_min = 30.0
        # Convex PWL: slopes 10, 20 (increasing)
        # (30, 0) → (60, 300) → (100, 1100)
        compact_points =
            [(x = 30.0, y = 0.0), (x = 60.0, y = 300.0), (x = 100.0, y = 1100.0)]

        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; time_steps = 1:1, points = compact_points)

        IOM._add_pwl_constraint_compact!(
            container,
            device,
            "gen1",
            break_points,
            IOM.SOSStatusVariable.NO_VARIABLE,
            1,
            power_var,
            P_min,
        )

        # Verify linking constraint: power_var + 30 == 30*δ₁ + 60*δ₂ + 100*δ₃
        linking_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        con_func = JuMP.constraint_object(linking_con).func
        # power_var has coefficient 1
        @test JuMP.coefficient(con_func, power_var) == 1.0
        # δ variables have negative coefficients (moved to LHS)
        pwl_vars = IOM.get_variable(
            container,
            IOM.PiecewiseLinearCostVariable,
            MockThermalGen,
        )
        for (i, bp) in enumerate(break_points)
            @test JuMP.coefficient(con_func, pwl_vars["gen1", i, 1]) == -bp
        end
        # RHS is -P_min (constant moved to RHS of == 0 form)
        con_set = JuMP.constraint_object(linking_con).set
        @test con_set.value ≈ -P_min

        # Verify normalization constraint: Σ δ_i == 1.0
        norm_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        norm_func = JuMP.constraint_object(norm_con).func
        for i in 1:length(break_points)
            @test JuMP.coefficient(norm_func, pwl_vars["gen1", i, 1]) == 1.0
        end
        norm_set = JuMP.constraint_object(norm_con).set
        @test norm_set.value ≈ 1.0
    end

    @testset "_determine_bin_lhs VARIABLE branch uses OnVariable" begin
        # When _sos_status returns VARIABLE, the normalization constraint
        # should reference the OnVariable for the device
        struct TestUCFormulation <: IOM.AbstractThermalUnitCommitment end
        IOM.objective_function_multiplier(
            ::Type{TestPWLVariable},
            ::Type{TestUCFormulation},
        ) = 1.0

        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; device_name = "gen_uc")

        # Add OnVariable container with a binary variable
        on_var_container = IOM.add_variable_container!(
            container,
            IOM.OnVariable,
            MockThermalGen,
            ["gen_uc"],
            1:1,
        )
        jump_model = IOM.get_jump_model(container)
        on_var = JuMP.@variable(jump_model, binary = true, base_name = "on_gen_uc")
        on_var_container["gen_uc", 1] = on_var

        IOM._add_pwl_constraint_standard!(
            container,
            device,
            break_points,
            IOM.SOSStatusVariable.VARIABLE,
            1,
            power_var,
        )

        # Normalization constraint should be: Σ δ_i == on_var
        norm_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )[
            "gen_uc",
            1,
        ]
        norm_func = JuMP.constraint_object(norm_con).func
        # on_var appears with coefficient -1 (moved to LHS: Σ δ_i - on_var == 0)
        @test JuMP.coefficient(norm_func, on_var) == -1.0
    end

    @testset "_determine_bin_lhs PARAMETER branch uses OnStatusParameter" begin
        # When OnStatusParameter exists in the container, _determine_bin_lhs
        # returns the parameter value (a Float64). The normalization constraint
        # becomes: Σ δ_i == param_value
        param_value = 0.75
        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; device_name = "gen_param")

        # Add OnStatusParameter container with a non-trivial value
        add_test_parameter!(
            container,
            IOM.OnStatusParameter,
            MockThermalGen,
            ["gen_param"],
            1:1,
            fill(param_value, 1, 1),
        )

        IOM._add_pwl_constraint_standard!(
            container,
            device,
            break_points,
            IOM.SOSStatusVariable.PARAMETER,
            1,
            power_var,
        )

        # Normalization constraint RHS should equal the parameter value
        norm_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )[
            "gen_param",
            1,
        ]
        norm_set = JuMP.constraint_object(norm_con).set
        @test norm_set.value ≈ param_value
    end

    @testset "FuelCurve: different fuel prices scale objective proportionally" begin
        # Two generators with identical heat rate curves but different fuel costs
        # Objective coefficients should scale proportionally with fuel_cost
        fuel_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 400.0),
            (x = 100.0, y = 900.0),
        ]
        y_coords = [0.0, 400.0, 900.0]

        for (fuel_cost, label) in [(3.0, "cheap"), (12.0, "expensive")]
            (; container, device, cost_curve) = setup_pwl_test(;
                time_steps = 1:1,
                points = fuel_points,
                fuel_cost = fuel_cost,
            )

            IOM.add_variable_cost_to_objective!(
                container,
                TestPWLVariable,
                device,
                cost_curve,
                TestPWLFormulation,
            )

            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            pwl_var_container = IOM.get_variable(
                container,
                IOM.PiecewiseLinearCostVariable,
                MockThermalGen,
            )

            # dt = 1.0 (hourly), multiplier = 1.0
            # Expected coefficient for point i: y_i * fuel_cost * dt
            for (i, y) in enumerate(y_coords)
                var = pwl_var_container["gen1", i, 1]
                expected = y * fuel_cost * 1.0
                @test JuMP.coefficient(invariant, var) ≈ expected atol = 1e-10
            end
        end
    end

    @testset "FuelCurve with non-unity resolution scales by dt" begin
        fuel_points = [
            (x = 0.0, y = 0.0),
            (x = 100.0, y = 800.0),
        ]
        fuel_cost = 10.0
        (; container, device, cost_curve) = setup_pwl_test(;
            time_steps = 1:4,
            points = fuel_points,
            fuel_cost = fuel_cost,
            resolution = Dates.Minute(15),
        )

        IOM.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            cost_curve,
            TestPWLFormulation,
        )

        obj = IOM.get_objective_expression(container)
        invariant = IOM.get_invariant_terms(obj)
        pwl_var_container = IOM.get_variable(
            container,
            IOM.PiecewiseLinearCostVariable,
            MockThermalGen,
        )

        # dt = 15/60 = 0.25, fuel_cost = 10.0, y = 800
        # Expected: 800 * 10.0 * 0.25 = 2000.0
        var_y800 = pwl_var_container["gen1", 2, 1]
        @test JuMP.coefficient(invariant, var_y800) ≈ 2000.0 atol = 1e-10
    end

    @testset "FuelCurve{PiecewisePointCurve} time-variant fuel cost goes to variant objective" begin
        fuel_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 400.0),
            (x = 100.0, y = 900.0),
        ]
        time_steps = 1:2
        fuel_prices = [3.0, 7.0]

        # Create container + device. We pass a scalar fuel_cost for the operation cost mock
        # but use a TimeSeriesKey in the FuelCurve itself.
        op_cost = MockOperationCost(0.0, false, 5.0)
        device = make_mock_thermal(
            "gen_tv"; base_power = 100.0, operation_cost = op_cost,
        )
        container = setup_pwl_container_with_variables(time_steps, device)

        # Pre-populate FuelCostParameter with time-varying fuel prices
        add_test_parameter!(
            container,
            IOM.FuelCostParameter,
            MockThermalGen,
            ["gen_tv"],
            time_steps,
            reshape(Float64.(fuel_prices), 1, :),
        )

        # Build FuelCurve with a TimeSeriesKey as fuel_cost to trigger is_time_variant
        ts_key = IS.StaticTimeSeriesKey(
            IS.SingleTimeSeries,
            "fuel_cost",
            Dates.DateTime(2024, 1, 1),
            Dates.Hour(1),
            length(time_steps),
            Dict{String, Any}(),
        )
        fuel_curve = IS.FuelCurve(
            IS.InputOutputCurve(IS.PiecewiseLinearData(fuel_points)),
            IS.UnitSystem.NATURAL_UNITS,
            ts_key,
        )

        IOM.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            fuel_curve,
            TestPWLFormulation,
        )

        # Cost should be in the VARIANT objective, not invariant
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)
        invariant = IOM.get_invariant_terms(obj)

        pwl_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearCostVariable, MockThermalGen,
        )

        # For t=1: y=400 point, fuel_price=3.0, dt=1.0 → cost coef = 400 * 3.0 = 1200
        var_y400_t1 = pwl_var_container["gen_tv", 2, 1]
        @test JuMP.coefficient(variant, var_y400_t1) ≈ 400.0 * fuel_prices[1] atol = 1e-10
        # Invariant should NOT contain this term
        @test JuMP.coefficient(invariant, var_y400_t1) ≈ 0.0 atol = 1e-10
    end

    @testset "FuelCurve{PiecewiseIncrementalCurve} converts and produces correct objective" begin
        # FuelCurve with incremental (marginal heat rate) data
        # x_coords: [0, 50, 100] MW, slopes: [8, 10] MMBTU/MWh
        # Converts to points: (0, 0), (50, 400), (100, 900) MMBTU/h
        fuel_cost = 5.0  # $/MMBTU
        incremental_curve = IS.PiecewiseIncrementalCurve(
            0.0,
            [0.0, 50.0, 100.0],
            [8.0, 10.0],
        )
        fuel_curve = IS.FuelCurve(
            incremental_curve,
            IS.UnitSystem.NATURAL_UNITS,
            fuel_cost,
        )
        op_cost = MockOperationCost(0.0, false, fuel_cost)
        device =
            make_mock_thermal("gen_fc_inc"; base_power = 100.0, operation_cost = op_cost)
        container = setup_pwl_container_with_variables(1:1, device)

        IOM.add_variable_cost_to_objective!(
            container,
            TestPWLVariable,
            device,
            fuel_curve,
            TestPWLFormulation,
        )

        obj = IOM.get_objective_expression(container)
        invariant = IOM.get_invariant_terms(obj)
        pwl_var_container = IOM.get_variable(
            container,
            IOM.PiecewiseLinearCostVariable,
            MockThermalGen,
        )

        # After conversion to points: (0,0), (50,400), (100,900)
        # Objective coefficients: y * fuel_cost * dt = y * 5.0 * 1.0
        @test JuMP.coefficient(invariant, pwl_var_container["gen_fc_inc", 1, 1]) ≈ 0.0 atol =
            1e-10
        @test JuMP.coefficient(invariant, pwl_var_container["gen_fc_inc", 2, 1]) ≈ 2000.0 atol =
            1e-10
        @test JuMP.coefficient(invariant, pwl_var_container["gen_fc_inc", 3, 1]) ≈ 4500.0 atol =
            1e-10
    end

    @testset "_add_pwl_constraint_compact! with must_run=true forces bin=1" begin
        # Even with VARIABLE sos_status, must_run should force bin=1.0
        # and NOT look for an OnVariable container
        P_min = 20.0
        # Convex PWL: slopes 5, 15 (increasing)
        # (20, 0) → (60, 200) → (100, 800)
        must_run_points =
            [(x = 20.0, y = 0.0), (x = 60.0, y = 200.0), (x = 100.0, y = 800.0)]

        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; time_steps = 1:1, points = must_run_points)

        # Pass VARIABLE status but must_run=true — should not look for OnVariable
        IOM._add_pwl_constraint_compact!(
            container,
            device,
            "gen1",
            break_points,
            IOM.SOSStatusVariable.VARIABLE,
            1,
            power_var,
            P_min,
            true,  # must_run
        )

        # Normalization: Σ δ_i == 1.0 (not a variable)
        norm_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        norm_set = JuMP.constraint_object(norm_con).set
        @test norm_set.value ≈ 1.0

        # Linking: constant term should be -P_min (since bin=1.0, P_min*1.0 = 20)
        linking_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        con_set = JuMP.constraint_object(linking_con).set
        @test con_set.value ≈ -P_min
    end

    @testset "_add_pwl_constraint_standard! with must_run=true forces bin=1.0" begin
        P_min = 20.0
        must_run_points =
            [(x = 20.0, y = 0.0), (x = 60.0, y = 200.0), (x = 100.0, y = 800.0)]
        (; container, device, break_points, power_var) =
            setup_pwl_constraint_test(; time_steps = 1:1, points = must_run_points)

        IOM._add_pwl_constraint_standard!(
            container,
            device,
            break_points,
            IOM.SOSStatusVariable.NO_VARIABLE,
            1,
            power_var,
            true,  # must_run
        )

        # Normalization constraint RHS should be 1.0 (bin forced)
        norm_con = IOM.get_constraint(
            container,
            IOM.PiecewiseLinearCostNormalizationConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        norm_set = JuMP.constraint_object(norm_con).set
        @test norm_set.value ≈ 1.0
    end

    @testset "_get_sos_value returns NO_VARIABLE when skip_proportional_cost" begin
        # Temporarily override skip_proportional_cost for our mock type
        IOM.skip_proportional_cost(::MockThermalGen) = true
        try
            setup = setup_pwl_test()
            result = IOM._get_sos_value(
                setup.container, TestPWLFormulation, setup.device,
            )
            @test result == IOM.SOSStatusVariable.NO_VARIABLE
        finally
            IOM.skip_proportional_cost(::MockThermalGen) = false
        end
    end

    @testset "_get_sos_value returns PARAMETER when OnStatusParameter exists" begin
        (; container, device) = setup_pwl_test(; time_steps = 1:1)

        # Add OnStatusParameter container
        add_test_parameter!(
            container,
            IOM.OnStatusParameter,
            MockThermalGen,
            ["gen1"],
            1:1,
            fill(1.0, 1, 1),
        )

        result = IOM._get_sos_value(container, TestPWLFormulation, device)
        @test result == IOM.SOSStatusVariable.PARAMETER
    end
end
