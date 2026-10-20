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

if !isdefined(InfrastructureOptimizationModelsTests, :TestPWLPminFormulation)
    struct TestPWLPminFormulation <:
           InfrastructureOptimizationModels.AbstractDeviceFormulation end
end
const TEST_PMIN = [0.2, 0.4]
InfrastructureOptimizationModels.objective_function_multiplier(
    ::Type{<:IOM.VariableType},
    ::Type{TestPWLPminFormulation},
) = 1.0
# A thermal mock with a per-step minimum power, so the hook override stays local to it.
struct MockPminStepGen <: AbstractMockGenerator
    name::String
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
end
get_name(g::MockPminStepGen) = g.name
IOM.get_active_power_limits(g::MockPminStepGen) = g.active_power_limits
IOM.get_base_power(::MockPminStepGen) = 100.0
IOM.get_operation_cost(::MockPminStepGen) = MockOperationCost(0.0)
InfrastructureOptimizationModels._sos_status(
    ::Type{MockPminStepGen},
    ::Type{TestPWLPminFormulation},
) = IOM.SOSStatusVariable.VARIABLE
IOM.get_min_active_power_values(
    ::IOM.OptimizationContainer,
    ::MockPminStepGen,
) = TEST_PMIN

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
    unit_system = IS.NaturalUnit(),
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

# Mocks for the _sos_status commitment traits
struct MockCommitGen <: AbstractMockGenerator end
struct MockPlainGen <: AbstractMockGenerator end
struct MockCommitFormulation <: IOM.AbstractDeviceFormulation end
struct MockNoCommitFormulation <: IOM.AbstractDeviceFormulation end
struct MockThermalCommitFormulation <: IOM.AbstractThermalUnitCommitment end
IOM.uses_commitment_variables(::Type{MockCommitGen}) = true
IOM.uses_commitment_formulation(::Type{MockCommitFormulation}) = true

# A named generator whose type uses commitment variables.
struct MockCommittedGen <: AbstractMockGenerator
    name::String
end
get_name(g::MockCommittedGen) = g.name
IOM.uses_commitment_variables(::Type{MockCommittedGen}) = true

@testset "_sos_status commitment traits" begin
    @test IOM._sos_status(MockCommitGen, MockCommitFormulation) ==
          IOM.SOSStatusVariable.VARIABLE
    @test IOM._sos_status(MockCommitGen, MockNoCommitFormulation) ==
          IOM.SOSStatusVariable.NO_VARIABLE
    @test IOM._sos_status(MockPlainGen, MockCommitFormulation) ==
          IOM.SOSStatusVariable.NO_VARIABLE
    @test IOM._sos_status(MockCommitGen, MockThermalCommitFormulation) ==
          IOM.SOSStatusVariable.VARIABLE
    @test IOM._sos_status(MockPlainGen, MockThermalCommitFormulation) ==
          IOM.SOSStatusVariable.NO_VARIABLE
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
                unit_system = IS.NaturalUnit(),
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
                unit_system = IS.SystemBaseUnit(),
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
            IS.NaturalUnit(),
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
            IS.NaturalUnit(),
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

    @testset "lambda link reads Pmin_t for PowerAboveMinimumVariable" begin
        points = [(x = 0.3, y = 5.0), (x = 0.6, y = 12.0), (x = 1.0, y = 25.0)]
        time_steps = 1:2
        device = MockPminStepGen("gen1", (min = 0.3, max = 1.0))
        sys = MockSystem(100.0)
        settings = IOM.Settings(
            sys; horizon = Dates.Hour(length(time_steps)), resolution = Dates.Hour(1))
        container =
            IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
        IOM.set_time_steps!(container, time_steps)
        jump_model = IOM.get_jump_model(container)
        on = IOM.add_variable_container!(
            container, IOM.OnVariable, MockPminStepGen, ["gen1"], time_steps)
        dp = IOM.add_variable_container!(
            container, IOM.PowerAboveMinimumVariable, MockPminStepGen,
            ["gen1"], time_steps)
        for t in time_steps
            on["gen1", t] = JuMP.@variable(jump_model, base_name = "on_$t")
            dp["gen1", t] = JuMP.@variable(jump_model, base_name = "dp_$t")
        end
        cost_curve = IS.CostCurve(
            IS.InputOutputCurve(IS.PiecewiseLinearData(points)), IS.NaturalUnit())

        IOM.add_variable_cost_to_objective!(
            container, IOM.PowerAboveMinimumVariable, device, cost_curve,
            TestPWLPminFormulation)

        con = IOM.get_constraint(
            container, IOM.PiecewiseLinearCostConstraint, MockPminStepGen)
        lambda = IOM.get_variable(
            container, IOM.PiecewiseLinearCostVariable, MockPminStepGen)
        for t in time_steps
            func = JuMP.constraint_object(con["gen1", t]).func
            @test JuMP.coefficient(func, on["gen1", t]) == TEST_PMIN[t]
            @test JuMP.coefficient(func, dp["gen1", t]) == 1.0
            for (k, p) in enumerate(points)
                @test JuMP.coefficient(func, lambda["gen1", k, t]) ≈ -p.x / 100.0
            end
        end
    end

    @testset "lambda link for ActivePowerVariable has no u_t term" begin
        time_steps = 1:2
        (; container, device, cost_curve) = setup_pwl_test(; time_steps)
        IOM.add_variable_cost_to_objective!(
            container, TestPWLVariable, device, cost_curve,
            TestPWLFormulation)
        con = IOM.get_constraint(
            container, IOM.PiecewiseLinearCostConstraint, MockThermalGen)
        for t in time_steps
            func = JuMP.constraint_object(con["gen1", t]).func
            @test JuMP.constant(func) == 0.0
            pwl = IOM.get_variable(
                container, IOM.PiecewiseLinearCostVariable, MockThermalGen)
            power = IOM.get_variable(container, TestPWLVariable, MockThermalGen)
            allowed = Set(vcat([power["gen1", t]], [pwl["gen1", k, t] for k in 1:3]))
            @test issubset(keys(func.terms), allowed)
        end
    end
end

@testset "add_pwl_term_lambda! links PowerAboveMinimumVariable to total power" begin
    device = make_mock_thermal("gen_compact"; limits = (min = 0.3, max = 1.0))
    sys = MockSystem(100.0)
    settings = IOM.Settings(sys; horizon = Dates.Hour(1), resolution = Dates.Hour(1))
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
    IOM.set_time_steps!(container, 1:1)
    p_container = IOM.add_variable_container!(
        container, IOM.PowerAboveMinimumVariable, MockThermalGen, ["gen_compact"], 1:1,
    )
    p_above = JuMP.@variable(IOM.get_jump_model(container), base_name = "p_above")
    p_container["gen_compact", 1] = p_above
    points = [(x = 0.3, y = 3.0), (x = 0.6, y = 7.0), (x = 1.0, y = 12.0)]
    cost = IS.CostCurve(
        IS.InputOutputCurve(IS.PiecewiseLinearData(points)),
        IS.SystemBaseUnit(),
    )

    IOM.add_pwl_term_lambda!(
        container, device, cost, IOM.PowerAboveMinimumVariable, TestPWLFormulation,
    )

    link = IOM.get_constraint(
        container, IOM.PiecewiseLinearCostConstraint, MockThermalGen,
    )[
        "gen_compact",
        1,
    ]
    func = JuMP.constraint_object(link).func
    lambda = IOM.get_variable(container, IOM.PiecewiseLinearCostVariable, MockThermalGen)
    @test JuMP.coefficient(func, p_above) == 1.0
    for (i, break_point) in enumerate((0.3, 0.6, 1.0))
        @test JuMP.coefficient(func, lambda["gen_compact", i, 1]) ≈ -break_point
    end
    # 1.0·P_min + p' == Σ λ_i P_i, so the normalized right-hand side is -P_min.
    @test JuMP.normalized_rhs(link) ≈ -0.3
end

@testset "Lambda PWL on status includes the component status" begin
    statuses = [0.0 0.5]

    @testset "_determine_bin_lhs" begin
        (; container) = setup_pwl_constraint_test(; time_steps = 1:2)
        set_test_statuses!(container, MockThermalGen, ["gen1"], statuses)
        for t in 1:2
            @test IOM._determine_bin_lhs(
                container, IOM.SOSStatusVariable.NO_VARIABLE, MockThermalGen, "gen1", t,
            ) == statuses[1, t]
        end
        add_test_parameter!(
            container, IOM.OnStatusParameter, MockThermalGen, ["gen1"], 1:2, [1.0 1.0])
        for t in 1:2
            @test IOM._determine_bin_lhs(
                container, IOM.SOSStatusVariable.PARAMETER, MockThermalGen, "gen1", t,
            ) == statuses[1, t]
        end
        on = IOM.add_variable_container!(
            container, IOM.OnVariable, MockThermalGen, ["gen1"], 1:2)
        jump_model = IOM.get_jump_model(container)
        for t in 1:2
            on["gen1", t] = JuMP.@variable(jump_model, binary = true)
            @test IOM._determine_bin_lhs(
                container, IOM.SOSStatusVariable.VARIABLE, MockThermalGen, "gen1", t,
            ) === on["gen1", t]
        end
    end

    @testset "get_on_status: the commitment variable first, when asked" begin
        container = make_test_container(1:2)
        device = MockCommittedGen("u")
        set_test_statuses!(container, MockCommittedGen, ["u"], statuses)
        on = IOM.add_variable_container!(
            container, IOM.OnVariable, MockCommittedGen, ["u"], 1:2)
        jump_model = IOM.get_jump_model(container)
        for t in 1:2
            on["u", t] = JuMP.@variable(jump_model, binary = true)
        end
        add_test_parameter!(
            container, IOM.OnStatusParameter, MockCommittedGen, ["u"], 1:2, [1.0 1.0])
        for t in 1:2
            # The piecewise linear rows read the parameter, scaled by the status.
            @test IOM.get_on_status(container, MockThermalCommitFormulation, device, t) ==
                  statuses[1, t]
            @test IOM.get_on_status(
                container, MockThermalCommitFormulation, device, t;
                commitment_first = true,
            ) === on["u", t]
            # A formulation without commitment has no variable to prefer.
            @test IOM.get_on_status(
                container, MockNoCommitFormulation, device, t; commitment_first = true,
            ) == statuses[1, t]
        end
    end

    @testset "a JuMP parameter on status is scaled, not replaced" begin
        (; container) = setup_pwl_constraint_test(; time_steps = 1:2)
        set_test_statuses!(container, MockThermalGen, ["gen1"], statuses)
        jump_model = IOM.get_jump_model(container)
        param_container = IOM.add_param_container_shared_axes!(
            container, IOM.ParameterKey(IOM.OnStatusParameter, MockThermalGen),
            IOM.NoAttributes(), JuMP.VariableRef, ["gen1"], 1:2)
        param_array = IOM.get_parameter_array(param_container)
        for t in 1:2
            param_array["gen1", t] = IOM.add_jump_parameter(jump_model, 1.0)
        end
        bin = IOM._determine_bin_lhs(
            container, IOM.SOSStatusVariable.PARAMETER, MockThermalGen, "gen1", 2)
        @test JuMP.coefficient(bin, param_array["gen1", 2]) ≈ 0.5
        @test IOM._determine_bin_lhs(
            container, IOM.SOSStatusVariable.PARAMETER, MockThermalGen, "gen1", 1,
        ) == 0.0
    end

    @testset "get_on_status" begin
        (; container, device) = setup_pwl_constraint_test(; time_steps = 1:2)
        @test IOM.get_on_status(container, TestPWLFormulation, device, 1) == 1.0
        set_test_statuses!(container, MockThermalGen, ["gen1"], statuses)
        @test IOM.get_on_status(container, TestPWLFormulation, device, 1) == 0.0
        @test IOM.get_on_status(container, TestPWLFormulation, device, 2) == 0.5
    end

    @testset "must-run normalization and linking rows" begin
        must_run_points =
            [(x = 20.0, y = 0.0), (x = 60.0, y = 200.0), (x = 100.0, y = 800.0)]
        for compact in (false, true)
            (; container, device, break_points) = setup_pwl_constraint_test(;
                time_steps = 1:2, points = must_run_points)
            set_test_statuses!(container, MockThermalGen, ["gen1"], statuses)
            power = IOM.get_variable(container, TestPWLVariable, MockThermalGen)
            for t in 1:2
                if compact
                    IOM._add_pwl_constraint_compact!(
                        container, device, "gen1", break_points,
                        IOM.SOSStatusVariable.VARIABLE, t, power["gen1", t], 20.0, true)
                else
                    IOM._add_pwl_constraint_standard!(
                        container, device, break_points,
                        IOM.SOSStatusVariable.VARIABLE, t, power["gen1", t], true)
                end
            end
            norm = IOM.get_constraint(
                container, IOM.PiecewiseLinearCostNormalizationConstraint,
                MockThermalGen)
            link = IOM.get_constraint(
                container, IOM.PiecewiseLinearCostConstraint, MockThermalGen)
            for t in 1:2
                @test JuMP.normalized_rhs(norm["gen1", t]) ≈ statuses[1, t]
                expected_link_rhs = compact ? -20.0 * statuses[1, t] : 0.0
                @test JuMP.normalized_rhs(link["gen1", t]) ≈ expected_link_rhs
            end
        end
    end
end
