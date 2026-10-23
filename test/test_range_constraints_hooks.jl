struct TestPerStepFormulation <: IOM.AbstractDeviceFormulation end
struct TestDefaultLimitsFormulation <: IOM.AbstractDeviceFormulation end

const HOOK_MIN = [1.0, 2.0, 3.0]
const HOOK_MAX = fill(10.0, 3)

function IOM.get_min_max_limit_values(
    container::IOM.OptimizationContainer,
    ::Type{TestConstraintType},
    device::MockThermalGen,
    ::IOM.DeviceModel{MockThermalGen, TestPerStepFormulation},
)
    return (min = HOOK_MIN, max = HOOK_MAX)
end

IOM.get_min_max_limits(
    ::MockThermalGen,
    ::Type{TestConstraintType},
    ::Type{TestDefaultLimitsFormulation},
) = (min = 4.0, max = 9.0)

function _make_hook_container(devices, time_steps)
    container = _make_parameterized_range_container(devices, time_steps)
    names = [get_name(d) for d in devices]
    jump_model = IOM.get_jump_model(container)
    on = IOM.add_variable_container!(
        container, IOM.OnVariable, MockThermalGen, names, time_steps)
    for name in names, t in time_steps
        on[name, t] = JuMP.@variable(jump_model, binary = true)
    end
    return container
end

@testset "Per-step range limit hook" begin
    time_steps = 1:3
    devices = [make_mock_thermal("A")]
    step_model = IOM.DeviceModel(MockThermalGen, TestPerStepFormulation)
    default_model = IOM.DeviceModel(MockThermalGen, TestDefaultLimitsFormulation)

    @testset "range constraints use per-step values" begin
        container = _make_hook_container(devices, time_steps)
        IOM.add_range_constraints!(
            container, TestConstraintType, TestVariableType,
            devices, step_model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        for t in time_steps
            @test JuMP.normalized_rhs(con_lb["A", t]) ≈ HOOK_MIN[t]
            @test JuMP.normalized_rhs(con_ub["A", t]) ≈ HOOK_MAX[t]
        end
    end

    @testset "semicontinuous constraints use per-step values" begin
        container = _make_hook_container(devices, time_steps)
        IOM.add_semicontinuous_range_constraints!(
            container, TestConstraintType, TestVariableType,
            devices, step_model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        on = IOM.get_variable(container, IOM.OnVariable, MockThermalGen)
        for t in time_steps
            @test JuMP.normalized_coefficient(con_lb["A", t], on["A", t]) ≈ -HOOK_MIN[t]
            @test JuMP.normalized_coefficient(con_ub["A", t], on["A", t]) ≈ -HOOK_MAX[t]
        end
    end

    @testset "reserve bound constraints use per-step values" begin
        container = _make_hook_container(devices, time_steps)
        names = ["A"]
        jump_model = IOM.get_jump_model(container)
        rv = IOM.add_variable_container!(
            container, IOM.ReservationVariable, MockThermalGen, names, time_steps)
        for t in time_steps
            rv["A", t] = JuMP.@variable(jump_model, binary = true)
        end
        array = IOM.get_variable(container, TestVariableType, MockThermalGen)
        IOM.add_reserve_bound_range_constraints!(
            container, TestConstraintType, IOM.LowerBound(), array,
            devices, step_model, false)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        for t in time_steps
            @test JuMP.normalized_coefficient(con_lb["A", t], rv["A", t]) ≈ -HOOK_MIN[t]
        end
    end

    @testset "default values repeat get_min_max_limits" begin
        container = _make_hook_container(devices, time_steps)
        IOM.add_range_constraints!(
            container, TestConstraintType, TestVariableType,
            devices, default_model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        for t in time_steps
            @test JuMP.normalized_rhs(con_lb["A", t]) ≈ 4.0
            @test JuMP.normalized_rhs(con_ub["A", t]) ≈ 9.0
        end
    end
end

IOM.get_min_max_limits(
    ::MockThermalGen,
    ::Type{TestCostConstraint},
    ::Type{TestDefaultLimitsFormulation},
) = (min = 4.0, max = 9.0)

@testset "Range rows scale by the component status" begin
    time_steps = 1:3
    devices = [make_mock_thermal("A"), make_mock_thermal("B")]
    model = IOM.DeviceModel(MockThermalGen, TestDefaultLimitsFormulation)
    statuses = [1.0 0.0 1.0; 0.0 1.0 0.5]

    @testset "plain range bounds" begin
        container = _make_hook_container(devices, time_steps)
        set_test_statuses!(container, MockThermalGen, ["A", "B"], statuses)
        IOM.add_range_constraints!(
            container, TestConstraintType, TestVariableType,
            devices, model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        for (i, name) in enumerate(["A", "B"]), t in time_steps
            @test JuMP.normalized_rhs(con_lb[name, t]) ≈ 4.0 * statuses[i, t]
            @test JuMP.normalized_rhs(con_ub[name, t]) ≈ 9.0 * statuses[i, t]
        end
    end

    @testset "semicontinuous rows read the status for names without an OnVariable" begin
        must_run_devices = [make_mock_thermal("A"), make_mock_thermal("B"; must_run = true)]
        container = _make_parameterized_range_container(must_run_devices, time_steps)
        jump_model = IOM.get_jump_model(container)
        on = IOM.add_variable_container!(
            container, IOM.OnVariable, MockThermalGen, ["A"], time_steps)
        for t in time_steps
            on["A", t] = JuMP.@variable(jump_model, binary = true)
        end
        set_test_statuses!(container, MockThermalGen, ["A", "B"], statuses)
        IOM.add_semicontinuous_range_constraints!(
            container, TestConstraintType, TestVariableType,
            must_run_devices, model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        for t in time_steps
            @test JuMP.normalized_coefficient(con_ub["A", t], on["A", t]) ≈ -9.0
            @test JuMP.normalized_rhs(con_ub["A", t]) ≈ 0.0
            @test JuMP.normalized_rhs(con_lb["B", t]) ≈ 4.0 * statuses[2, t]
            @test JuMP.normalized_rhs(con_ub["B", t]) ≈ 9.0 * statuses[2, t]
        end
    end

    @testset "reservation rows scale by the status" begin
        container = _make_hook_container(devices, time_steps)
        jump_model = IOM.get_jump_model(container)
        reservation = IOM.add_variable_container!(
            container, IOM.ReservationVariable, MockThermalGen, ["A", "B"], time_steps)
        for name in ["A", "B"], t in time_steps
            reservation[name, t] = JuMP.@variable(jump_model, binary = true)
        end
        set_test_statuses!(container, MockThermalGen, ["A", "B"], statuses)
        array = IOM.get_variable(container, TestVariableType, MockThermalGen)
        IOM.add_reserve_bound_range_constraints!(
            container, TestConstraintType, IOM.LowerBound(), array, devices, model, true)
        con = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        for (i, name) in enumerate(["A", "B"]), t in time_steps
            # x >= min * s * (1 - r): the constant moves to the right-hand side.
            @test JuMP.normalized_rhs(con[name, t]) ≈ 4.0 * statuses[i, t]
            @test JuMP.normalized_coefficient(con[name, t], reservation[name, t]) ≈
                  4.0 * statuses[i, t]
        end
    end

    @testset "a semicontinuous name without an OnVariable must be always committed" begin
        container = _make_parameterized_range_container(devices, time_steps)
        jump_model = IOM.get_jump_model(container)
        on = IOM.add_variable_container!(
            container, IOM.OnVariable, MockThermalGen, ["A"], time_steps)
        for t in time_steps
            on["A", t] = JuMP.@variable(jump_model, binary = true)
        end
        @test !IOM.is_always_committed(devices[2])
        @test_throws IS.ConflictingInputsError IOM.add_semicontinuous_range_constraints!(
            container, TestConstraintType, TestVariableType,
            devices, model, TestPowerModel)
    end

    @testset "bounds on a variable that stays free while inactive keep their value" begin
        container = _make_hook_container(devices, time_steps)
        jump_model = IOM.get_jump_model(container)
        level = IOM.add_variable_container!(
            container, TestVariableType2, MockThermalGen, ["A", "B"], time_steps)
        for name in ["A", "B"], t in time_steps
            level[name, t] = JuMP.@variable(jump_model)
        end
        set_test_statuses!(container, MockThermalGen, ["A", "B"], statuses)
        @test !IOM.fixed_when_inactive(TestVariableType2)
        IOM.add_range_constraints!(
            container, TestConstraintType, TestVariableType2,
            devices, model, TestPowerModel)
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        for name in ["A", "B"], t in time_steps
            @test JuMP.normalized_rhs(con_lb[name, t]) ≈ 4.0
            @test JuMP.normalized_rhs(con_ub[name, t]) ≈ 9.0
        end
    end

    @testset "all-ones statuses build the same rows as the default" begin
        function build(with_statuses)
            container = _make_hook_container(devices, time_steps)
            if with_statuses
                set_test_statuses!(container, MockThermalGen, ["A", "B"], ones(2, 3))
            end
            IOM.add_range_constraints!(
                container, TestConstraintType, TestVariableType,
                devices, model, TestPowerModel)
            IOM.add_semicontinuous_range_constraints!(
                container, TestCostConstraint, TestVariableType,
                devices, model, TestPowerModel)
            return lp_text(container)
        end
        @test build(true) == build(false)
    end
end
