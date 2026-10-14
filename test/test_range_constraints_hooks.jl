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
