struct WarmStartTestVariable <: IOM.VariableType end

IOM.get_variable_binary(
    ::Type{WarmStartTestVariable},
    ::Type{MockThermalGen},
    ::Type{TestDeviceFormulation},
) = false
IOM.get_variable_warm_start_value(
    ::Type{WarmStartTestVariable},
    d::MockThermalGen,
    ::Type{TestDeviceFormulation},
) = d.active_power_limits.max

struct SkippedTestVariable <: IOM.VariableType end

IOM.get_variable_binary(
    ::Type{SkippedTestVariable},
    ::Type{MockThermalGen},
    ::Type{TestDeviceFormulation},
) = false
IOM.skip_variable(
    ::Type{SkippedTestVariable},
    d::MockThermalGen,
    ::Type{TestDeviceFormulation},
) = d.must_run

function _warm_start_container(warm_start::Bool)
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(2),
        resolution = Dates.Hour(1),
        warm_start = warm_start,
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:2)
    return container
end

@testset "Start values" begin
    @testset "set_start_value! follows the warm_start setting" begin
        for warm_start in (true, false)
            container = _warm_start_container(warm_start)
            var = JuMP.@variable(IOM.get_jump_model(container))
            IOM.set_start_value!(container, var, 2.5)
            @test JuMP.start_value(var) == (warm_start ? 2.5 : nothing)
        end
    end

    @testset "set_start_value! with nothing sets no start value" begin
        container = _warm_start_container(true)
        var = JuMP.@variable(IOM.get_jump_model(container))
        IOM.set_start_value!(container, var, nothing)
        @test JuMP.start_value(var) === nothing
    end

    @testset "add_variables! applies the warm start hook only with warm_start" begin
        devices = [make_mock_thermal("gen1"; limits = (min = 0.0, max = 40.0))]
        for warm_start in (true, false)
            container = _warm_start_container(warm_start)
            IOM.add_variables!(
                container,
                WarmStartTestVariable,
                devices,
                TestDeviceFormulation,
            )
            var = IOM.get_variable(container, WarmStartTestVariable, MockThermalGen)
            for t in 1:2
                @test JuMP.start_value(var["gen1", t]) == (warm_start ? 40.0 : nothing)
            end
        end
    end

    @testset "add_variables! leaves skipped devices off the axis" begin
        container = _warm_start_container(true)
        devices = [
            make_mock_thermal("committed"),
            make_mock_thermal("must_run"; must_run = true),
        ]
        IOM.add_variables!(container, SkippedTestVariable, devices, TestDeviceFormulation)
        var = IOM.get_variable(container, SkippedTestVariable, MockThermalGen)
        @test axes(var)[1] == ["committed"]

        container = _warm_start_container(true)
        all_skipped = [make_mock_thermal("must_run"; must_run = true)]
        IOM.add_variables!(
            container,
            SkippedTestVariable,
            all_skipped,
            TestDeviceFormulation,
        )
        var = IOM.get_variable(container, SkippedTestVariable, MockThermalGen)
        @test isempty(axes(var)[1])
    end
end
