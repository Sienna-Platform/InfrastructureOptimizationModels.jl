"""
Unit tests for rebuild-time parameters (`requires_rebuild`), the parameter reuse guard
(`should_add_parameter`, `on_parameter_reused!`), `reset_optimization_model!`,
`rebuild_model!` and the per-model parameter refresh hook (`update_model_parameters!`).
"""

struct StructuralTestParameter <: IOM.ParameterType end
struct OtherStructuralTestParameter <: IOM.ParameterType end
struct ReusedTestParameter <: IOM.ParameterType end
IOM.requires_rebuild(::Type{StructuralTestParameter}) = true
IOM.requires_rebuild(::Type{OtherStructuralTestParameter}) = true

const REUSED_PARAMETER_CALLS = Any[]
IOM.on_parameter_reused!(
    ::IOM.OptimizationContainer,
    key::IOM.ParameterKey{ReusedTestParameter, MockThermalGen},
    devices,
    model,
) = push!(REUSED_PARAMETER_CALLS, (key, devices, model))

function _rebuild_container(; recurrent = true, rebuild_model = false, time_steps = 1:2)
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
        rebuild_model = rebuild_model,
        initial_time = DateTime("2024-01-01T00:00:00"),
    )
    container = IOM.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, time_steps)
    container.built_for_recurrent_solves = recurrent
    return container
end

function _add_shared_parameter!(container, ::Type{P}, names, time_steps) where {P}
    return IOM.add_param_container_shared_axes!(
        container, IOM.ParameterKey(P, MockThermalGen), IOM.NoAttributes(),
        IOM.get_param_eltype(container, P), names, time_steps)
end

@testset "Rebuild-time parameters" begin
    @testset "requires_rebuild trait" begin
        @test !IOM.requires_rebuild(TestParameterType)
        @test IOM.requires_rebuild(StructuralTestParameter)
        @test IOM.requires_rebuild(IOM.LeftHandSideTimeSeriesParameter)
    end

    @testset "get_param_eltype by parameter type" begin
        container = _rebuild_container()
        @test IOM.get_param_eltype(container, TestParameterType) == JuMP.VariableRef
        @test IOM.get_param_eltype(container, StructuralTestParameter) == Float64
        non_recurrent = _rebuild_container(; recurrent = false)
        @test IOM.get_param_eltype(non_recurrent, TestParameterType) == Float64
    end

    @testset "a rebuild-time parameter requires rebuild_model under recurrent solves" begin
        container = _rebuild_container()
        @test_throws r"StructuralTestParameter.*rebuild_model = true" _add_shared_parameter!(
            container, StructuralTestParameter, ["A"], 1:2)
        @test !IOM.get_rebuild_model(IOM.get_settings(container))
        @test !IOM.has_container_key(container, StructuralTestParameter, MockThermalGen)

        rebuilt = _rebuild_container(; rebuild_model = true)
        _add_shared_parameter!(rebuilt, StructuralTestParameter, ["A"], 1:2)
        key = IOM.ParameterKey(StructuralTestParameter, MockThermalGen)
        @test eltype(IOM.get_parameter_array(rebuilt, key)) == Float64
    end

    @testset "allowed outside recurrent solves" begin
        container = _rebuild_container(; recurrent = false)
        _add_shared_parameter!(container, StructuralTestParameter, ["A"], 1:2)
        @test IOM.has_container_key(container, StructuralTestParameter, MockThermalGen)
        @test !IOM.get_rebuild_model(IOM.get_settings(container))
    end

    @testset "should_add_parameter and on_parameter_reused!" begin
        empty!(REUSED_PARAMETER_CALLS)
        key = IOM.ParameterKey(ReusedTestParameter, MockThermalGen)
        container = _rebuild_container(; rebuild_model = true)
        @test IOM.should_add_parameter(container, key)
        _add_shared_parameter!(container, ReusedTestParameter, ["A"], 1:2)
        @test !IOM.should_add_parameter(container, key)
        devices = [make_mock_thermal("A")]
        model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)
        @test !IOM.should_add_parameter(container, key, devices, model)
        @test only(REUSED_PARAMETER_CALLS) == (key, devices, model)

        no_rebuild = _rebuild_container(; recurrent = false)
        _add_shared_parameter!(no_rebuild, ReusedTestParameter, ["A"], 1:2)
        @test IOM.should_add_parameter(no_rebuild, key)
        @test IOM.should_add_parameter(no_rebuild, key, devices, model)
        @test length(REUSED_PARAMETER_CALLS) == 1

        other_key = IOM.ParameterKey(TestParameterType, MockThermalGen)
        _add_shared_parameter!(container, TestParameterType, ["A"], 1:2)
        @test !IOM.should_add_parameter(container, other_key, devices, model)
        @test length(REUSED_PARAMETER_CALLS) == 1
    end
end

@testset "reset_optimization_model!" begin
    @testset "keeps parameters and their values" begin
        container = _rebuild_container(; rebuild_model = true)
        param_container = _add_shared_parameter!(container, TestParameterType, ["A"], 1:2)
        IOM.get_parameter_array(param_container)["A", 2] = 7.0
        jump_model = IOM.get_jump_model(container)
        var = IOM.add_variable_container!(
            container, TestVariableType, MockThermalGen, ["A"], 1:2)
        var["A", 1] = JuMP.@variable(jump_model)
        IOM.reset_optimization_model!(container)
        @test isempty(IOM.get_variables(container))
        @test JuMP.num_variables(jump_model) == 0
        key = IOM.ParameterKey(TestParameterType, MockThermalGen)
        @test IOM.get_parameter_array(container, key)["A", 2] == 7.0
    end

    @testset "leaves parameter and initial condition entries unchanged" begin
        container = _rebuild_container()
        jump_model = IOM.get_jump_model(container)
        param_container = _add_shared_parameter!(container, TestParameterType, ["A"], 1:2)
        refs = [IOM.add_jump_parameter(jump_model, 3.0 * t) for t in 1:2]
        for t in 1:2
            IOM.get_parameter_array(param_container)["A", t] = refs[t]
        end
        device = make_mock_thermal("A")
        ics = IOM.add_initial_condition_container!(
            container, MockInitialCondition(), MockThermalGen, ["A"])
        ic_ref = IOM.add_jump_parameter(jump_model, 4.0)
        ics[1] = IOM.InitialCondition(MockInitialCondition, device, ic_ref)

        IOM.reset_optimization_model!(container)

        key = IOM.ParameterKey(TestParameterType, MockThermalGen)
        @test IOM.get_parameter(container, key) === param_container
        @test IOM.get_parameter_array(container, key)["A", 2] === refs[2]
        ic_key = IOM.InitialConditionKey(MockInitialCondition, MockThermalGen)
        @test only(IOM.get_initial_condition(container, ic_key)) === ics[1]
        @test IOM.get_value(ics[1]) === ic_ref
    end

    @testset "is documented" begin
        @test Base.Docs.hasdoc(IOM, :reset_optimization_model!)
    end
end

struct RebuildTestTemplate <: IOM.AbstractProblemTemplate end
struct RebuildTestProblem <: IOM.AbstractOptimizationProblem end

mutable struct RebuildTestModel <: IOM.AbstractOptimizationModel
    name::Symbol
    template::RebuildTestTemplate
    sys::MockSystem
    internal::IOM.ModelInternal
    simulation_info::Union{Nothing, IOM.SimulationInfo}
    store::IOM.DecisionModelStore
    ext::Dict{String, Any}
end

const REBUILD_INITIAL_TIMES = DateTime[]

function IOM.build_problem!(
    container::IOM.OptimizationContainer,
    ::RebuildTestTemplate,
    ::MockSystem,
)
    push!(REBUILD_INITIAL_TIMES, IOM.get_initial_time(IOM.get_settings(container)))
    @test isempty(IOM.get_variables(container))
    var = IOM.add_variable_container!(
        container, TestVariableType, MockThermalGen, ["A"],
        IOM.get_time_steps(container))
    var["A", 1] = JuMP.@variable(IOM.get_jump_model(container))
    return
end

@testset "rebuild_model!" begin
    empty!(REBUILD_INITIAL_TIMES)
    container = _rebuild_container(; rebuild_model = true)
    model = RebuildTestModel(
        :rebuild_test, RebuildTestTemplate(), MockSystem(100.0),
        IOM.ModelInternal(container), nothing, IOM.DecisionModelStore(),
        Dict{String, Any}())
    IOM.build_problem!(container, RebuildTestTemplate(), MockSystem(100.0))
    @test JuMP.num_variables(IOM.get_jump_model(container)) == 1
    original = IOM.get_initial_time(model)
    step_time = original + Dates.Hour(2)
    IOM.rebuild_model!(model; initial_time = step_time)
    @test REBUILD_INITIAL_TIMES == [original, step_time]
    @test IOM.get_initial_time(model) == original
    @test JuMP.num_variables(IOM.get_jump_model(container)) == 1
    @test Base.Docs.hasdoc(IOM, :rebuild_model!)
    @test Base.Docs.hasdoc(IOM, :build_problem!)
end

struct RefreshTestModel <: IOM.AbstractOptimizationModel end

@testset "update_model_parameters! defaults to a no-op" begin
    container = _rebuild_container()
    @test IOM.update_model_parameters!(container, RefreshTestModel(), nothing) === nothing
    @test Base.Docs.hasdoc(IOM, :update_model_parameters!)
    for name in (
        :requires_rebuild, :should_add_parameter, :on_parameter_reused!,
        :rebuild_model!, :update_model_parameters!, :reset_optimization_model!,
    )
        @test Base.isexported(IOM, name)
    end
end
