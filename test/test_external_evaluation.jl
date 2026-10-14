import InfrastructureOptimizationModels:
    AbstractEvaluator,
    AbstractEvaluationData,
    EvaluationContainer,
    get_evaluators,
    get_evaluation_data,
    get_evaluator,
    add_evaluator!,
    add_evaluation_data!,
    reset_evaluations!,
    initialize_evaluation_data,
    evaluate!,
    reset!,
    is_solved,
    get_inner_data,
    is_from_evaluator,
    bind_aux_variable!,
    get_bound_aux_var_keys,
    add_evaluator_aux_variables!

# Concrete-but-minimal subtypes for interface-stub coverage. No methods registered.
struct DummyEvaluator <: AbstractEvaluator end
struct DummyEvaluationData <: AbstractEvaluationData end

# Full mock implementation used to exercise the dispatch path end-to-end.
struct MockEvaluator <: AbstractEvaluator
    tag::Symbol
end

mutable struct MockEvaluationData <: AbstractEvaluationData
    evaluator::MockEvaluator
    solved::Bool
    payload::Vector{Float64}
end

IOM.initialize_evaluation_data(ev::MockEvaluator, _container, _system) =
    MockEvaluationData(ev, false, Float64[])

function IOM.evaluate!(data::MockEvaluationData, _container, _system)
    push!(data.payload, length(data.payload) + 1.0)
    data.solved = true
    return
end

IOM.reset!(data::MockEvaluationData) = (data.solved = false; return)
IOM.is_solved(data::MockEvaluationData) = data.solved
IOM.get_inner_data(data::MockEvaluationData) = data.payload

@testset "EvaluationContainer CRUD" begin
    ec = EvaluationContainer()
    @test ec isa EvaluationContainer
    @test isempty(ec)
    @test length(ec) == 0
    @test !haskey(ec, MockEvaluator)
    @test isempty(get_evaluators(ec))
    @test !hasfield(EvaluationContainer, :evaluation_data)

    ev = MockEvaluator(:primary)
    add_evaluator!(ec, MockEvaluator, ev)
    @test !isempty(ec)
    @test length(ec) == 1
    @test haskey(ec, MockEvaluator)
    @test get_evaluator(ec, MockEvaluator) === ev
end

function _mock_evaluation_container()
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    return IOM.OptimizationContainer(mock_sys, settings, JuMP.Model(), MockDeterministic)
end

struct OrderProbeEvaluatorA <: AbstractEvaluator end
struct OrderProbeEvaluatorB <: AbstractEvaluator end
struct OrderProbeEvaluatorC <: AbstractEvaluator end
struct OrderProbeData <: AbstractEvaluationData end

@testset "Evaluators keep registration order" begin
    types = [OrderProbeEvaluatorA, OrderProbeEvaluatorB, OrderProbeEvaluatorC]
    for order in (types, reverse(types), types[[2, 3, 1]])
        ec = EvaluationContainer()
        container = _mock_evaluation_container()
        for T in order
            add_evaluator!(ec, T, T())
            add_evaluation_data!(container, T, OrderProbeData())
        end
        @test collect(keys(get_evaluators(ec))) == order
        @test collect(keys(get_evaluation_data(container))) == order
        @test first.(collect(get_evaluation_data(container))) == order
    end
end

@testset "AbstractEvaluator interface stubs throw with type names" begin
    container = nothing
    system = nothing
    dummy_ev = DummyEvaluator()
    dummy_data = DummyEvaluationData()

    err1 = @test_throws ErrorException initialize_evaluation_data(
        dummy_ev,
        container,
        system,
    )
    @test occursin("DummyEvaluator", err1.value.msg)

    err2 = @test_throws ErrorException evaluate!(dummy_data, container, system)
    @test occursin("DummyEvaluationData", err2.value.msg)

    err3 = @test_throws ErrorException reset!(dummy_data)
    @test occursin("DummyEvaluationData", err3.value.msg)

    err4 = @test_throws ErrorException is_solved(dummy_data)
    @test occursin("DummyEvaluationData", err4.value.msg)

    err5 = @test_throws ErrorException get_inner_data(dummy_data)
    @test occursin("DummyEvaluationData", err5.value.msg)
end

@testset "MockEvaluator end-to-end dispatch" begin
    container = _mock_evaluation_container()
    ev = MockEvaluator(:e1)
    data = initialize_evaluation_data(ev, nothing, nothing)
    add_evaluation_data!(container, MockEvaluator, data)
    @test get_evaluation_data(container, MockEvaluator) === data
    @test length(get_evaluation_data(container)) == 1

    @test !is_solved(data)
    evaluate!(data, nothing, nothing)
    @test is_solved(data)
    @test get_inner_data(data) == [1.0]

    reset_evaluations!(container)
    @test !is_solved(data)

    evaluate!(data, nothing, nothing)
    @test is_solved(data)
    @test get_inner_data(data) == [1.0, 2.0]
end

@testset "is_from_evaluator default is false" begin
    @test is_from_evaluator(MockAuxVariable) === false
end

@testset "NetworkModel holds no evaluators" begin
    nw = IOM.NetworkModel(TestPowerModel)
    @test !hasfield(IOM.NetworkModel, :evaluations)
    @test isempty(IOM.get_modeled_branch_types(nw))
    @test_throws MethodError IOM.NetworkModel(
        TestPowerModel;
        evaluations = EvaluationContainer(),
    )
end

@testset "reset_optimization_model! empties the evaluation data" begin
    container = _mock_evaluation_container()
    @test isempty(get_evaluation_data(container))
    ev = MockEvaluator(:reset)
    add_evaluation_data!(container, MockEvaluator, MockEvaluationData(ev, true, [1.0]))
    @test !isempty(get_evaluation_data(container))

    IOM.reset_optimization_model!(container)
    @test isempty(get_evaluation_data(container))
end

struct BoundAuxA <: IOM.AuxVariableType end
struct BoundAuxB <: IOM.AuxVariableType end
struct UnboundEvaluatorAux <: IOM.AuxVariableType end
IOM.is_from_evaluator(::Type{BoundAuxA}) = true
IOM.is_from_evaluator(::Type{BoundAuxB}) = true
IOM.is_from_evaluator(::Type{UnboundEvaluatorAux}) = true

struct OtherMockEvaluator <: AbstractEvaluator end
mutable struct OtherMockEvaluationData <: AbstractEvaluationData
    solved::Bool
end
function IOM.evaluate!(d::OtherMockEvaluationData, _container, _system)
    d.solved = true
    return
end
function IOM.reset!(d::OtherMockEvaluationData)
    d.solved = false
    return
end

struct BindingTestContext <: IOM.AbstractSolveContext end

const AUX_CALLS = Tuple{DataType, DataType}[]

# Any (aux type, data type) pair without a method here is a MethodError.
function IOM.calculate_aux_variable_value!(
    ::IOM.OptimizationContainer,
    ::IOM.AuxVarKey{BoundAuxA, MockComponentType},
    ::IS.InfrastructureSystemsContainer,
    data::MockEvaluationData,
)
    push!(AUX_CALLS, (BoundAuxA, typeof(data)))
    return
end
function IOM.calculate_aux_variable_value!(
    ::IOM.OptimizationContainer,
    ::IOM.AuxVarKey{BoundAuxB, MockComponentType},
    ::IS.InfrastructureSystemsContainer,
    data::OtherMockEvaluationData,
)
    push!(AUX_CALLS, (BoundAuxB, typeof(data)))
    return
end

function _binding_container()
    container = _mock_evaluation_container()
    IOM.set_time_steps!(container, 1:1)
    for aux in (BoundAuxA, BoundAuxB)
        IOM.add_aux_variable_container!(container, aux, MockComponentType, ["g1"], 1:1)
    end
    add_evaluation_data!(
        container,
        MockEvaluator,
        MockEvaluationData(MockEvaluator(:a), false, Float64[]),
    )
    add_evaluation_data!(container, OtherMockEvaluator, OtherMockEvaluationData(false))
    return container
end

@testset "Each evaluator computes only its bound aux variables" begin
    container = _binding_container()
    key_a = IOM.AuxVarKey(BoundAuxA, MockComponentType)
    key_b = IOM.AuxVarKey(BoundAuxB, MockComponentType)
    bind_aux_variable!(container, MockEvaluator, key_a)
    bind_aux_variable!(container, OtherMockEvaluator, key_b)
    bind_aux_variable!(container, OtherMockEvaluator, key_b)
    @test get_bound_aux_var_keys(container, MockEvaluator) == [key_a]
    @test get_bound_aux_var_keys(container, OtherMockEvaluator) == [key_b]
    IOM.check_optimization_container(container)

    empty!(AUX_CALLS)
    IOM.calculate_aux_variables!(container, MockSystem(100.0), BindingTestContext())
    @test AUX_CALLS ==
          [(BoundAuxA, MockEvaluationData), (BoundAuxB, OtherMockEvaluationData)]
end

@testset "bind_aux_variable! rejects invalid bindings" begin
    container = _binding_container()
    key_a = IOM.AuxVarKey(BoundAuxA, MockComponentType)
    @test_throws ArgumentError bind_aux_variable!(container, DummyEvaluator, key_a)
    @test_throws ArgumentError bind_aux_variable!(
        container,
        MockEvaluator,
        IOM.AuxVarKey(UnboundEvaluatorAux, MockComponentType),
    )
    IOM.add_aux_variable_container!(
        container,
        MockAuxVariable,
        MockComponentType,
        ["g1"],
        1:1,
    )
    @test_throws ArgumentError bind_aux_variable!(
        container,
        MockEvaluator,
        IOM.AuxVarKey(MockAuxVariable, MockComponentType),
    )
end

@testset "bind_aux_variable! rejects a second evaluator for the same key" begin
    container = _binding_container()
    key_a = IOM.AuxVarKey(BoundAuxA, MockComponentType)
    bind_aux_variable!(container, MockEvaluator, key_a)
    err = @test_throws ArgumentError bind_aux_variable!(
        container,
        OtherMockEvaluator,
        key_a,
    )
    @test occursin("MockEvaluator", err.value.msg)
    @test occursin("OtherMockEvaluator", err.value.msg)
    @test get_bound_aux_var_keys(container, MockEvaluator) == [key_a]
    @test isempty(get_bound_aux_var_keys(container, OtherMockEvaluator))
end

@testset "An unbound evaluator aux variable fails the container check" begin
    container = _binding_container()
    bind_aux_variable!(
        container,
        MockEvaluator,
        IOM.AuxVarKey(BoundAuxA, MockComponentType),
    )
    err = @test_throws ErrorException IOM.check_optimization_container(container)
    @test occursin("BoundAuxB", err.value.msg)
end

@testset "Rebuild clears the bindings" begin
    container = _binding_container()
    bind_aux_variable!(
        container,
        MockEvaluator,
        IOM.AuxVarKey(BoundAuxA, MockComponentType),
    )
    IOM.reset_optimization_model!(container)
    @test isempty(get_bound_aux_var_keys(container, MockEvaluator))
end

@testset "add_evaluator_aux_variables! defaults to a no-op" begin
    container = _binding_container()
    @test add_evaluator_aux_variables!(
        container,
        MockEvaluator,
        get_evaluation_data(container, MockEvaluator),
        MockSystem(100.0),
    ) === nothing
end

@testset "Calculating aux variables rejects an unbound evaluator aux variable" begin
    container = _binding_container()
    bind_aux_variable!(
        container,
        MockEvaluator,
        IOM.AuxVarKey(BoundAuxA, MockComponentType),
    )
    err = @test_throws ErrorException IOM.calculate_aux_variables!(
        container,
        MockSystem(100.0),
        BindingTestContext(),
    )
    @test occursin("BoundAuxB", err.value.msg)
end
