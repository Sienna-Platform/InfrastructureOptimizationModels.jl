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
    is_from_evaluator

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
