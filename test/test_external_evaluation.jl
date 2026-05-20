import InfrastructureOptimizationModels:
    AbstractEvaluator,
    AbstractEvaluationData,
    EvaluationContainer,
    get_evaluators,
    get_evaluation_data,
    get_evaluator,
    add_evaluator!,
    add_evaluation_data!,
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
    @test isempty(get_evaluation_data(ec))

    ev = MockEvaluator(:primary)
    add_evaluator!(ec, MockEvaluator, ev)
    @test !isempty(ec)
    @test length(ec) == 1
    @test haskey(ec, MockEvaluator)
    @test get_evaluator(ec, MockEvaluator) === ev

    data = MockEvaluationData(ev, false, Float64[])
    add_evaluation_data!(ec, MockEvaluator, data)
    @test get_evaluation_data(ec, MockEvaluator) === data
    @test length(get_evaluation_data(ec)) == 1
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
    ec = EvaluationContainer()
    ev = MockEvaluator(:e1)
    add_evaluator!(ec, MockEvaluator, ev)
    data = initialize_evaluation_data(ev, nothing, nothing)
    add_evaluation_data!(ec, MockEvaluator, data)

    @test !is_solved(data)
    evaluate!(data, nothing, nothing)
    @test is_solved(data)
    @test get_inner_data(data) == [1.0]

    # Iterate via the dict view, as `calculate_aux_variables!` does
    for (T, d) in get_evaluation_data(ec)
        reset!(d)
    end
    @test !is_solved(data)

    evaluate!(data, nothing, nothing)
    @test is_solved(data)
    @test get_inner_data(data) == [1.0, 2.0]
end

@testset "is_from_evaluator default is false" begin
    @test is_from_evaluator(MockAuxVariable) === false
end

@testset "NetworkModel evaluations field defaults to empty" begin
    nw = IOM.NetworkModel(TestPowerModel)
    @test IOM.get_evaluations(nw) isa EvaluationContainer
    @test isempty(IOM.get_evaluations(nw))
end

@testset "NetworkModel evaluations accepts a populated container" begin
    ec = EvaluationContainer()
    add_evaluator!(ec, MockEvaluator, MockEvaluator(:populated))
    nw = IOM.NetworkModel(TestPowerModel; evaluations = ec)
    @test IOM.get_evaluations(nw) === ec
    @test haskey(IOM.get_evaluations(nw), MockEvaluator)
end
