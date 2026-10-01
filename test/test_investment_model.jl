"""
Unit tests for InvestmentModel (investments/investment_model.jl) and the investment
`init_optimization_container!` build hook, plus the investment-template accessors now
hosted in operation/problem_template.jl. Uses mocks + HiGHS — no PowerSystems dependency.
"""

using Test
using InfrastructureOptimizationModels
using Dates
using JuMP
using InfrastructureSystems
const IS = InfrastructureSystems

struct MockInvestProblem <: IOM.AbstractOptimizationProblem end
struct MockTransportAggModel <: IOM.AbstractTransportAggregation end

# Minimal sub-models carrying exactly the fields the investment build reads.
struct MockCapitalModel
    investment_years::Vector{Tuple{Date, Date}}
end
struct MockOperationModel
    representative_series::Vector{Vector{DateTime}}
    series_weights::Vector{Float64}
end
struct MockFeasibilityModel
    sample_periods::Vector{Vector{DateTime}}
end

mutable struct MockInvestTemplate <: IOM.AbstractProblemTemplate
    technology_models::Dict{Symbol, Any}
    branch_models::Dict{Symbol, Any}
    requirement_models::Dict{Symbol, Any}
    transport_model::IOM.TransportModel
    capital_model::MockCapitalModel
    operation_model::MockOperationModel
    feasibility_model::MockFeasibilityModel
end

function build_mock_invest_template()
    return MockInvestTemplate(
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
        IOM.TransportModel(MockTransportAggModel),
        MockCapitalModel([(Date(2030, 1, 1), Date(2030, 12, 31))]),
        MockOperationModel([[DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1)]], [1.0]),
        MockFeasibilityModel(Vector{Vector{DateTime}}()),
    )
end

@testset "Investment template accessors" begin
    t = build_mock_invest_template()
    @test IOM.get_capital_model(t) === t.capital_model
    @test IOM.get_operation_model(t) === t.operation_model
    @test IOM.get_feasibility_model(t) === t.feasibility_model
    @test IOM.get_transport_model(t) === t.transport_model
    @test IOM.get_technology_models(t) === t.technology_models
    @test IOM.get_requirement_models(t) === t.requirement_models
end

@testset "InvestmentModel construction and accessors" begin
    t = build_mock_invest_template()
    portfolio = MockSystem(100.0)
    settings = IOM.Settings(
        portfolio;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    model = IOM.InvestmentModel{MockInvestProblem}(
        t,
        MockInvestProblem,
        portfolio,
        settings,
        nothing,
    )

    @test IOM.get_name(model) == :CEM
    @test IOM.get_template(model) === t
    @test IOM.get_portfolio(model) === portfolio
    @test IOM.get_settings(model) === settings
    # Portfolio models are always in natural units.
    @test IOM.get_problem_base_power(model) == 1.0
    @test !IOM.is_built(model)
    @test isempty(model)
    @test IOM.get_status(model) == IOM.ModelBuildStatus.EMPTY
    @test IOM.get_jump_model(model) isa JuMP.Model
    @test IOM.get_optimization_container(model) isa IOM.OptimizationContainer
    @test IOM.get_store(model) isa IOM.InvestmentModelStore
    @test IOM.get_portfolio_to_file(settings) == IOM.get_system_to_file(settings)
end

@testset "init_optimization_container! builds the time mapping" begin
    t = build_mock_invest_template()
    portfolio = MockSystem(100.0)
    settings = IOM.Settings(
        portfolio;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    model = IOM.InvestmentModel{MockInvestProblem}(
        t,
        MockInvestProblem,
        portfolio,
        settings,
        nothing,
    )
    container = IOM.get_optimization_container(model)

    @test IOM.get_time_mapping(container) === nothing
    IOM.init_optimization_container!(container, t, portfolio)

    tmap = IOM.get_time_mapping(container)
    @test tmap isa IOM.TimeMapping
    @test IOM.get_total_investment_period_count(tmap) == 1
    @test IOM.get_base_date(tmap) == Date(2030, 1, 1)
    @test IOM.get_total_operation_period_count(tmap) == 2
end
