"""
Unit tests for InvestmentModel (investments/investment_model.jl) and the investment
`init_optimization_container!` build hook, plus the investment-template accessors now
hosted in operation/problem_template.jl. Uses mocks (mock_investments.jl) + HiGHS — no
PowerSystems dependency.
"""

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
    model = IOM.InvestmentModel{MockInvestProblem}(t, portfolio, settings; name = :cem)

    @test IOM.get_name(model) == :cem
    # The model builds against a finalized copy; the caller's template is left untouched.
    @test IOM.get_template(model) !== t
    @test IOM.get_template(model) isa MockInvestTemplate
    @test IOM.get_template(model).finalized
    @test !t.finalized
    @test model.ext["time_series_validated"]
    @test IOM.get_portfolio(model) === portfolio
    @test IOM.get_system(model) === portfolio
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

    named = IOM.InvestmentModel{MockInvestProblem}(t, portfolio, settings; name = "my_cem")
    @test IOM.get_name(named) == :my_cem
    # Without a name, the settings-taking constructor uses the problem type's name.
    unnamed = IOM.InvestmentModel{MockInvestProblem}(t, portfolio, settings)
    @test IOM.get_name(unnamed) == :MockInvestProblem

    # Keyword and type-first constructors; the name defaults to the problem type.
    kw_model = IOM.InvestmentModel{MockInvestProblem}(t, portfolio; name = :kw)
    @test IOM.get_name(kw_model) == :kw
    typed = IOM.InvestmentModel(MockInvestProblem, t, portfolio)
    @test typed isa IOM.InvestmentModel{MockInvestProblem}
    @test IOM.get_name(typed) == :MockInvestProblem

    # Settings that apply to investment models are forwarded; operational-only ones are
    # not accepted.
    exported = IOM.InvestmentModel{MockInvestProblem}(
        t,
        portfolio;
        warm_start = false,
        export_optimization_model = IOM.OptimizationModelExportFormat.LP,
    )
    @test !IOM.get_warm_start(IOM.get_settings(exported))
    @test IOM.get_export_optimization_model(IOM.get_settings(exported)) ==
          IOM.OptimizationModelExportFormat.LP
    @test_throws MethodError IOM.InvestmentModel{MockInvestProblem}(
        t,
        portfolio;
        rebuild_model = true,
    )
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
    model = IOM.InvestmentModel{MockInvestProblem}(t, portfolio, settings; name = :cem)
    container = IOM.get_optimization_container(model)

    @test IOM.get_time_mapping(container) === nothing
    IOM.init_optimization_container!(container, t, portfolio)

    tmap = IOM.get_time_mapping(container)
    @test tmap isa IOM.TimeMapping
    @test IOM.get_total_investment_period_count(tmap) == 1
    @test IOM.get_base_date(tmap) == Date(2030, 1, 1)
    @test IOM.get_total_operation_period_count(tmap) == 2
    # Operational weights from operation_model.series_weights flow onto the TimeMapping,
    # reachable both from the mapping and via the container convenience accessor.
    @test IOM.get_operational_weights(tmap) == [1.0]
    @test IOM.get_operational_weights(container) == [1.0]
end
