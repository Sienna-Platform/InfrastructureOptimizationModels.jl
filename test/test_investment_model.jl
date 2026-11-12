"""
Unit tests for a `DecisionModel` over an investment-style problem (static time series, no
IOM network model) and the investment `init_optimization_container!` build hook. Uses
mocks (mock_investments.jl) + HiGHS — no PowerSystems dependency.
"""

@testset "DecisionModel over an investment problem: construction and accessors" begin
    t = build_mock_invest_template()
    portfolio = MockSystem(100.0)
    settings = IOM.Settings(
        portfolio;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    model = IOM.DecisionModel{MockInvestProblem}(t, portfolio, settings; name = :cem)

    @test IOM.get_name(model) == :cem
    # The problem type's prepare_time_series! picks the container's time series type.
    container = IOM.get_optimization_container(model)
    @test IOM.get_default_time_series_type(container) === IS.SingleTimeSeries
    # The model builds against a finalized copy; the caller's template is left untouched.
    @test IOM.get_template(model) !== t
    @test IOM.get_template(model) isa MockInvestTemplate
    @test IOM.get_template(model).finalized
    @test !t.finalized
    @test model.ext["time_series_validated"]
    @test IOM.get_system(model) === portfolio
    @test IOM.get_settings(model) === settings
    @test !IOM.is_built(model)
    @test isempty(model)
    @test IOM.get_status(model) == IOM.ModelBuildStatus.EMPTY
    @test IOM.get_jump_model(model) isa JuMP.Model
    @test container isa IOM.OptimizationContainer
    @test IOM.get_store(model) isa IOM.DecisionModelStore
    @test IOM.get_portfolio_to_file(settings) == IOM.get_system_to_file(settings)

    named = IOM.DecisionModel{MockInvestProblem}(t, portfolio, settings; name = "my_cem")
    @test IOM.get_name(named) == :my_cem
    # Without a name, the settings-taking constructor uses the problem type's name.
    unnamed = IOM.DecisionModel{MockInvestProblem}(t, portfolio, settings)
    @test IOM.get_name(unnamed) == :MockInvestProblem

    # Keyword and type-first constructors; the name defaults to the problem type.
    kw_model = IOM.DecisionModel{MockInvestProblem}(t, portfolio; name = :kw)
    @test IOM.get_name(kw_model) == :kw
    typed = IOM.DecisionModel(MockInvestProblem, t, portfolio)
    @test typed isa IOM.DecisionModel{MockInvestProblem}
    @test IOM.get_name(typed) == :MockInvestProblem

    # Keyword settings are forwarded.
    exported = IOM.DecisionModel{MockInvestProblem}(
        t,
        portfolio;
        warm_start = false,
        export_optimization_model = IOM.OptimizationModelExportFormat.LP,
    )
    @test !IOM.get_warm_start(IOM.get_settings(exported))
    @test IOM.get_export_optimization_model(IOM.get_settings(exported)) ==
          IOM.OptimizationModelExportFormat.LP
end

@testset "init_optimization_container! leaves the time mapping to the caller" begin
    t = build_mock_invest_template()
    portfolio = MockSystem(100.0)
    settings = IOM.Settings(
        portfolio;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    model = IOM.DecisionModel{MockInvestProblem}(t, portfolio, settings; name = :cem)
    container = IOM.get_optimization_container(model)

    IOM.init_optimization_container!(container, t, portfolio)
    @test IOM.get_time_mapping(container) === nothing

    tmap = build_mock_invest_time_mapping()
    IOM.set_time_mapping!(container, tmap)
    @test IOM.get_time_mapping(container) === tmap
end
