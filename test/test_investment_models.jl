"""
Unit tests for the investment formulation hierarchy and the generic time-mapping accessors
on OptimizationContainer.
Uses mocks (mock_components.jl, mock_system.jl, mock_investments.jl) — no PowerSystems
dependency.
"""

@testset "Investment formulation hierarchy" begin
    @test MockInvestTechForm <: IOM.AbstractTechnologyFormulation
    @test MockOpsTechForm <: IOM.AbstractTechnologyFormulation
    @test MockFeasTechForm <: IOM.AbstractTechnologyFormulation
    @test IOM.AbstractTechnologyFormulation <: IOM.AbstractDeviceFormulation
    @test IOM.InvestmentTechnologyFormulation <: IOM.AbstractTechnologyFormulation
    @test IOM.OperationsTechnologyFormulation <: IOM.AbstractTechnologyFormulation
    @test IOM.FeasibilityTechnologyFormulation <: IOM.AbstractTechnologyFormulation
    @test IOM.RequirementFormulation <: IOM.AbstractServiceFormulation
end

@testset "OptimizationContainer time-mapping accessors" begin
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)

    # Default is unset (`nothing`) until a mapping is attached.
    @test IOM.get_time_mapping(container) === nothing

    tmap = build_mock_invest_time_mapping()
    IOM.set_time_mapping!(container, tmap)
    @test IOM.get_time_mapping(container) === tmap
    @test IOM.get_period_count(IOM.get_time_mapping(container)) == 2
end
