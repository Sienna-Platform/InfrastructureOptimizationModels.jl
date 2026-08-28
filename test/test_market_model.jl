"""
Unit tests for the neutral MarketModel container: the formulation type, the
component models keyed like device models, and the settlement domain the
market's equalities are keyed on.
"""

struct _TestMarketFormulation <: IOM.AbstractMarketModel end

@testset "MarketModel container" begin
    m = IOM.MarketModel(
        _TestMarketFormulation;
        settlement_domain = IS.InfrastructureSystemsComponent,
    )
    @test IOM.get_settlement_domain(m) === IS.InfrastructureSystemsComponent
    @test isempty(IOM.get_market_component_models(m))
    @test isempty(IOM.get_duals(m))
    @test IOM.get_subsystem(m) === nothing
    @test_throws ArgumentError IOM.MarketModel(
        IOM.AbstractMarketModel;
        settlement_domain = IS.InfrastructureSystemsComponent,
    )
end

@testset "MarketModel with a container-typed settlement domain" begin
    m = IOM.MarketModel(_TestMarketFormulation; settlement_domain = IS.SystemData)
    @test IOM.get_settlement_domain(m) === IS.SystemData
    @test IS.SystemData <: IS.InfrastructureSystemsContainer
end
