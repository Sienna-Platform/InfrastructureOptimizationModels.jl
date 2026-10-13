"""
Unit tests for NetworkModel's neutral network anchors. IOM holds the network source,
the derived network data, and the branch-reduction tracker behind abstract types with
no PowerNetworkMatrices dependency; these mocks stand in for the implementing package
(POM/PNM).
"""

struct MockNetworkSource <: IOM.AbstractNetworkSource end

struct MockNetworkData <: IOM.AbstractNetworkData end

struct MockReductionTracker <: IOM.AbstractBranchReductionTracker end

@testset "NetworkModel with abstract network source and data" begin
    source = MockNetworkSource()
    selector = IS.make_selector(IS.TestComponent, "component_1")
    nw = IOM.NetworkModel(
        TestPowerModel;
        network_source = source,
        irreducible_buses = selector,
    )
    @test IOM.get_network_source(nw) === source
    @test IOM.get_irreducible_buses(nw) == selector
    # Network data and tracker are populated by the matrix-aware downstream package.
    @test IOM.get_network_data(nw) === nothing
    @test IOM.get_reduced_branch_tracker(nw) === nothing

    data = MockNetworkData()
    IOM.set_network_data!(nw, data)
    @test IOM.get_network_data(nw) === data

    tracker = MockReductionTracker()
    IOM.set_reduced_branch_tracker!(nw, tracker)
    @test IOM.get_reduced_branch_tracker(nw) === tracker
end

@testset "NetworkModel defaults to the neutral source" begin
    nw = IOM.NetworkModel(TestPowerModel)
    @test IOM.get_network_source(nw) === IOM.DefaultNetworkSource()
    @test IOM.get_irreducible_buses(nw) == IS.make_selector()
    @test isempty(IOM.get_subnetworks(nw))
end
