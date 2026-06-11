"""
Unit tests for NetworkModel with the IS network-matrix abstractions. IOM holds
matrices, reduction data, and the branch-reduction tracker behind abstract types
with no PowerNetworkMatrices dependency; these mocks stand in for the
implementing package (POM/PNM).
"""

struct MockNetworkMatrix <: IOM.AbstractInfrastructureNetworkMatrix{Float64}
    data::Matrix{Float64}
end
Base.size(m::MockNetworkMatrix) = size(m.data)
Base.getindex(m::MockNetworkMatrix, i::Int, j::Int) = m.data[i, j]

struct MockReductionData <: IOM.AbstractInfrastructureNetworkReductionData end

struct MockReductionTracker <: IOM.AbstractBranchReductionTracker end

@testset "NetworkModel with abstract network matrices" begin
    matrix = MockNetworkMatrix([1.0 0.0; 0.0 1.0])
    nw = IOM.NetworkModel(TestPowerModel; PTDF_matrix = matrix, MODF_matrix = matrix)
    @test IOM.get_PTDF_matrix(nw) === matrix
    @test IOM.get_MODF_matrix(nw) === matrix
    # Reduction data and tracker are populated by the matrix-aware downstream package.
    @test IOM.get_network_reduction(nw) === nothing
    @test IOM.get_reduced_branch_tracker(nw) === nothing

    nw.network_reduction = MockReductionData()
    @test IOM.get_network_reduction(nw) isa MockReductionData

    tracker = MockReductionTracker()
    IOM.set_reduced_branch_tracker!(nw, tracker)
    @test IOM.get_reduced_branch_tracker(nw) === tracker
end
