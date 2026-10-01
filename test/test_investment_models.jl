"""
Unit tests for the investment model-container types and template/container accessors
ported into IOM: TechnologyModel, RequirementModel, TransportModel, the investment
formulation hierarchy, and the generic time-mapping accessors on OptimizationContainer.
Uses mocks (mock_components.jl, mock_system.jl) — no PowerSystems dependency.
"""

using Test
using InfrastructureOptimizationModels
using Dates
using InfrastructureSystems
const IS = InfrastructureSystems

# Mock investment formulations (concrete leaves of the ported abstract hierarchy).
struct MockInvestTechForm <: IOM.InvestmentTechnologyFormulation end
struct MockOpsTechForm <: IOM.OperationsTechnologyFormulation end
struct MockFeasTechForm <: IOM.FeasibilityTechnologyFormulation end
struct MockReqForm <: IOM.RequirementFormulation end
struct MockTransportAgg <: IOM.AbstractTransportAggregation end

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

@testset "TechnologyModel" begin
    @testset "construction and accessors" begin
        model = IOM.TechnologyModel(
            MockComponentType,
            MockInvestTechForm,
            MockOpsTechForm,
            MockFeasTechForm,
        )
        @test IOM.get_technology_type(model) == MockComponentType
        @test IOM.get_investment_formulation(model) == MockInvestTechForm
        @test IOM.get_operations_formulation(model) == MockOpsTechForm
        @test IOM.get_feasibility_formulation(model) == MockFeasTechForm
        @test model.use_slacks == false
        @test model.attributes isa Dict{String, Any}
    end

    @testset "custom attributes merge with defaults" begin
        model = IOM.TechnologyModel(
            MockComponentType,
            MockInvestTechForm,
            MockOpsTechForm,
            MockFeasTechForm;
            use_slacks = true,
            attributes = Dict{String, Any}("k" => 42),
        )
        @test model.use_slacks == true
        @test model.attributes["k"] == 42
    end

    @testset "_set_model! stores and warns on overwrite" begin
        dict = Dict{Any, Vector{String}}()
        model = IOM.TechnologyModel(
            MockComponentType,
            MockInvestTechForm,
            MockOpsTechForm,
            MockFeasTechForm,
        )
        IOM._set_model!(dict, ["a", "b"], model)
        @test dict[model] == ["a", "b"]
        @test_logs (:warn, r"Overwriting") IOM._set_model!(dict, ["c"], model)
    end
end

@testset "RequirementModel" begin
    @testset "construction and accessors" begin
        model = IOM.RequirementModel(MockComponentType, MockReqForm)
        @test IOM.get_requirement_type(model) == MockComponentType
        @test IOM.get_requirement_formulation(model) == MockReqForm
        @test IOM.get_use_slacks(model) == false
        @test isempty(IOM.get_duals(model))
        @test IOM.get_attributes(model) isa Dict{String, Any}
    end

    @testset "custom attributes merge with defaults" begin
        model = IOM.RequirementModel(
            MockComponentType,
            MockReqForm;
            use_slacks = true,
            attributes = Dict{String, Any}("x" => 7),
        )
        @test IOM.get_use_slacks(model) == true
        @test IOM.get_attributes(model)["x"] == 7
    end

    @testset "_set_model! stores and warns on overwrite" begin
        dict = Dict{Any, Vector{String}}()
        model = IOM.RequirementModel(MockComponentType, MockReqForm)
        IOM._set_model!(dict, ["r"], model)
        @test dict[model] == ["r"]
        @test_logs (:warn, r"Overwriting.*requirement") IOM._set_model!(dict, ["s"], model)
    end
end

@testset "TransportModel" begin
    @test IOM.get_use_slacks(IOM.TransportModel(MockTransportAgg)) == false
    @test IOM.get_use_slacks(IOM.TransportModel(MockTransportAgg; use_slacks = true)) ==
          true
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

    tmap = IOM.TimeMapping(
        [(Date(2030, 1, 1), Date(2030, 12, 31))],
        [[DateTime(2030, 1, 1, 0)]],
        Vector{Vector{DateTime}}(),
    )
    IOM.set_time_mapping!(container, tmap)
    @test IOM.get_time_mapping(container) === tmap
    @test IOM.get_total_investment_period_count(IOM.get_time_mapping(container)) == 1
end
