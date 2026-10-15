"""
Investment-model mocks: formulation and entry types, a problem type, and a minimal
investment template whose construction-time hooks record that they ran.
"""

# Investment formulations (concrete leaves of the abstract hierarchy).
struct MockInvestTechForm <: IOM.InvestmentTechnologyFormulation end
struct MockOpsTechForm <: IOM.OperationsTechnologyFormulation end
struct MockFeasTechForm <: IOM.FeasibilityTechnologyFormulation end
struct MockReqForm <: IOM.RequirementFormulation end
struct MockTransportAgg <: IOM.AbstractTransportAggregation end

# Operation/investment entry types for store classification and sizing.
struct MockStoreInvestVar <: IOM.InvestmentVariableType end
struct MockStoreOpsVar <: IOM.OperationsVariableType end
struct MockStoreInvestExpr <: IOM.InvestmentExpressionType end
struct MockStoreOpsExpr <: IOM.OperationsExpressionType end

struct MockInvestProblem <: IOM.AbstractOptimizationProblem end

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
    finalized::Bool
end

function build_mock_invest_template()
    return MockInvestTemplate(
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
        IOM.TransportModel(MockTransportAgg),
        MockCapitalModel([(Date(2030, 1, 1), Date(2030, 12, 31))]),
        MockOperationModel([[DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1)]], [1.0]),
        MockFeasibilityModel(Vector{Vector{DateTime}}()),
        false,
    )
end

# Construction-time extension points; they record that they ran so the tests can check
# the constructor calls them, and on which template.
function IOM.finalize_template!(template::MockInvestTemplate, ::MockSystem)
    template.finalized = true
    return
end

function IOM.validate_time_series!(model::IOM.InvestmentModel{MockInvestProblem})
    model.ext["time_series_validated"] = true
    return
end
