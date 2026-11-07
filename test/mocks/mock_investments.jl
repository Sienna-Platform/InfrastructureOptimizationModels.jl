"""
Investment-model mocks: formulation and entry types, a problem type, and a minimal
investment template whose construction-time hooks record that they ran.
"""

# Investment formulations (concrete leaves of the abstract hierarchy).
struct MockInvestTechForm <: IOM.InvestmentTechnologyFormulation end
struct MockOpsTechForm <: IOM.OperationsTechnologyFormulation end
struct MockFeasTechForm <: IOM.FeasibilityTechnologyFormulation end

# Operation/investment entry types for store classification and sizing.
struct MockStoreInvestVar <: IOM.InvestmentVariableType end
struct MockStoreOpsVar <: IOM.OperationsVariableType end
struct MockStoreInvestExpr <: IOM.InvestmentExpressionType end
struct MockStoreOpsExpr <: IOM.OperationsExpressionType end

struct MockInvestProblem <: IOM.AbstractOptimizationProblem end

# The first `operation_step_count` time steps are operational
struct MockInvestTimeMapping <: IOM.AbstractTimeMapping
    base_date::Date
    timestamps::Vector{DateTime}
    block_ranges::Vector{UnitRange{Int}}
    block_period_index::Vector{Int}
    period_count::Int
    operation_step_count::Int
end

IOM.get_period_count(tm::MockInvestTimeMapping) = tm.period_count
IOM.get_block_ranges(tm::MockInvestTimeMapping) = tm.block_ranges
IOM.get_block_period_index(tm::MockInvestTimeMapping) = tm.block_period_index
IOM.get_timestamps(tm::MockInvestTimeMapping) = tm.timestamps
IOM.get_base_date(tm::MockInvestTimeMapping) = tm.base_date
IOM.get_total_operation_period_count(tm::MockInvestTimeMapping) = tm.operation_step_count

function build_mock_invest_time_mapping()
    timestamps = [
        DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1),
        DateTime(2030, 7, 1, 0), DateTime(2030, 7, 1, 1),
        DateTime(2040, 1, 1, 0), DateTime(2040, 1, 1, 1),
        DateTime(2040, 7, 1, 0), DateTime(2040, 7, 1, 1),
    ]
    return MockInvestTimeMapping(
        Date(2030, 1, 1),
        timestamps,
        [1:2, 3:4, 5:6, 7:8],
        [1, 1, 2, 2],
        2,
        8,
    )
end

function build_empty_mock_invest_time_mapping()
    return MockInvestTimeMapping(
        Date(2030, 1, 1),
        DateTime[],
        UnitRange{Int}[],
        Int[],
        0,
        0,
    )
end

mutable struct MockInvestTemplate <: IOM.AbstractProblemTemplate
    finalized::Bool
end

build_mock_invest_template() = MockInvestTemplate(false)

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
