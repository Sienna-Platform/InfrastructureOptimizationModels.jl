get_technology_models(template::AbstractProblemTemplate) = template.technology_models
get_branch_models(template::AbstractProblemTemplate) = template.branch_models
get_requirement_models(template::AbstractProblemTemplate) = template.requirement_models
get_transport_model(template::AbstractProblemTemplate) = template.transport_model

get_capital_model(template::AbstractProblemTemplate) = template.capital_model
get_operation_model(template::AbstractProblemTemplate) = template.operation_model
get_feasibility_model(template::AbstractProblemTemplate) = template.feasibility_model

"""
Investment-specific data stored in OptimizationContainer's `ext` dictionary.
Contains time mapping, financial parameters, and operational weights.
"""

# TODO: Move financial parameters out of the container, they can stay in the portfolio
mutable struct InvestmentContainerData
    time_mapping::TimeMapping
    operational_weights::Union{Nothing, Vector{Float64}}
    base_year::Int
    discount_rate::Float64
    inflation_rate::Float64
    interest_rate::Float64
end

function InvestmentContainerData()
    return InvestmentContainerData(
        TimeMapping(nothing),
        nothing,
        2020,
        0.0,
        0.0,
        0.0,
    )
end
