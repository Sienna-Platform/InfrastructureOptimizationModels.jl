#################################################################################
# Left-hand-side parameters: values written into constraints as fixed coefficients
#################################################################################

"""
Whether `name` has a series row in the LHS parameter container `key`. Components without one
contribute a fixed coefficient chosen by the formulation.
"""
function has_lhs_parameter_component(
    container::OptimizationContainer,
    key::ParameterKey{<:LeftHandSideTimeSeriesParameter},
    name::AbstractString,
)
    attributes = get_attributes(get_parameter(container, key))
    return haskey(attributes.component_name_to_ts_uuid, name)
end

"""
Per-time-step values of the LHS parameter `key` for component `name`, as parameter times
multiplier. Formulations write these into constraints as fixed coefficients; the model is
rebuilt every simulation step, so each build reads the refreshed values.
"""
function get_lhs_parameter_values(
    container::OptimizationContainer,
    key::ParameterKey{<:LeftHandSideTimeSeriesParameter},
    name::AbstractString,
)::Vector{Float64}
    has_lhs_parameter_component(container, key, name) ||
        throw(ArgumentError("$key has no time series row for component $name"))
    param_container = get_parameter(container, key)
    values = get_parameter_column_refs(param_container, name)
    multipliers = get_multiplier_array(param_container)[name, :]
    return [values[t] * multipliers[t] for t in get_time_steps(container)]
end
