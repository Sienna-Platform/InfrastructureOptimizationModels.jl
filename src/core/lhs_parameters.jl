#################################################################################
# Left-hand-side parameters: values written into constraints as fixed coefficients
#################################################################################

# Parameters a formulation may write into constraints as fixed coefficients.
const LeftHandSideParameter = Union{LeftHandSideTimeSeriesParameter, EventParameter}

"""
Whether `name` has a row in the LHS parameter container `key`: a series row for a time-series
parameter, an event row for an [`EventParameter`](@ref). Components without one contribute a
fixed coefficient chosen by the formulation.
"""
function has_lhs_parameter_component(
    container::OptimizationContainer,
    key::ParameterKey{<:LeftHandSideParameter},
    name::AbstractString,
)
    param_container = get_parameter(container, key)
    return _has_lhs_row(get_attributes(param_container), param_container, name)
end

_has_lhs_row(attributes::TimeSeriesAttributes, _, name) =
    haskey(attributes.component_name_to_ts_uuid, name)
_has_lhs_row(::EventParametersAttributes, param_container, name) =
    name in axes(get_parameter_array(param_container), 1)

"""
Values of the LHS parameter `key` for component `name`, as parameter times multiplier: a
vector over time steps, or a `(position, time)` matrix when the container has an extra axis.
Formulations write these into constraints as fixed coefficients; the model is rebuilt every
simulation step, so each build reads the refreshed values. An [`EventParameter`](@ref) read this
way must hold numbers, so a recurrent model using one must be built with `rebuild_model = true`.
"""
function get_lhs_parameter_values(
    container::OptimizationContainer,
    key::ParameterKey{<:LeftHandSideParameter},
    name::AbstractString,
)
    has_lhs_parameter_component(container, key, name) ||
        throw(ArgumentError("$key has no row for component $name"))
    param_container = get_parameter(container, key)
    values = _lhs_numbers(_lhs_row(get_attributes(param_container), param_container, name), key)
    multipliers = get_multiplier_array(param_container)
    return values.data .* multipliers[expand_ixs((name,), multipliers)...].data
end

_lhs_row(::TimeSeriesAttributes, param_container, name) =
    get_parameter_column_refs(param_container, name)
function _lhs_row(::EventParametersAttributes, param_container, name)
    param_array = get_parameter_array(param_container)
    return param_array[expand_ixs((name,), param_array)...]
end

_lhs_numbers(values::AbstractArray{Float64}, _) = values
_lhs_numbers(::AbstractArray{JuMP.VariableRef}, key) = throw(
    ArgumentError(
        "$key is read as a left-hand-side coefficient, so it must hold numbers: build the \
         model with rebuild_model = true.",
    ),
)

"""
Values of the LHS parameter `key` for component `name` in the shape of its series: `(length
of each value axis..., time)`. The container's extra axis holds each step flattened
column-major; positions past the component's own values are padding and are dropped.
"""
function get_lhs_parameter_values(
    container::OptimizationContainer,
    key::ParameterKey{<:LeftHandSideTimeSeriesParameter},
    name::AbstractString,
    value_axes::Vector{IS.TimeSeriesAxis},
)
    values = get_lhs_parameter_values(container, key, name)
    _check_extra_axis(values, key)
    dims = Tuple(length(axis.labels) for axis in value_axes)
    return reshape(values[1:get_value_length(value_axes), :], dims..., size(values, 2))
end

_check_extra_axis(::Matrix{Float64}, _) = nothing
_check_extra_axis(::AbstractArray, key) =
    throw(ArgumentError("$key has no extra axis to shape by value axes"))

"""
Number of values in one time step of a series with `value_axes`. A left-hand-side parameter
holds each step flattened column-major along one positional extra axis, sized to the longest
owner of its batch.
"""
get_value_length(value_axes::Vector{IS.TimeSeriesAxis}) =
    prod(axis -> length(axis.labels), value_axes)

# One flattened step as Float64, padded with zeros to the batch's extra axis.
function unwrap_for_param(
    ::LeftHandSideTimeSeriesParameter,
    ts_elem::AbstractVector{<:Real},
    expected_axs::Tuple{AbstractVector},
)
    max_len = length(only(expected_axs))
    if length(ts_elem) > max_len
        throw(
            ArgumentError(
                "A time step holds $(length(ts_elem)) values; the parameter axis has $max_len.",
            ),
        )
    end
    out = zeros(max_len)
    copyto!(out, ts_elem)
    return out
end
