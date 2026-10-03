#################################################################################
# Left-hand-side parameters: constraint coefficients refreshed in place
#################################################################################

"Defining row `y - v * x == 0` of a [`ParameterizedProductVariable`](@ref)."
struct ParameterizedProductConstraint <: ConstraintType end

function _lhs_parameter_container(
    container::OptimizationContainer,
    key::ParameterKey{<:TimeSeriesLHSParameter},
)
    param_container = get_parameter(container, key)
    if ndims(get_parameter_array(param_container)) != 2
        error("$key: LHS parameters support a (component, time) layout only")
    end
    return param_container
end

"""
Whether `name` has a series row in the LHS parameter container `key`. Components without one
contribute a fixed coefficient chosen by the formulation.
"""
function has_lhs_parameter_component(
    container::OptimizationContainer,
    key::ParameterKey{<:TimeSeriesLHSParameter},
    name::AbstractString,
)
    attributes = get_attributes(get_parameter(container, key))
    return haskey(attributes.component_name_to_ts_uuid, name)
end

"""
Add the row `y - v * x == 0`, where `v` is the value of the LHS parameter `key` for component
`name` at time step `t` (parameter times multiplier), and record it so
[`apply_coefficient_bindings!`](@ref) can rewrite `v` in place. Returns the row.

`x`'s coefficient in this row belongs to the binding: the caller consumes `y` elsewhere and
must not add `x` to this row by other means.
"""
function add_parameterized_product_constraint!(
    container::OptimizationContainer,
    key::ParameterKey{<:TimeSeriesLHSParameter},
    name::AbstractString,
    t::Int,
    y::JuMP.VariableRef,
    x::JuMP.VariableRef,
)
    param_container = _lhs_parameter_container(container, key)
    attributes = get_attributes(param_container)
    if !haskey(attributes.component_name_to_ts_uuid, name)
        throw(ArgumentError("$key has no time series row for component $name"))
    end
    param_axis = axes(get_parameter_array(param_container), 1)
    mult_axis = axes(get_multiplier_array(param_container), 1)
    param_row = findfirst(==(_get_ts_uuid(attributes, name)), param_axis)
    mult_row = findfirst(==(name), mult_axis)
    value =
        get_parameter_array_data(param_container)[param_row, t] *
        get_multiplier_array_data(param_container)[mult_row, t]
    constraint = JuMP.@constraint(get_jump_model(container), y - value * x == 0.0)
    bindings = get!(CoefficientBindings, container.coefficient_bindings, key)
    add_binding!(bindings, constraint, x, param_row, mult_row, t)
    return constraint
end

"""
Rewrite every bound coefficient from the current parameter and multiplier values, one
vectorized call per parameter. Run after the parameter arrays are refreshed.
"""
function apply_coefficient_bindings!(container::OptimizationContainer)
    for (key, bindings) in container.coefficient_bindings
        isempty(bindings) && continue
        param_container = get_parameter(container, key)
        _refresh_coefficients!(
            bindings,
            get_parameter_array_data(param_container),
            get_multiplier_array_data(param_container),
        )
        JuMP.set_normalized_coefficient(
            bindings.constraints,
            bindings.variables,
            bindings.coefficients,
        )
    end
    return
end

function _refresh_coefficients!(
    bindings::CoefficientBindings,
    params::AbstractMatrix{Float64},
    mults::AbstractMatrix{Float64},
)
    for i in eachindex(bindings.coefficients)
        t = bindings.time_steps[i]
        bindings.coefficients[i] =
            -(params[bindings.param_rows[i], t] * mults[bindings.mult_rows[i], t])
    end
    return
end
