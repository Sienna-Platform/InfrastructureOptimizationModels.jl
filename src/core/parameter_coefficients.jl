#################################################################################
# Parameter-driven constraint coefficients, updated in place
#################################################################################

get_parameter_coefficients(container::OptimizationContainer) =
    container.parameter_coefficients

"""
    add_parameter_coefficient!(container, key, name, t, constraint, variable)

Make the coefficient of `variable` in `constraint` follow the parameter `key` for component
`name` at time step `t`. The coefficient already written in `constraint` becomes the base
value, which is scaled by the parameter's current value now and again on every
[`update_parameter_coefficients!`](@ref), so the model never needs a rebuild.
"""
function add_parameter_coefficient!(
    container::OptimizationContainer,
    key::ParameterKey,
    name::AbstractString,
    t::Int,
    constraint::JuMP.ConstraintRef,
    variable::JuMP.VariableRef,
)
    param_array = get_parameter_array(container, key)
    term = ParameterCoefficientTerm(
        constraint,
        variable,
        JuMP.normalized_coefficient(constraint, variable),
        param_array.lookup[1][name],
        t,
    )
    push!(
        get!(get_parameter_coefficients(container), key, ParameterCoefficientTerm[]),
        term,
    )
    _set_parameter_coefficient!(term, param_array.data)
    return
end

"""
    update_parameter_coefficients!(container, key)

Rewrite every coefficient registered for `key` from the parameter's current values. Call it
after the parameter values change.
"""
function update_parameter_coefficients!(container::OptimizationContainer, key::ParameterKey)
    terms = get(get_parameter_coefficients(container), key, nothing)
    isnothing(terms) && return
    parent_param = get_parameter_array(container, key).data
    for term in terms
        _set_parameter_coefficient!(term, parent_param)
    end
    return
end

function _set_parameter_coefficient!(term::ParameterCoefficientTerm, parent_param)
    value = jump_fixed_value(parent_param[term.i, term.t])
    JuMP.set_normalized_coefficient(term.constraint, term.variable, term.base * value)
    return
end
