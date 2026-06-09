#################################################################################
# JuMP expression helpers
# These wrap JuMP.add_to_expression! with consistent patterns
# Named to clarify their different purposes:
# - add_constant_to_jump_expression!: adds a single constant value
# - add_proportional_to_jump_expression!: adds multiplier * variable (or parameter * multiplier)
# - add_linear_to_jump_expression!: adds constant + multiplier * variable
#################################################################################

"""
Add constant value to JuMP expression.
"""
function add_constant_to_jump_expression!(
    expression::T,
    value::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    JuMP.add_to_expression!(expression, value)
    return
end

"""
Add variable with multiplier to JuMP expression: expression += multiplier * var
"""
function add_proportional_to_jump_expression!(
    expression::T,
    var::U,
    multiplier::Float64,
) where {T <: JuMP.AbstractJuMPScalar, U <: JuMP.AbstractJuMPScalar}
    JuMP.add_to_expression!(expression, multiplier, var)
    return
end

"""
Add product of parameter and multiplier to JuMP expression: expression += parameter * multiplier
"""
function add_proportional_to_jump_expression!(
    expression::T,
    parameter::Float64,
    multiplier::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    add_constant_to_jump_expression!(expression, parameter * multiplier)
    return
end

"""
Add affine term to JuMP expression: expression += constant + multiplier * var
"""
function add_linear_to_jump_expression!(
    expression::T,
    var::JuMP.VariableRef,
    multiplier::Float64,
    constant::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    add_constant_to_jump_expression!(expression, constant)
    add_proportional_to_jump_expression!(expression, var, multiplier)
    return
end

"""
Generic driver for device-injection `add_to_expression!` methods.

For each device in `devices` and each time step, adds a proportional term to one or
more expression entries. Two closures separate the axes that distinguish these
methods, so the network-model and the term-source can vary independently while the
loop itself is written once:

  - `targets_fn(d)` returns a 1- or 2-element tuple of `(expression_matrix, row_index)`
    targets identifying which expression entries device `d` contributes to: one entry
    for single-bus/area/system network models, two for PTDF/AreaPTDF (a nodal entry
    plus a system/area entry). This is where the network-model dependence lives.
  - `term_fn(d)` returns a per-device closure `t -> (value, multiplier)` giving the
    term added at each time step, where `value` is a JuMP variable/parameter
    reference or a `Float64` constant and `multiplier::Float64`. This is where the
    variable/parameter/constant source lives. Per-device setup (name lookups, bounds,
    warnings) belongs in `term_fn` so it runs once per device rather than per step.

The same `value * multiplier` term is added to every target via
[`add_proportional_to_jump_expression!`](@ref).
"""
function add_device_terms_to_expression!(
    container::OptimizationContainer,
    targets_fn::F,
    term_fn::G,
    devices::Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
) where {F <: Function, G <: Function, D}
    time_steps = get_time_steps(container)
    for d in devices
        targets = targets_fn(d)
        term = term_fn(d)
        for t in time_steps
            value, multiplier = term(t)
            _apply_term_to_targets!(targets, value, multiplier, t)
        end
    end
    return
end

# Apply a term to each `(expression_matrix, row_index)` target. A device contributes to
# either one target (single-bus/area/system network models) or two (PTDF/AreaPTDF: a
# nodal entry plus a system/area entry). Tail recursion ensures type stability for 
# the hetereogeneous length 2 tuples.
const _BalanceTermValue = Union{Float64, JuMP.AbstractJuMPScalar}

_apply_term_to_targets!(::Tuple{}, ::_BalanceTermValue, ::Float64, ::Int) = nothing

function _apply_term_to_targets!(
    targets::Tuple,
    value::_BalanceTermValue,
    multiplier::Float64,
    t::Int,
)
    expression, row = targets[1]
    add_proportional_to_jump_expression!(expression[row, t], value, multiplier)
    # perf note: only called with length 1 or 2 tuples, but writing separate methods
    # has no advantage, because compiler unrolls the tail recursion.
    return _apply_term_to_targets!(Base.tail(targets), value, multiplier, t)
end
