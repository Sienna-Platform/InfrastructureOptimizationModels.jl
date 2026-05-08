# Generic helpers for adding costs to expressions and objectives
#
# Design principles:
# - Pass (name::String, ::Type{C}) instead of component::C when that's all we need
# - Separate quantity computation from cost conversion
# - Explicit function names (invariant/variant) instead of multiple dispatch
# - No concrete expression or parameter types - caller passes them
# - Single timestep only - looping stays in PSI/POM

#######################################
###### Constituent Propagation ########
#######################################

# Default no-op: any expression type that is not a ConstituentCostExpression
# does not propagate into the aggregate ProductionCostExpression.
_propagate_to_production_cost!(
    ::OptimizationContainer,
    ::Type{<:ExpressionType},
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::String,
    ::Int,
    ::Any,
) = nothing

# ConstituentCostExpression overload: also write the same cost into
# ProductionCostExpression so the aggregate stays consistent with its parts.
function _propagate_to_production_cost!(
    container::OptimizationContainer,
    ::Type{<:ConstituentCostExpression},
    ::Type{C},
    name::String,
    t::Int,
    cost,
) where {C <: IS.InfrastructureSystemsComponent}
    has_container_key(container, ProductionCostExpression, C) || return
    prod_expr = get_expression(container, ProductionCostExpression, C)
    JuMP.add_to_expression!(prod_expr[name, t], cost)
    return
end

#######################################
######## Linear Cost Helpers ##########
#######################################

"""
Add cost term to a target expression (no objective hook).

Computes `cost = quantity * rate`, adds it to expression `E` for component `C` at
time `t` if that expression exists, and propagates into `ProductionCostExpression`
when `E <: ConstituentCostExpression`. Use this when the caller wants to record
the cost in an expression container without adding to the JuMP objective (e.g.,
fuel consumption, where the term is a downstream quantity rather than a cost
that should be minimized).

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `rate`: scalar cost rate (e.g., \$/MWh, \$/MMBTU)
- `E`: target expression type (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_to_expression!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    rate::Float64,
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    cost = quantity * rate
    if has_container_key(container, E, C)
        expr = get_expression(container, E, C)
        JuMP.add_to_expression!(expr[name, t], cost)
    end
    _propagate_to_production_cost!(container, E, C, name, t, cost)
    return cost
end

"""
Add cost term to expression and invariant objective.

Computes `cost = quantity * rate`, adds to target expression (if present),
propagates to `ProductionCostExpression` (when applicable), and adds to the
time-invariant part of the objective.

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `rate`: scalar cost rate (e.g., \$/MWh, \$/MMBTU)
- `E`: target expression type (caller provides, e.g., ProductionCostExpression)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_invariant!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    rate::Float64,
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    cost = add_cost_term_to_expression!(container, quantity, rate, E, C, name, t)
    add_to_objective_invariant_expression!(container, cost)
    return cost
end

"""
Add cost term to expression and variant objective.

Fetches rate from parameter, computes `cost = quantity * rate`, adds to target
expression (if present), and adds to the time-variant part of the objective.

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `P`: parameter type for the rate (caller provides, e.g., FuelCostParameter)
- `E`: target expression type (caller provides, e.g., ProductionCostExpression)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_variant!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    ::Type{P},
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {P <: ParameterType, E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    param = get_parameter_array(container, P, C)
    mult = get_parameter_multiplier_array(container, P, C)
    rate = param[name, t] * mult[name, t]
    cost = quantity * rate
    if has_container_key(container, E, C)
        expr = get_expression(container, E, C)
        JuMP.add_to_expression!(expr[name, t], cost)
    end
    _propagate_to_production_cost!(container, E, C, name, t, cost)
    add_to_objective_variant_expression!(container, cost)
    return cost
end

"""
Add cost term to expression and variant objective with explicit rate.

Like `add_cost_term_invariant!` but adds to variant objective. Use when the rate
is computed at runtime rather than looked up from parameters.

Note: the variant/invariant split is about whether the objective expression gets
rebuilt between simulation steps, not about parameter references. Variant terms
are regenerated each step; invariant terms stay constant.

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `rate`: scalar cost rate (e.g., \$/MWh)
- `E`: target expression type (caller provides, e.g., ProductionCostExpression)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_variant!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    rate::Float64,
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    cost = quantity * rate
    if has_container_key(container, E, C)
        expr = get_expression(container, E, C)
        JuMP.add_to_expression!(expr[name, t], cost)
    end
    _propagate_to_production_cost!(container, E, C, name, t, cost)
    add_to_objective_variant_expression!(container, cost)
    return cost
end

# TODO could pass base_power, name, and type instead, to keep things device-agnostic.
"""
Add a proportional (linear) cost to the invariant objective across all time steps.

Normalizes `cost_term` from `power_units` to system per-unit, multiplies by `dt` and
`multiplier`, then adds `variable * rate` to the target expression `E` and the invariant
objective for each time step.

# Arguments
- `container`: the optimization container
- `T`: variable type (caller provides)
- `component`: the component (used for name, base power, and variable lookup)
- `cost_term`: raw proportional cost (e.g., \$/MWh before normalization)
- `power_units`: unit system of `cost_term`
- `multiplier`: additional scalar (e.g., `objective_function_multiplier`, fuel cost)
- `E`: target cost expression type (e.g., `FuelCostExpression`, `VOMCostExpression`).
  Constituent types auto-propagate into `ProductionCostExpression` via
  `_propagate_to_production_cost!`.
"""
function add_proportional_cost_invariant!(
    container::OptimizationContainer,
    ::Type{T},
    component::C,
    cost_term::Float64,
    power_units::IS.UnitSystem,
    multiplier::Float64,
    ::Type{E},
) where {T <: VariableType, C <: IS.InfrastructureSystemsComponent, E <: CostExpressions}
    iszero(cost_term) && return
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    cost_per_unit = get_proportional_cost_per_system_unit(
        cost_term, power_units, base_power, device_base_power)
    dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR
    name = get_name(component)
    rate = cost_per_unit * multiplier * dt
    for t in get_time_steps(container)
        variable = get_variable(container, T, C)[name, t]
        add_cost_term_invariant!(container, variable, rate, E, C, name, t)
    end
    return
end
