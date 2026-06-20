##################################################
# PWL Delta (Incremental/Block) Formulation
#
# Pure objective function formulation for the delta/incremental PWL method.
# Variables δ_k >= 0 with block width bounds,
# P = Σ δ_k + offset, objective = Σ δ_k * slope_k.
#
# ValueCurve-specific mapping (ValueCurve → slopes/breakpoints)
# stays in value_curve_cost.jl.
#
# Data type relationship:
#   IS.PiecewiseStepData  →  this formulation (slopes already stored per segment)
#   IS.PiecewiseIncrementalCurve  =  IncrementalCurve{PiecewiseStepData}  →  this formulation
#   IS.PiecewiseAverageCurve      =  AverageRateCurve{PiecewiseStepData}  →  this formulation
#
# The segment-width upper bounds (δ_k ≤ P_{k+1} - P_k) naturally enforce ordering
# without SOS2, so non-convex (declining slope) curves remain LP-feasible.
# Contrast with the lambda formulation (objective_function_pwl_lambda.jl) which
# operates on IS.PiecewiseLinearData (point values) and requires SOS2 for non-convex curves.
##################################################

##################################################
############## PWL Delta Variables ###############
##################################################

"""
Create PWL delta variables for a component at a given time period.

Creates `n_points` variables with specified bounds.

# Arguments
- `container`: the optimization container
- `V`: variable type for the delta variables (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `n_points`: number of PWL points (= number of delta variables)
- `upper_bound`: upper bound for variables (default 1.0; use `Inf` for block offer
   formulation where segment capacity is enforced by constraints instead)

# Returns
Vector of the created JuMP variables.
"""
function add_pwl_variables_delta!(
    container::OptimizationContainer,
    ::Type{V},
    ::Type{C},
    name::String,
    t::Int,
    n_points::Int;
    upper_bound::Float64 = 1.0,
) where {V <: SparseVariableType, C <: IS.InfrastructureSystemsComponent}
    # SparseVariableType dispatch automatically creates container with (String, Int, Int) keys
    # axes are (name, pwl_index, time_step).
    var_container = lazy_container_addition!(container, V, C)
    pwl_vars = Vector{JuMP.VariableRef}(undef, n_points)
    jump_model = get_jump_model(container)
    for i in 1:n_points
        if isfinite(upper_bound)
            pwl_vars[i] =
                var_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "$(V)_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = upper_bound,
                )
        else
            pwl_vars[i] =
                var_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "$(V)_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                )
        end
    end
    return pwl_vars
end

##################################################
############## PWL Delta Expressions #############
##################################################

"""
Compute PWL objective expression from delta variables and slopes.

Returns the objective expression without adding it to the objective (caller decides
whether to use invariant or variant).

    objective_term = Σ δ[i] * slope[i] * multiplier

# Arguments
- `pwl_vars`: vector of PWL delta variables
- `slopes`: vector of slope values (rate per segment, already normalized)
- `multiplier`: additional multiplier (e.g., dt for time resolution)

# Returns
JuMP affine expression representing the objective function term.
"""
function get_pwl_cost_expression_delta(
    pwl_vars::AbstractVector{JuMP.VariableRef},
    slopes::AbstractVector{Float64},
    multiplier::Float64,
)
    @assert length(pwl_vars) == length(slopes)
    cost = JuMP.AffExpr(0.0)
    for (i, slope) in enumerate(slopes)
        JuMP.add_to_expression!(cost, slope * multiplier, pwl_vars[i])
    end
    return cost
end

##################################################
############ PWL Block Offer Constraints #########
##################################################

"""
Add block-offer PWL constraints: linking constraint and per-block upper bounds.

    power_var == Σ δ[k] + min_power_offset
    δ[k] <= breakpoints[k+1] - breakpoints[k]   for each k

# Arguments
- `jump_model`: the JuMP model
- `con_container`: constraint container to store the linking constraint (indexed [name, t])
- `name`: component name
- `t`: time period
- `power_var`: the power variable being linked
- `pwl_vars`: vector of block-offer delta variables (length = n_blocks)
- `breakpoints`: vector of breakpoint values (length = n_blocks + 1)
- `min_power_offset`: offset for minimum generation power (default 0.0)
"""
function add_pwl_block_offer_constraints!(
    jump_model::JuMP.Model,
    con_container,
    name::String,
    t::Int,
    power_var::JuMPOrFloat,
    pwl_vars::Vector{JuMP.VariableRef},
    breakpoints::Vector{<:JuMPOrFloat},
    min_power_offset::JuMPOrFloat = 0.0,
)
    @assert length(pwl_vars) == length(breakpoints) - 1
    sum_pwl = sum(pwl_vars) + min_power_offset
    con_container[name, t] = JuMP.@constraint(jump_model, power_var == sum_pwl)
    for (ix, var) in enumerate(pwl_vars)
        JuMP.@constraint(jump_model, var <= breakpoints[ix + 1] - breakpoints[ix])
    end
    return
end

##################################################
# Min-Gen-Power Dispatch Defaults
# POM overrides these for specific device types and formulations.
##################################################

# Fallbacks accept any formulation (device or service): service formulations
# (e.g. StepwiseCostReserve) carry no min-gen-power offset, so they resolve here.
_include_min_gen_power_in_constraint(
    ::Type,
    ::Type{<:VariableType},
    ::Type,
) = false

_include_constant_min_gen_power_in_constraint(
    ::Type,
    ::Type{<:VariableType},
    ::Type,
) = false

##################################################
# PWL Block Offer Constraint Wrapper
# The ReserveDemandCurve-specific overload is in POM.
##################################################

"""
Implement the constraints for PWL Block Offer variables. That is:

```math
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = p_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} \\leq P_{k+1,t}^{max} - P_{k,t}^{max}
```
"""
function add_pwl_constraint_delta!(
    container::OptimizationContainer,
    component::T,
    ::Type{U},
    ::Type{D},
    break_points::Vector{<:JuMPOrFloat},
    pwl_vars::Vector{JuMP.VariableRef},
    period::Int,
    ::Type{W};
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: IS.InfrastructureSystemsComponent, U <: VariableType,
    D,
    W <: AbstractPiecewiseLinearBlockOfferConstraint}
    variables = get_variable(container, U, T, meta)
    const_container = lazy_container_addition!(
        container,
        W,
        T,
        axes(variables)...;
        meta = meta,
    )
    name = get_name(component)

    min_power_offset = if _include_constant_min_gen_power_in_constraint(T, U, D)
        jump_fixed_value(first(break_points))::Float64
    elseif _include_min_gen_power_in_constraint(T, U, D)
        p1::Float64 = jump_fixed_value(first(break_points))
        on_vars = get_variable(container, OnVariable, T)
        p1 * on_vars[name, period]
    else
        0.0
    end

    add_pwl_block_offer_constraints!(
        get_jump_model(container),
        const_container,
        name,
        period,
        variables[name, period],
        pwl_vars,
        break_points,
        min_power_offset,
    )
    return
end
