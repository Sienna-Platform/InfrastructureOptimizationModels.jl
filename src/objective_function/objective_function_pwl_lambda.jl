##################################################
# PWL Lambda (Convex Combination) Formulation
#
# Pure formulation math for the lambda/convex-combination PWL method.
# Variables λ_i ∈ [0,1], Σ λ_i = on_status, P = Σ λ_i * P_i,
# cost = Σ λ_i * C_i.
#
# Cost-data-specific mapping (CostCurve/FuelCurve → PiecewiseLinearData)
# stays in piecewise_linear.jl.
#
# Data type relationship:
#   IS.PiecewiseLinearData  →  this formulation (absolute (P, C) values at breakpoints)
#   IS.PiecewisePointCurve  =  InputOutputCurve{PiecewiseLinearData}  →  this formulation
#
# For convex cost curves the LP relaxation is exact — no extra constraints needed.
# For non-convex curves (marginal rate decreases at some breakpoint), an SOS2 constraint
# is automatically added, restricting at most two neighboring λ values to be nonzero.
# This converts the problem to a MILP.
# Contrast with the delta formulation (objective_function_pwl_delta.jl) which
# operates on IS.PiecewiseStepData (slopes) and never needs SOS2.
##################################################

##################################################
################# SOS Methods ####################
##################################################

# might belong in POM, but here for now.
abstract type VariableValueParameter <: RightHandSideParameter end
"""
Parameter to define unit commitment status updated from the system state
"""
struct OnStatusParameter <: VariableValueParameter end

"""
Parameter to fix a variable value (e.g., from feedforward).
"""
struct FixValueParameter <: VariableValueParameter end

"""
Struct to create the PiecewiseLinearCostConstraint associated with a specified variable.

See the piecewise linear cost functions section for more information.
"""
struct PiecewiseLinearCostConstraint <: ConstraintType end

"""
Normalization constraint for PWL cost: sum of delta variables equals on-status.
"""
struct PiecewiseLinearCostNormalizationConstraint <: ConstraintType end

_sos_status(
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:AbstractDeviceFormulation},
) =
    SOSStatusVariable.NO_VARIABLE

"""
Trait function: does device type `T` use commitment (on/off) variables?
Defaults to `false`; POM specializes for thermal device types.
"""
uses_commitment_variables(::Type{<:IS.InfrastructureSystemsComponent}) = false

function _sos_status(
    ::Type{T}, ::Type{<:AbstractThermalUnitCommitment},
) where {T <: IS.InfrastructureSystemsComponent}
    return if uses_commitment_variables(T)
        SOSStatusVariable.VARIABLE
    else
        SOSStatusVariable.NO_VARIABLE
    end
end

function _get_sos_value(
    container::OptimizationContainer,
    ::Type{V},
    component::T,
) where {T <: IS.InfrastructureSystemsComponent, V <: AbstractDeviceFormulation}
    if skip_proportional_cost(component)
        return SOSStatusVariable.NO_VARIABLE
    elseif has_container_key(container, OnStatusParameter, T)
        return SOSStatusVariable.PARAMETER
    else
        return _sos_status(T, V)
    end
end

_get_sos_value(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:AbstractServiceFormulation},
) = SOSStatusVariable.NO_VARIABLE

##################################################
################# PWL Variables ##################
##################################################

# This cases bounds the data by 1 - 0
function add_pwl_variables_lambda!(
    container::OptimizationContainer,
    ::Type{T},
    component_name::String,
    time_period::Int,
    cost_data::IS.PiecewiseLinearData,
) where {T <: IS.InfrastructureSystemsComponent}
    var_container = lazy_container_addition!(container, PiecewiseLinearCostVariable, T)
    # length(PiecewiseStepData) gets number of segments, here we want number of points
    pwlvars = Array{JuMP.VariableRef}(undef, length(cost_data) + 1)
    for i in 1:(length(cost_data) + 1)
        pwlvars[i] =
            var_container[component_name, i, time_period] = JuMP.@variable(
                get_jump_model(container),
                base_name = "PiecewiseLinearCostVariable_$(component_name)_{pwl_$(i), $time_period}",
                lower_bound = 0.0,
                upper_bound = 1.0
            )
    end
    return pwlvars
end

##################################################
################# PWL Constraints ################
##################################################

function _determine_bin_lhs(
    container::OptimizationContainer,
    sos_status::SOSStatusVariable,
    ::Type{T},
    name::String,
    period::Int) where {T <: IS.InfrastructureSystemsComponent}
    if sos_status == SOSStatusVariable.NO_VARIABLE
        @debug "Using Piecewise Linear cost function but no variable/parameter ref for ON status is passed. Default status will be set to online (1.0)" _group =
            LOG_GROUP_COST_FUNCTIONS
        return 1.0
    elseif sos_status == SOSStatusVariable.PARAMETER
        @debug "Using Piecewise Linear cost function with parameter OnStatusParameter, $T" _group =
            LOG_GROUP_COST_FUNCTIONS
        return get_parameter(container, OnStatusParameter, T).parameter_array[name, period]
    elseif sos_status == SOSStatusVariable.VARIABLE
        @debug "Using Piecewise Linear cost function with variable OnVariable $T" _group =
            LOG_GROUP_COST_FUNCTIONS
        return get_variable(container, OnVariable, T)[name, period]
    else
        @assert false
    end
end

"""
Implement the standard constraints for PWL variables. That is:

```math
\\sum_{k\\in\\mathcal{K}} P_k^{max} \\delta_{k,t} = p_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = on_t
```

For compact form (PowerAboveMinimumVariable), use `_add_pwl_constraint_compact!` instead.
"""
function _add_pwl_constraint_standard!(
    container::OptimizationContainer,
    component::T,
    break_points::Vector{Float64},
    sos_status::SOSStatusVariable,
    period::Int,
    power_var::JuMP.VariableRef,
    must_run::Bool = false,
) where {T <: IS.InfrastructureSystemsComponent}
    name = get_name(component)
    n_points = length(break_points)

    # Get PWL delta variables
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable, T)
    pwl_vars = [pwl_var_container[name, i, period] for i in 1:n_points]

    # Linking constraint: power_var == sum(pwl_vars * breakpoints)
    add_pwl_linking_constraint!(
        container,
        PiecewiseLinearCostConstraint,
        T,
        name,
        period,
        power_var,
        pwl_vars,
        break_points,
    )

    # Normalization constraint: sum(pwl_vars) == on_status
    if must_run
        bin = 1.0
    else
        bin = _determine_bin_lhs(container, sos_status, T, name, period)
    end
    add_pwl_normalization_constraint!(
        container,
        PiecewiseLinearCostNormalizationConstraint,
        T,
        name,
        period,
        pwl_vars,
        bin,
    )
    return
end

"""
Implement the constraints for PWL variables for Compact form. That is:

```math
\\sum_{k\\in\\mathcal{K}} P_k^{max} \\delta_{k,t} = p_t + P_{\\min} \\cdot u_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = on_t
```

For standard form, use `_add_pwl_constraint_standard!` instead.
"""
function _add_pwl_constraint_compact!(
    container::OptimizationContainer,
    ::T,
    name::String,
    break_points::Vector{Float64},
    sos_status::SOSStatusVariable,
    period::Int,
    power_var::JuMP.VariableRef,
    P_min::Float64,
    must_run::Bool = false,
) where {T <: IS.InfrastructureSystemsComponent}
    n_points = length(break_points)

    # Get on-status for compact form (needed for both linking and normalization)
    if must_run
        bin = 1.0
    else
        bin = _determine_bin_lhs(container, sos_status, T, name, period)
    end

    # Get PWL delta variables
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable, T)
    pwl_vars = [pwl_var_container[name, i, period] for i in 1:n_points]

    # Create constraint container if needed
    if !has_container_key(container, PiecewiseLinearCostConstraint, T)
        con_key = ConstraintKey(PiecewiseLinearCostConstraint, T)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, PiecewiseLinearCostConstraint, T)
    jump_model = get_jump_model(container)

    # Compact form linking constraint includes P_min offset
    con_container[name, period] = JuMP.@constraint(
        jump_model,
        bin * P_min + power_var ==
        sum(pwl_vars[i] * break_points[i] for i in 1:n_points)
    )

    # Normalization constraint: sum(pwl_vars) == on_status
    add_pwl_normalization_constraint!(
        container,
        PiecewiseLinearCostNormalizationConstraint,
        T,
        name,
        period,
        pwl_vars,
        bin,
    )
    return
end

##################################################
################ PWL Expressions #################
##################################################

# accepts scaled function data.
function get_pwl_cost_expression_lambda(
    container::OptimizationContainer,
    ::Type{T},
    name::String,
    time_period::Int,
    cost_data::IS.PiecewiseLinearData,
    multiplier::Float64,
) where {T <: IS.InfrastructureSystemsComponent}
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable, T)
    gen_cost = JuMP.AffExpr(0.0)
    y_coords_cost_data = IS.get_y_coords(cost_data)
    for (i, cost) in enumerate(y_coords_cost_data)
        JuMP.add_to_expression!(
            gen_cost,
            (cost * multiplier),
            pwl_var_container[name, i, time_period],
        )
    end
    return gen_cost
end

##################################################
# Lambda PWL Constraint Helpers
# (linking, normalization, SOS2)
##################################################

"""
Add PWL linking constraint: power variable equals weighted sum of breakpoints.

    P[name, t] == Σ δ[i] * breakpoint[i]

# Arguments
- `container`: the optimization container
- `K`: constraint type (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `power_var`: the power variable to link (JuMP variable reference)
- `pwl_vars`: vector of PWL delta variables
- `breakpoints`: vector of breakpoint values (in p.u.)
"""
function add_pwl_linking_constraint!(
    container::OptimizationContainer,
    ::Type{K},
    ::Type{C},
    name::String,
    t::Int,
    power_var::JuMP.VariableRef,
    pwl_vars::Vector{JuMP.VariableRef},
    breakpoints::Vector{Float64},
) where {K <: ConstraintType, C <: IS.InfrastructureSystemsComponent}
    @assert length(pwl_vars) == length(breakpoints)
    # Create sparse container with (name, time) indexing if it doesn't exist
    if !has_container_key(container, K, C)
        con_key = ConstraintKey(K, C)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, K, C)
    jump_model = get_jump_model(container)
    con_container[name, t] = JuMP.@constraint(
        jump_model,
        power_var == sum(pwl_vars[i] * breakpoints[i] for i in eachindex(breakpoints))
    )
    return
end

"""
Add PWL normalization constraint: delta variables sum to on_status.

    Σ δ[i] == on_status

# Arguments
- `container`: the optimization container
- `K`: constraint type (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `pwl_vars`: vector of PWL delta variables
- `on_status`: the on/off status (1.0, or a JuMP variable/parameter)
"""
function add_pwl_normalization_constraint!(
    container::OptimizationContainer,
    ::Type{K},
    ::Type{C},
    name::String,
    t::Int,
    pwl_vars::Vector{JuMP.VariableRef},
    on_status::JuMPOrFloat,
) where {K <: ConstraintType, C <: IS.InfrastructureSystemsComponent}
    # Create sparse container with (name, time) indexing if it doesn't exist
    if !has_container_key(container, K, C)
        con_key = ConstraintKey(K, C)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, K, C)
    jump_model = get_jump_model(container)
    con_container[name, t] = JuMP.@constraint(
        jump_model,
        sum(pwl_vars) == on_status
    )
    return
end

"""
Add SOS2 constraint for PWL variables (required for non-convex curves).

# Arguments
- `container`: the optimization container
- `C`: component type
- `name`: component name
- `t`: time period
- `pwl_vars`: vector of PWL delta variables
"""
function add_pwl_sos2_constraint!(
    container::OptimizationContainer,
    ::Type{C},
    name::String,
    t::Int,
    pwl_vars::Vector{JuMP.VariableRef},
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    n_points = length(pwl_vars)
    JuMP.@constraint(jump_model, pwl_vars in MOI.SOS2(collect(1:n_points)))
    return
end
