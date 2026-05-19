##################################################
######## CostCurve/FuelCurve → Lambda PWL ########
##################################################

_get_pwl_cost_multiplier(::IS.FuelCurve{IS.PiecewisePointCurve},
    ::Type{<:VariableType},
    ::Type{<:AbstractDeviceFormulation},
) = 1.0

_get_pwl_cost_multiplier(::IS.CostCurve{IS.PiecewisePointCurve},
    ::Type{U},
    ::Type{V},
) where {U <: VariableType, V <: AbstractDeviceFormulation} =
    objective_function_multiplier(U, V)

# FuelCurve/CostCurve: scale for units, then call function data version
function get_pwl_cost_expression_lambda(
    container::OptimizationContainer,
    component::T,
    time_period::Int,
    cost_function::Union{
        IS.FuelCurve{IS.PiecewisePointCurve},
        IS.CostCurve{IS.PiecewisePointCurve},
    },
    ::Type{U},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    cost_component = IS.get_function_data(value_curve)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    cost_data_normalized = get_piecewise_pointcurve_per_system_unit(
        cost_component,
        power_units,
        base_power,
        device_base_power,
    )
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR
    multiplier = _get_pwl_cost_multiplier(cost_function, U, V)
    name = get_name(component)
    fuel_consumption_expression = get_pwl_cost_expression_lambda(
        container,
        T,
        name,
        time_period,
        cost_data_normalized,
        dt * multiplier,
    )
    return fuel_consumption_expression
end

##################################################
######## CostCurve: PiecewisePointCurve ##########
##################################################

"""
Add PWL cost terms using the **lambda (convex combination) formulation**.

Given a `PiecewisePointCurve` with breakpoints ``(P_i, C_i)``, this function:

1. Creates lambda variables ``\\lambda_i \\in [0, 1]`` at each breakpoint via `add_pwl_variables_lambda!`.
2. Adds linking and normalization constraints via `_add_pwl_constraint_standard!`.
3. If the cost curve is **non-convex**, adds an SOS2 adjacency constraint so that at most
   two neighboring ``\\lambda`` values are nonzero.
4. Builds the cost expression ``C = \\sum_i \\lambda_i \\, C(P_i)``.

Returns a vector of cost expressions, one per time step, which the caller adds to the
objective function.

See also: [`add_pwl_term_delta!`](@ref) for the delta (block-offer) formulation used by
`MarketBidCost`.
"""
function add_pwl_term_lambda!(
    container::OptimizationContainer,
    component::T,
    cost_function::Union{
        IS.CostCurve{IS.PiecewisePointCurve},
        IS.FuelCurve{IS.PiecewisePointCurve},
    },
    ::Type{U},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    # multiplier = objective_function_multiplier(U(), V())
    name = get_name(component)
    value_curve = IS.get_value_curve(cost_function)
    cost_component = IS.get_function_data(value_curve)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    power_units = IS.get_power_units(cost_function)

    # Normalize data
    data = get_piecewise_pointcurve_per_system_unit(
        cost_component,
        power_units,
        base_power,
        device_base_power,
    )

    if all(iszero.((point -> point.y).(IS.get_points(data))))
        @debug "All cost terms for component $(name) are 0.0" _group =
            LOG_GROUP_COST_FUNCTIONS
        return
    end

    cost_is_convex = IS.is_convex(data)
    if !cost_is_convex
        @warn(
            "The cost function provided for $(name) is not compatible with a linear PWL cost function. " *
            "An SOS-2 formulation will be added to the model. This will result in additional binary variables."
        )
    end
    break_points = IS.get_x_coords(data)
    time_steps = get_time_steps(container)
    pwl_cost_expressions = Vector{JuMP.AffExpr}(undef, time_steps[end])
    sos_val = _get_sos_value(container, V, component)
    power_variables = get_variable(container, U, T)
    n_points = length(break_points)
    for t in time_steps
        add_pwl_variables_lambda!(container, T, name, t, data)
        _add_pwl_constraint_standard!(
            container,
            component,
            break_points,
            sos_val,
            t,
            power_variables[name, t],
        )
        if !cost_is_convex
            pwl_var_container = get_variable(container, PiecewiseLinearCostVariable, T)
            pwl_vars = [pwl_var_container[name, i, t] for i in 1:n_points]
            add_pwl_sos2_constraint!(container, T, name, t, pwl_vars)
        end
        pwl_cost =
            get_pwl_cost_expression_lambda(container, component, t, cost_function, U, V)
        pwl_cost_expressions[t] = pwl_cost
    end
    return pwl_cost_expressions
end

# ThermalDispatchNoMin implementation is in POM

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::IS.CostCurve{IS.PiecewisePointCurve}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::C,
    cost_function::IS.CostCurve{IS.PiecewisePointCurve},
    ::Type{U},
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    component_name = get_name(component)
    @debug "PWL Variable Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    pwl_cost_expressions =
        add_pwl_term_lambda!(container, component, cost_function, T, U)
    isnothing(pwl_cost_expressions) && return
    for t in get_time_steps(container)
        add_cost_to_expression!(
            container,
            FuelCostExpression,
            pwl_cost_expressions[t],
            C,
            component_name,
            t,
        )
        add_to_objective_invariant_expression!(container, pwl_cost_expressions[t])
    end
    return
end

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.
# Arguments
  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::IS.FuelCurve{IS.PiecewisePointCurve}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::V,
    cost_function::IS.FuelCurve{IS.PiecewisePointCurve},
    ::Type{U},
) where {
    T <: VariableType,
    V <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    component_name = get_name(component)
    @debug "PWL Variable Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    pwl_fuel_consumption_expressions =
        add_pwl_term_lambda!(container, component, cost_function, T, U)
    isnothing(pwl_fuel_consumption_expressions) && return

    # IS getter: simply returns the field of the FuelCurve struct
    is_time_variant_ = is_time_variant(IS.get_fuel_cost(cost_function))
    for t in get_time_steps(container)
        fuel_cost_value = if is_time_variant_
            param = get_parameter_array(container, FuelCostParameter, V)
            mult = get_parameter_multiplier_array(container, FuelCostParameter, V)
            param[component_name, t] * mult[component_name, t]
        else
            get_fuel_cost(component)
        end
        pwl_cost_expression = pwl_fuel_consumption_expressions[t] * fuel_cost_value
        add_cost_to_expression!(
            container,
            FuelCostExpression,
            pwl_cost_expression,
            V,
            component_name,
            t,
        )
        add_cost_to_expression!(
            container,
            FuelConsumptionExpression,
            pwl_fuel_consumption_expressions[t],
            V,
            component_name,
            t,
        )
        if is_time_variant_
            add_to_objective_variant_expression!(container, pwl_cost_expression)
        else
            add_to_objective_invariant_expression!(container, pwl_cost_expression)
        end
    end
    return
end

##################################################
###### CostCurve: PiecewiseIncrementalCurve ######
######### and PiecewiseAverageCurve ##############
##################################################

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::Union{IS.CostCurve{IS.PiecewiseIncrementalCurve}, IS.CostCurve{IS.PiecewiseAverageCurve}}: container for piecewise linear cost
"""
_rebuild_with_value_curve(c::IS.CostCurve, vc) =
    IS.CostCurve(; value_curve = vc, power_units = IS.get_power_units(c))
_rebuild_with_value_curve(c::IS.FuelCurve, vc) = IS.FuelCurve(;
    value_curve = vc,
    power_units = IS.get_power_units(c),
    fuel_cost = IS.get_fuel_cost(c),
)

function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    cost_function::Union{
        IS.CostCurve{IS.PiecewiseIncrementalCurve},
        IS.CostCurve{IS.PiecewiseAverageCurve},
        IS.FuelCurve{IS.PiecewiseIncrementalCurve},
        IS.FuelCurve{IS.PiecewiseAverageCurve},
    },
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    pointbased_value_curve = IS.InputOutputCurve(IS.get_value_curve(cost_function))
    pointbased_cost_function =
        _rebuild_with_value_curve(cost_function, pointbased_value_curve)
    add_variable_cost_to_objective!(container, T, component, pointbased_cost_function, U)
    return
end
