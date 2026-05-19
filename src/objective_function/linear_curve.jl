"""
Adds to the cost function cost terms for sum of variables with common factor to be used for cost expression for optimization_container model.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_component::IS.CostCurve{IS.LinearCurve} : container for cost to be associated with variable
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    cost_function::IS.CostCurve{IS.LinearCurve},
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    value_curve = get_value_curve(cost_function)
    power_units = get_power_units(cost_function)
    proportional_term = get_proportional_term(get_function_data(value_curve))
    multiplier = objective_function_multiplier(T, U)
    add_proportional_cost_invariant!(
        container, T, component, proportional_term, power_units, multiplier,
        FuelCostExpression)
    return
end

function _add_fuel_linear_variable_cost!(
    container::OptimizationContainer,
    ::Type{T},
    component::V,
    ::Float64, # already normalized in MMBTU/p.u.
    fuel_cost::IS.TimeSeriesKey,
) where {T <: VariableType, V <: IS.InfrastructureSystemsComponent}
    _add_time_varying_fuel_variable_cost!(container, T, component, fuel_cost)
    return
end

"""
Adds to the cost function cost terms for sum of variables with common factor to be used for cost expression for optimization_container model.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_component::IS.FuelCurve{IS.LinearCurve} : container for cost to be associated with variable
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    cost_function::IS.FuelCurve{IS.LinearCurve},
    ::Type{<:AbstractDeviceFormulation},
) where {T <: VariableType}
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    proportional_term = IS.get_proportional_term(IS.get_function_data(value_curve))
    fuel_cost = IS.get_fuel_cost(cost_function)
    # Multiplier is not necessary here. There is no negative cost for fuel curves.
    if fuel_cost isa Float64
        add_proportional_cost_invariant!(
            container, T, component, proportional_term, power_units, fuel_cost,
            FuelCostExpression)
    else
        # Time-varying fuel cost: normalize, then delegate to variant path
        base_power = get_model_base_power(container)
        device_base_power = get_base_power(component)
        fuel_curve_per_unit = get_proportional_cost_per_system_unit(
            proportional_term, power_units, base_power, device_base_power)
        _add_fuel_linear_variable_cost!(
            container, T, component, fuel_curve_per_unit, fuel_cost)
    end
    return
end
