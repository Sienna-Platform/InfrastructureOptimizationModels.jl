# Add quadratic cost term to objective and expression
function _add_quadraticcurve_variable_term_to_model!(
    container::OptimizationContainer,
    ::Type{T},
    component::V,
    proportional_term_per_unit::Float64,
    quadratic_term_per_unit::Float64,
    time_period::Int,
) where {T <: VariableType, V <: IS.InfrastructureSystemsComponent}
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR
    name = get_name(component)
    var = get_variable(container, T, V)[name, time_period]

    if quadratic_term_per_unit >= eps()
        @debug "$name Quadratic Variable Cost" _group = LOG_GROUP_COST_FUNCTIONS name
        q_cost =
            (var .^ 2 * quadratic_term_per_unit + var * proportional_term_per_unit) * dt
        add_to_objective_invariant_expression!(container, q_cost)
        # add_cost_to_expression! handles ConstituentCostExpression -> ProductionCostExpression
        # propagation via _propagate_to_production_cost!.
        add_cost_to_expression!(
            container, FuelCostExpression, q_cost, V, name, time_period)
    else
        add_cost_term_invariant!(
            container, var, proportional_term_per_unit * dt,
            FuelCostExpression, V, name, time_period)
    end
    return
end

# Dispatch for vector proportional/quadratic terms
function _add_quadraticcurve_variable_cost!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    component::IS.InfrastructureSystemsComponent,
    proportional_term_per_unit::Vector{Float64},
    quadratic_term_per_unit::Vector{Float64},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    lb, ub = get_active_power_limits(component)
    for t in get_time_steps(container)
        _check_quadratic_monotonicity(
            get_name(component),
            quadratic_term_per_unit[t],
            proportional_term_per_unit[t],
            lb,
            ub,
        )
        _add_quadraticcurve_variable_term_to_model!(
            container,
            T,
            component,
            proportional_term_per_unit[t],
            quadratic_term_per_unit[t],
            t,
        )
    end
    return
end

# Dispatch for scalar proportional/quadratic terms
function _add_quadraticcurve_variable_cost!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    component::IS.InfrastructureSystemsComponent,
    proportional_term_per_unit::Float64,
    quadratic_term_per_unit::Float64,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    lb, ub = get_active_power_limits(component)
    _check_quadratic_monotonicity(get_name(component),
        quadratic_term_per_unit,
        proportional_term_per_unit,
        lb,
        ub,
    )
    for t in get_time_steps(container)
        _add_quadraticcurve_variable_term_to_model!(
            container,
            T,
            component,
            proportional_term_per_unit,
            quadratic_term_per_unit,
            t,
        )
    end
    return
end

function _check_quadratic_monotonicity(
    name::String,
    quad_term::Float64,
    linear_term::Float64,
    lb::Float64,
    ub::Float64,
)
    fp_lb = 2 * quad_term * lb + linear_term
    fp_ub = 2 * quad_term * ub + linear_term

    if fp_lb < 0 || fp_ub < 0
        @warn "Cost function for component $name is not monotonically increasing in the range [$lb, $ub]. \
               This can lead to unexpected outputs"
    end
    return
end

@doc raw"""
Adds to the cost function cost terms for sum of variables with common factor to be used for cost expression for optimization_container model.

# Equation

``` gen_cost = dt*sign*(sum(variable.^2)*cost_data[1] + sum(variable)*cost_data[2]) ```

# LaTeX

`` cost = dt\times sign (sum_{i\in I} c_1 v_i^2 + sum_{i\in I} c_2 v_i ) ``

for quadratic factor large enough. If the first term of the quadratic objective is 0.0, adds a
linear cost term `sum(variable)*cost_data[2]`

# Arguments

* container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
* var_key::VariableKey: The variable name
* component_name::String: The component_name of the variable container
* cost_component::IS.CostCurve{IS.QuadraticCurve} : container for quadratic factors
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    cost_function::IS.CostCurve{IS.QuadraticCurve},
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    multiplier = objective_function_multiplier(T, U)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    value_curve = get_value_curve(cost_function)
    power_units = get_power_units(cost_function)
    cost_component = get_function_data(value_curve)
    quadratic_term = get_quadratic_term(cost_component)
    proportional_term = get_proportional_term(cost_component)
    proportional_term_per_unit = get_proportional_cost_per_system_unit(
        proportional_term,
        power_units,
        base_power,
        device_base_power,
    )
    quadratic_term_per_unit = get_quadratic_cost_per_system_unit(
        quadratic_term,
        power_units,
        base_power,
        device_base_power,
    )
    _add_quadraticcurve_variable_cost!(
        container,
        T,
        U,
        component,
        multiplier * proportional_term_per_unit,
        multiplier * quadratic_term_per_unit,
    )
    return
end

function _add_fuel_quadratic_variable_cost!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    component::IS.InfrastructureSystemsComponent,
    proportional_fuel_curve::Float64,
    quadratic_fuel_curve::Float64,
    fuel_cost::Float64,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    _add_quadraticcurve_variable_cost!(
        container,
        T,
        U,
        component,
        proportional_fuel_curve * fuel_cost,
        quadratic_fuel_curve * fuel_cost,
    )
end

function _add_fuel_quadratic_variable_cost!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{<:AbstractDeviceFormulation},
    component::IS.InfrastructureSystemsComponent,
    proportional_fuel_curve::Float64,
    quadratic_fuel_curve::Float64,
    fuel_cost::IS.TimeSeriesKey,
) where {T <: VariableType}
    _add_time_varying_fuel_variable_cost!(container, T, component, fuel_cost)
end

@doc raw"""
Adds to the cost function cost terms for sum of variables with common factor to be used for cost expression for optimization_container model.

# Equation

``` gen_cost = dt*(sum(variable.^2)*cost_data[1]*fuel_cost + sum(variable)*cost_data[2]*fuel_cost) ```

# LaTeX

`` cost = dt\times  (sum_{i\in I} c_f c_1 v_i^2 + sum_{i\in I} c_f c_2 v_i ) ``

for quadratic factor large enough. If the first term of the quadratic objective is 0.0, adds a
linear cost term `sum(variable)*cost_data[2]`

# Arguments

* container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
* var_key::VariableKey: The variable name
* component_name::String: The component_name of the variable container
* cost_component::IS.FuelCurve{IS.QuadraticCurve} : container for quadratic factors
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    cost_function::IS.FuelCurve{IS.QuadraticCurve},
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    multiplier = objective_function_multiplier(T, U)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    cost_component = IS.get_function_data(value_curve)
    quadratic_term = IS.get_quadratic_term(cost_component)
    proportional_term = IS.get_proportional_term(cost_component)
    proportional_term_per_unit = get_proportional_cost_per_system_unit(
        proportional_term,
        power_units,
        base_power,
        device_base_power,
    )
    quadratic_term_per_unit = get_quadratic_cost_per_system_unit(
        quadratic_term,
        power_units,
        base_power,
        device_base_power,
    )
    fuel_cost = IS.get_fuel_cost(cost_function)
    # Multiplier is not necessary here. There is no negative cost for fuel curves.
    _add_fuel_quadratic_variable_cost!(
        container,
        T,
        U,
        component,
        multiplier * proportional_term_per_unit,
        multiplier * quadratic_term_per_unit,
        fuel_cost,
    )
    return
end
