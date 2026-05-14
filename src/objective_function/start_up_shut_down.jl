# ideally would define in POM, but put here for now.
"Parameter to define startup cost time series"
struct StartupCostParameter <: ObjectiveFunctionParameter end

"Parameter to define shutdown cost time series"
struct ShutdownCostParameter <: ObjectiveFunctionParameter end

# Extract the scalar shutdown cost from either a Float64 (ThermalGenerationCost) or
# a LinearCurve (MarketBidCost) whose proportional term is the cost.
_shutdown_cost_value(x::Float64) = x
_shutdown_cost_value(x::IS.LinearCurve) = IS.get_proportional_term(x)

# Trait: does this cost type store startup/shutdown in time-series parameters?
# POM adds an override for PSY.MarketBidTimeSeriesCost; mocks can duck-type.
_is_time_series_cost(::IS.DeviceParameter) = false

#################################################################################
# Shutdown cost
#################################################################################

function add_shut_down_cost!(
    container::OptimizationContainer,
    ::Type{U},
    devices::IS.FlattenIteratorWrapper{T},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(U, V)
    for d in devices
        get_must_run(d) && continue
        # Function barrier: op_cost type becomes a compile-time parameter
        _add_shut_down_cost_per_device!(
            container, U, T, get_name(d), get_operation_cost(d), multiplier)
    end
    return
end

function _add_shut_down_cost_per_device!(
    container::OptimizationContainer,
    ::Type{U},
    ::Type{T},
    name::String,
    op_cost::IS.DeviceParameter,
    multiplier,
) where {U <: VariableType, T <: IS.InfrastructureSystemsComponent}
    if _is_time_series_cost(op_cost)
        param = get_parameter_array(container, ShutdownCostParameter, T)
        mult = get_parameter_multiplier_array(container, ShutdownCostParameter, T)
        for t in get_time_steps(container)
            cost_term = param[name, t] * mult[name, t]
            iszero(cost_term) && continue
            rate = cost_term * multiplier
            variable = get_variable(container, U, T)[name, t]
            add_cost_term_variant!(
                container, variable, rate, ShutDownCostExpression, T, name, t)
        end
    else
        cost_term = _shutdown_cost_value(get_shut_down(op_cost))
        iszero(cost_term) && return
        rate = cost_term * multiplier
        for t in get_time_steps(container)
            variable = get_variable(container, U, T)[name, t]
            add_cost_term_invariant!(
                container, variable, rate, ShutDownCostExpression, T, name, t)
        end
    end
end

#################################################################################
# Startup cost
#################################################################################

function add_start_up_cost!(
    container::OptimizationContainer,
    ::Type{U},
    devices::IS.FlattenIteratorWrapper{T},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    for d in devices
        # Function barrier: op_cost type becomes a compile-time parameter
        _add_start_up_cost_to_objective!(
            container, U, d, get_operation_cost(d), V)
    end
    return
end

function _add_start_up_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::C,
    op_cost::IS.DeviceParameter,
    ::Type{U},
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(T, U)
    get_must_run(component) && return
    name = get_name(component)
    if _is_time_series_cost(op_cost)
        param = get_parameter_array(container, StartupCostParameter, C)
        mult = get_parameter_multiplier_array(container, StartupCostParameter, C)
        for t in get_time_steps(container)
            # Broadcast so Tuple-valued parameters (for multi-start formulations) work
            # alongside Float64-valued ones.
            raw_startup_cost = param[name, t] .* mult[name, t]
            cost_term = start_up_cost(raw_startup_cost, C, T, U)
            iszero(cost_term) && continue
            rate = cost_term * multiplier
            variable = get_variable(container, T, C)[name, t]
            add_cost_term_variant!(
                container, variable, rate, StartUpCostExpression, C, name, t)
        end
    else
        raw_startup_cost = get_start_up(op_cost)
        for t in get_time_steps(container)
            cost_term = start_up_cost(raw_startup_cost, C, T, U)
            iszero(cost_term) && continue
            rate = cost_term * multiplier
            variable = get_variable(container, T, C)[name, t]
            add_cost_term_invariant!(
                container, variable, rate, StartUpCostExpression, C, name, t)
        end
    end
    return
end
