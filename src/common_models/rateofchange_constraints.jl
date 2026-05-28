function _get_minutes_per_period(container::OptimizationContainer)
    resolution = get_resolution(container)
    if resolution > Dates.Minute(1)
        minutes_per_period = Dates.value(Dates.Minute(resolution))
    else
        @warn("Not all formulations support under 1-minute resolutions. Exercise caution.")
        minutes_per_period = Dates.value(Dates.Second(resolution)) / 60
    end
    return minutes_per_period
end

function _get_ramp_constraint_devices(
    container::OptimizationContainer,
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
) where {U <: IS.InfrastructureSystemsComponent}
    minutes_per_period = _get_minutes_per_period(container)
    filtered_device = Vector{U}()
    for d in devices
        ramp_limits = get_ramp_limits(d)
        if ramp_limits !== nothing
            p_lims = get_active_power_limits(d)
            max_rate = abs(p_lims.min - p_lims.max) / minutes_per_period
            if (ramp_limits.up >= max_rate) & (ramp_limits.down >= max_rate)
                @debug "Generator has a nonbinding ramp limits. Constraints Skipped" IS.get_name(
                    d,
                )
                continue
            else
                push!(filtered_device, d)
            end
        end
    end
    return filtered_device
end

function _get_ramp_slack_vars(
    container::OptimizationContainer,
    model::DeviceModel{V, W},
    name::String,
    t::Int,
) where {V <: IS.InfrastructureSystemsComponent, W <: AbstractDeviceFormulation}
    if get_use_slacks(model)
        slack_up = get_variable(container, RateofChangeConstraintSlackUp, V)
        slack_dn = get_variable(container, RateofChangeConstraintSlackDown, V)
        return (up = slack_up[name, t], down = slack_dn[name, t])
    else
        return (up = 0.0, down = 0.0)
    end
end

@doc raw"""
Constructs allowed rate-of-change constraints from variables, initial condtions, and rate data.


If t = 1:

``` variable[name, 1] - initial_conditions[ix].value <= rate_data[1][ix].up ```

``` initial_conditions[ix].value - variable[name, 1] <= rate_data[1][ix].down ```

If t > 1:

``` variable[name, t] - variable[name, t-1] <= rate_data[1][ix].up ```

``` variable[name, t-1] - variable[name, t] <= rate_data[1][ix].down ```

# LaTeX

`` r^{down} \leq x_1 - x_{init} \leq r^{up}, \text{ for } t = 1 ``

`` r^{down} \leq x_t - x_{t-1} \leq r^{up}, \forall t \geq 2 ``

"""
function add_linear_ramp_constraints!(
    container::OptimizationContainer,
    T::Type{<:ConstraintType},
    U::Type{S},
    devices::IS.FlattenIteratorWrapper{V},
    model::DeviceModel{V, W},
    ::Type{<:AbstractPowerModel},
) where {
    S <: Union{PowerAboveMinimumVariable, ActivePowerVariable},
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    # common setup for all ramp constraints
    time_steps = get_time_steps(container)
    variable = get_variable(container, U, V)
    ramp_devices = _get_ramp_constraint_devices(container, devices)
    minutes_per_period = _get_minutes_per_period(container)
    IC = _get_initial_condition_type(T, V, W)
    initial_conditions_power = get_initial_condition(container, IC, V)
    jump_model = get_jump_model(container)
    device_name_set = IS.get_name.(ramp_devices)
    cons = add_updown_constraints_containers!(container, T, V, device_name_set, time_steps)

    expr_dn = get_expression(container, ActivePowerRangeExpressionLB, V)
    expr_up = get_expression(container, ActivePowerRangeExpressionUB, V)

    for ic in initial_conditions_power
        name = get_component_name(ic)
        # This is to filter out devices that dont need a ramping constraint
        name ∉ device_name_set && continue
        ramp_limits = get_ramp_limits(get_component(ic))
        ic_power = get_value(ic)
        @debug "add rate_of_change_constraint" name ic_power

        slack = _get_ramp_slack_vars(container, model, name, 1)
        add_ramp_constraint_pair!(
            jump_model, cons, name, 1,
            (up = expr_up[name, 1], down = expr_dn[name, 1]),
            ic_power, ramp_limits, minutes_per_period, slack, false,
        )

        for t in time_steps[2:end]
            slack = _get_ramp_slack_vars(container, model, name, t)
            add_ramp_constraint_pair!(
                jump_model, cons, name, t,
                (up = expr_up[name, t], down = expr_dn[name, t]),
                variable[name, t - 1], ramp_limits, minutes_per_period, slack,
                false)
        end
    end
    return
end

# Helper function containing the shared ramp constraint logic
function _add_linear_ramp_constraints_impl!(
    container::OptimizationContainer,
    T::Type{<:ConstraintType},
    U::Type{<:VariableType},
    devices::IS.FlattenIteratorWrapper{V},
    model::DeviceModel{V, W},
) where {V <: IS.InfrastructureSystemsComponent, W <: AbstractDeviceFormulation}
    # common setup for all ramp constraints
    time_steps = get_time_steps(container)
    variable = get_variable(container, U, V)
    ramp_devices = _get_ramp_constraint_devices(container, devices)
    minutes_per_period = _get_minutes_per_period(container)
    IC = _get_initial_condition_type(T, V, W)
    initial_conditions_power = get_initial_condition(container, IC, V)
    jump_model = get_jump_model(container)
    device_name_set = IS.get_name.(ramp_devices)
    cons = add_updown_constraints_containers!(container, T, V, device_name_set, time_steps)

    parameters = built_for_recurrent_solves(container)

    for ic in initial_conditions_power
        name = get_component_name(ic)
        # This is to filter out devices that dont need a ramping constraint
        name ∉ device_name_set && continue
        ramp_limits = get_ramp_limits(get_component(ic))
        ic_power = get_value(ic)
        @debug "add rate_of_change_constraint" name ic_power
        @assert (parameters && isa(ic_power, JuMP.VariableRef)) || !parameters

        slack = _get_ramp_slack_vars(container, model, name, 1)
        cur = (up = variable[name, 1], down = variable[name, 1])
        add_ramp_constraint_pair!(
            jump_model, cons, name, 1,
            cur, ic_power, ramp_limits, minutes_per_period, slack)

        for t in time_steps[2:end]
            slack = _get_ramp_slack_vars(container, model, name, t)
            cur = (up = variable[name, t], down = variable[name, t])
            add_ramp_constraint_pair!(
                jump_model, cons, name, t,
                cur, variable[name, t - 1], ramp_limits, minutes_per_period, slack)
        end
    end
    return
end

function add_linear_ramp_constraints!(
    container::OptimizationContainer,
    T::Type{<:ConstraintType},
    U::Type{<:VariableType},
    devices::IS.FlattenIteratorWrapper{V},
    model::DeviceModel{V, W},
    X::Type{<:AbstractPowerModel},
) where {V <: IS.InfrastructureSystemsComponent, W <: AbstractDeviceFormulation}
    return _add_linear_ramp_constraints_impl!(container, T, U, devices, model)
end

# TODO thermal-specific: move to POM?
abstract type AbstractThermalDispatchFormulation <: AbstractThermalFormulation end
abstract type AbstractThermalUnitCommitment <: AbstractThermalFormulation end

function add_linear_ramp_constraints!(
    container::OptimizationContainer,
    T::Type{<:ConstraintType},
    U::Type{ActivePowerVariable},
    devices::IS.FlattenIteratorWrapper{V},
    model::DeviceModel{V, W},
    X::Type{<:AbstractPowerModel},
) where {V <: IS.InfrastructureSystemsComponent, W <: AbstractThermalDispatchFormulation}

    # Fallback to generic implementation if OnStatusParameter is not present
    if !has_container_key(container, OnStatusParameter, V)
        return _add_linear_ramp_constraints_impl!(container, T, U, devices, model)
    end

    # common setup for all ramp constraints
    time_steps = get_time_steps(container)
    variable = get_variable(container, U, V)
    ramp_devices = _get_ramp_constraint_devices(container, devices)
    minutes_per_period = _get_minutes_per_period(container)
    IC = _get_initial_condition_type(T, V, W)
    initial_conditions_power = get_initial_condition(container, IC, V)
    jump_model = get_jump_model(container)
    device_name_set = [IS.get_name(r) for r in ramp_devices]
    cons = add_updown_constraints_containers!(container, T, V, device_name_set, time_steps)

    # Commitment path from UC as a PARAMETER (fixed 0/1)
    on_param = get_parameter(container, OnStatusParameter, V)
    on_status = on_param.parameter_array  # on_status[name, t] ∈ {0,1} (fixed)

    ic_power_by_name = Dict(
        get_component_name(ic) => get_value(ic) for ic in initial_conditions_power
    )

    for dev in ramp_devices
        name = IS.get_name(dev)
        ramp_limits = get_ramp_limits(dev)
        power_limits = get_active_power_limits(dev)

        # --- t = 1: Use ic_power to determine starting ramp condition
        ic_power = ic_power_by_name[name]
        ycur = on_status[name, 1]
        slack = _get_ramp_slack_vars(container, model, name, 1)
        big_m = power_limits.max * (1 - ycur)
        startstop = (up = big_m, down = big_m)
        cur = (up = variable[name, 1], down = variable[name, 1])
        add_ramp_constraint_startstop_pair!(
            jump_model, cons, name, 1,
            cur, ic_power, ramp_limits, minutes_per_period, startstop, slack)

        # --- t ≥ 2: gate by previous status y_{t-1}
        for t in time_steps[2:end]
            yprev = on_status[name, t - 1]   # 0/1 fixed from UC
            ycur = on_status[name, t]       # 0/1 fixed from UC
            slack = _get_ramp_slack_vars(container, model, name, t)
            big_m = power_limits.max * (2 - yprev - ycur)
            startstop = (up = big_m, down = big_m)
            cur = (up = variable[name, t], down = variable[name, t])
            add_ramp_constraint_startstop_pair!(
                jump_model, cons, name, t,
                cur, variable[name, t - 1], ramp_limits, minutes_per_period, startstop,
                slack)
        end
    end

    return
end

@doc raw"""
Constructs allowed rate-of-change constraints from variables, initial condtions, start/stop status, and rate data

# Equations
If t = 1:

``` variable[name, 1] - initial_conditions[ix].value <= rate_data[1][ix].up + rate_data[2][ix].max*varstart[name, 1] ```

``` initial_conditions[ix].value - variable[name, 1] <= rate_data[1][ix].down + rate_data[2][ix].min*varstop[name, 1] ```

If t > 1:

``` variable[name, t] - variable[name, t-1] <= rate_data[1][ix].up + rate_data[2][ix].max*varstart[name, t] ```

``` variable[name, t-1] - variable[name, t] <= rate_data[1][ix].down + rate_data[2][ix].min*varstop[name, t] ```

# LaTeX

`` r^{down} + r^{min} x^{stop}_1 \leq x_1 - x_{init} \leq r^{up} + r^{max} x^{start}_1, \text{ for } t = 1 ``

`` r^{down} + r^{min} x^{stop}_t \leq x_t - x_{t-1} \leq r^{up} + r^{max} x^{start}_t, \forall t \geq 2 ``
"""
function add_semicontinuous_ramp_constraints!(
    container::OptimizationContainer,
    T::Type{<:ConstraintType},
    U::Type{S},
    devices::IS.FlattenIteratorWrapper{V},
    model::DeviceModel{V, W},
    ::Type{<:AbstractPowerModel},
) where {
    S <: Union{PowerAboveMinimumVariable, ActivePowerVariable},
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    # common setup for all ramp constraints
    time_steps = get_time_steps(container)
    variable = get_variable(container, U, V)
    ramp_devices = _get_ramp_constraint_devices(container, devices)
    minutes_per_period = _get_minutes_per_period(container)
    IC = _get_initial_condition_type(T, V, W)
    initial_conditions_power = get_initial_condition(container, IC, V)
    jump_model = get_jump_model(container)
    device_name_set = IS.get_name.(ramp_devices)
    cons = add_updown_constraints_containers!(container, T, V, device_name_set, time_steps)

    varstart = get_variable(container, StartVariable, V)
    varstop = get_variable(container, StopVariable, V)
    expr_dn = get_expression(container, ActivePowerRangeExpressionLB, V)
    expr_up = get_expression(container, ActivePowerRangeExpressionUB, V)

    for ic in initial_conditions_power
        name = get_component_name(ic)
        # This is to filter out devices that dont need a ramping constraint
        name ∉ device_name_set && continue
        device = get_component(ic)
        ramp_limits = get_ramp_limits(device)
        power_limits = get_active_power_limits(device)
        ic_power = get_value(ic)
        @debug "add rate_of_change_constraint" name ic_power

        must_run = hasmethod(get_must_run, Tuple{V}) && get_must_run(device)

        for t in time_steps
            slack = _get_ramp_slack_vars(container, model, name, t)
            prev = t == 1 ? ic_power : variable[name, t - 1]
            cur = (up = expr_up[name, t], down = expr_dn[name, t])
            if must_run
                add_ramp_constraint_pair!(
                    jump_model, cons, name, t,
                    cur, prev, ramp_limits, minutes_per_period, slack)
            else
                startstop = (
                    up = power_limits.min * varstart[name, t],
                    down = power_limits.min * varstop[name, t],
                )
                add_ramp_constraint_startstop_pair!(
                    jump_model, cons, name, t,
                    cur, prev, ramp_limits, minutes_per_period, startstop, slack)
            end
        end
    end
    return
end
