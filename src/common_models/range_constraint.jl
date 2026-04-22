######## CONSTRAINTS ############

get_bound_direction(::RangeConstraintLBExpressions) = LowerBound()
get_bound_direction(::RangeConstraintUBExpressions) = UpperBound()

@doc raw"""
Constructs min/max range constraint from device variable.


If min and max within an epsilon width:

``` variable[name, t] == limits.max ```

Otherwise:

``` limits.min <= variable[name, t] <= limits.max ```

where limits in constraint_infos.

# LaTeX

`` x = limits^{max}, \text{ for } |limits^{max} - limits^{min}| < \varepsilon ``

`` limits^{min} \leq x \leq limits^{max}, \text{ otherwise } ``
"""
function add_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: VariableType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_variable(container, U, V)
    _add_bound_range_constraints_impl!(container, T, LowerBound(), array, devices, model)
    _add_bound_range_constraints_impl!(container, T, UpperBound(), array, devices, model)
    return
end

function add_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: RangeConstraintLBExpressions,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_expression(container, U, V)
    _add_bound_range_constraints_impl!(container, T, LowerBound(), array, devices, model)
    return
end

function add_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: RangeConstraintUBExpressions,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_expression(container, U, V)
    _add_bound_range_constraints_impl!(container, T, UpperBound(), array, devices, model)
    return
end

function _add_bound_range_constraints_impl!(
    container::OptimizationContainer,
    ::Type{T},
    dir::BoundDirection,
    array,
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    T <: ConstraintType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    device_names = IS.get_name.(devices)
    jump_model = get_jump_model(container)

    con = add_constraints_container!(
        container, T, V, device_names, time_steps; meta = constraint_meta(dir))

    for device in devices, t in time_steps
        ci_name = IS.get_name(device)
        limits = get_min_max_limits(device, T, W)
        add_range_bound_constraint!(
            dir, jump_model, con, ci_name, t, array[ci_name, t], get_bound(dir, limits))
    end
    return
end

@doc raw"""
Constructs min/max range constraint from device variable and on/off decision variable.


If device min = 0:

``` varcts[name, t] <= limits.max*varbin[name, t]) ```

``` varcts[name, t] >= 0.0 ```

Otherwise:

``` varcts[name, t] <= limits.max*varbin[name, t] ```

``` varcts[name, t] >= limits.min*varbin[name, t] ```

where limits in constraint_infos.

# LaTeX

`` 0 \leq x^{cts} \leq limits^{max} x^{bin}, \text{ for } limits^{min} = 0 ``

`` limits^{min} x^{bin} \leq x^{cts} \leq limits^{max} x^{bin}, \text{ otherwise } ``
"""
function add_semicontinuous_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: VariableType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_variable(container, U, V)
    _add_semicontinuous_bound_range_constraints_impl!(
        container, T, LowerBound(), array, devices, model)
    _add_semicontinuous_bound_range_constraints_impl!(
        container, T, UpperBound(), array, devices, model)
    return
end

function add_semicontinuous_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: RangeConstraintLBExpressions,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_expression(container, U, V)
    _add_semicontinuous_bound_range_constraints_impl!(
        container, T, LowerBound(), array, devices, model)
    return
end

function add_semicontinuous_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: RangeConstraintUBExpressions,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_expression(container, U, V)
    _add_semicontinuous_bound_range_constraints_impl!(
        container, T, UpperBound(), array, devices, model)
    return
end

# Generic component version - always uses binary variable
function _add_semicontinuous_bound_range_constraints_impl!(
    container::OptimizationContainer,
    ::Type{T},
    dir::BoundDirection,
    array,
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    ::DeviceModel{V, W},
) where {
    T <: ConstraintType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    names = IS.get_name.(devices)
    jump_model = get_jump_model(container)
    con = add_constraints_container!(
        container, T, V, names, time_steps; meta = constraint_meta(dir))
    varbin = get_variable(container, OnVariable, V)

    for device in devices, t in time_steps
        ci_name = IS.get_name(device)
        limits = get_min_max_limits(device, T, W)
        add_range_bound_constraint!(
            dir, jump_model, con, ci_name, t,
            array[ci_name, t], get_bound(dir, limits), varbin[ci_name, t])
    end
    return
end

# Unified reserve range constraints impl
# invert_binary: true for InputActivePower (uses 1-varbin), false for others (uses varbin)
function add_reserve_bound_range_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    dir::BoundDirection,
    array,
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    ::DeviceModel{V, W},
    invert_binary::Bool,
) where {
    T <: ConstraintType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    names = IS.get_name.(devices)
    jump_model = get_jump_model(container)

    con = add_constraints_container!(
        container, T, V, names, time_steps; meta = constraint_meta(dir))
    varbin = get_variable(container, ReservationVariable, V)

    for device in devices, t in time_steps
        ci_name = IS.get_name(device)
        limits = get_min_max_limits(device, T, W)
        bin = invert_binary ? (1 - varbin[ci_name, t]) : varbin[ci_name, t]
        add_range_bound_constraint!(
            dir, jump_model, con, ci_name, t,
            array[ci_name, t], get_bound(dir, limits), bin)
    end
    return
end

function add_parameterized_bound_range_constraints(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{P},
    dir::BoundDirection,
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: ExpressionType,
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_expression(container, U, V)
    _add_parameterized_bound_range_constraints_impl!(
        container, T, dir, array, P, devices, model)
    return
end

function add_parameterized_bound_range_constraints(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{P},
    dir::BoundDirection,
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: VariableType,
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    array = get_variable(container, U, V)
    _add_parameterized_bound_range_constraints_impl!(
        container, T, dir, array, P, devices, model)
    return
end

# Backwards-compatible wrappers
function add_parameterized_lower_bound_range_constraints(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: Union{ExpressionType, VariableType},
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    add_parameterized_bound_range_constraints(
        container,
        T,
        U,
        P,
        LowerBound(),
        devices,
        model,
        X,
    )
    return
end

function add_parameterized_upper_bound_range_constraints(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
    ::Type{X},
) where {
    T <: ConstraintType,
    U <: Union{ExpressionType, VariableType},
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
    X <: AbstractPowerModel,
}
    add_parameterized_bound_range_constraints(
        container,
        T,
        U,
        P,
        UpperBound(),
        devices,
        model,
        X,
    )
    return
end

#######################################
######## Parameterized Bound Helpers ##
#######################################

# Internal unified implementation - dispatches on parameter type
function _add_parameterized_bound_range_constraints_impl!(
    container::OptimizationContainer,
    ::Type{T},
    dir::BoundDirection,
    array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    T <: ConstraintType,
    P <: TimeSeriesParameter,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    ts_name = get_time_series_names(model)[P]
    ts_type = get_default_time_series_type(container)
    # PERF: compilation hotspot. Switch to TSC.
    names = [IS.get_name(d) for d in devices if IS.has_time_series(d, ts_type, ts_name)]
    if isempty(names)
        @debug "There are no $V devices with time series data $ts_type, $ts_name"
        return
    end

    constraint = add_constraints_container!(
        container, T, V, names, time_steps; meta = constraint_meta(dir))

    _bound_range_with_parameter!(
        container, dir, constraint, array, P, devices, model)
    return
end

function _add_parameterized_bound_range_constraints_impl!(
    container::OptimizationContainer,
    ::Type{T},
    dir::BoundDirection,
    array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    T <: ConstraintType,
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    names = IS.get_name.(devices)
    constraint = add_constraints_container!(
        container, T, V, names, time_steps; meta = constraint_meta(dir))

    _bound_range_with_parameter!(
        container, dir, constraint, array, P, devices, model)
    return
end

# Unified internal function - generic ParameterType
function _bound_range_with_parameter!(
    container::OptimizationContainer,
    dir::BoundDirection,
    constraint_container::JuMPConstraintArray,
    lhs_array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    ::DeviceModel{V, W},
) where {
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    param_array = get_parameter_array(container, P, V)
    param_multiplier = get_parameter_multiplier_array(container, P, V)
    jump_model = get_jump_model(container)
    time_steps = axes(constraint_container)[2]
    for device in devices, t in time_steps
        name = IS.get_name(device)
        rhs = param_multiplier[name, t] * param_array[name, t]
        constraint_container[name, t] =
            _make_bound_constraint(dir, jump_model, lhs_array[name, t], rhs)
    end
    return
end

# FIXME better spot for this?
abstract type EventParameter <: ParameterType end

# originally was AvailableStatusParameter, but broadened to EventParameter to keep
# concrete types out of IOM.
function _bound_range_with_parameter!(
    container::OptimizationContainer,
    dir::BoundDirection,
    constraint_container::JuMPConstraintArray,
    lhs_array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    ::DeviceModel{V, W},
) where {
    P <: EventParameter,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    param_array = get_parameter_array(container, P, V)
    jump_model = get_jump_model(container)
    time_steps = axes(constraint_container)[2]
    for device in devices, t in time_steps
        ub = get_max_active_power(device)
        name = IS.get_name(device)
        rhs = ub * param_array[name, t]
        constraint_container[name, t] =
            _make_bound_constraint(dir, jump_model, lhs_array[name, t], rhs)
    end
    return
end

# TimeSeriesParameter variant - handles time series lookup
function _bound_range_with_parameter!(
    container::OptimizationContainer,
    dir::BoundDirection,
    constraint_container::JuMPConstraintArray,
    lhs_array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    P <: TimeSeriesParameter,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    param_container = get_parameter(container, P, V)
    mult = get_multiplier_array(param_container)
    jump_model = get_jump_model(container)
    time_steps = axes(constraint_container)[2]
    ts_name = get_time_series_names(model)[P]
    ts_type = get_default_time_series_type(container)
    for device in devices
        name = IS.get_name(device)
        if !(IS.has_time_series(device, ts_type, ts_name))
            continue
        end
        param_col = get_parameter_column_refs(param_container, name)
        for t in time_steps
            rhs = mult[name, t] * param_col[t]
            constraint_container[name, t] =
                _make_bound_constraint(dir, jump_model, lhs_array[name, t], rhs)
        end
    end
    return
end

# Backwards-compatible wrappers - re-used in SemiContinuousFeedforward
# TODO just call the unified function directly from POM? decide later.
function lower_bound_range_with_parameter!(
    container::OptimizationContainer,
    constraint_container::JuMPConstraintArray,
    lhs_array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    _bound_range_with_parameter!(
        container, LowerBound(), constraint_container, lhs_array, P, devices, model)
    return
end

function upper_bound_range_with_parameter!(
    container::OptimizationContainer,
    constraint_container::JuMPConstraintArray,
    lhs_array,
    ::Type{P},
    devices::Union{Vector{V}, IS.FlattenIteratorWrapper{V}},
    model::DeviceModel{V, W},
) where {
    P <: ParameterType,
    V <: IS.InfrastructureSystemsComponent,
    W <: AbstractDeviceFormulation,
}
    _bound_range_with_parameter!(
        container, UpperBound(), constraint_container, lhs_array, P, devices, model)
    return
end
