# NOTE not included currently.
function _get_time_series(
    container::OptimizationContainer,
    component::PSY.Component,
    attributes::TimeSeriesAttributes{T};
    interval::Dates.Millisecond = UNSET_INTERVAL,
) where {T <: IS.TimeSeriesData}
    return get_time_series_initial_values!(
        container,
        T,
        component,
        get_time_series_name(attributes);
        interval = interval,
    )
end

function get_time_series(
    container::OptimizationContainer,
    component::T,
    ::Type{P},
    meta = CONTAINER_KEY_EMPTY_META;
    interval::Dates.Millisecond = UNSET_INTERVAL,
) where {T <: PSY.Component, P <: TimeSeriesParameter}
    parameter_container = get_parameter(container, P, T, meta)
    return _get_time_series(
        container,
        component,
        parameter_container.attributes;
        interval = interval,
    )
end

# This is just for temporary compatibility with current code. Needs to be eliminated once the time series
# refactor is done.
function get_time_series(
    container::OptimizationContainer,
    component::PSY.Component,
    forecast_name::String;
    interval::Dates.Millisecond = UNSET_INTERVAL,
)
    ts_type = get_default_time_series_type(container)
    return _get_time_series(
        container,
        component,
        TimeSeriesAttributes(ts_type, forecast_name);
        interval = interval,
    )
end

# TODO better place for this? I'm reluctant to put it in interfaces.jl, because
# it actually defines some meaningful behavior.
"""
Extension point: Get default time series name for a parameter type.
Returns the name of the time series to look for in the component.
"""
function get_time_series_name(
    ::T,
    d::U,
    model::DeviceModel{U, F},
) where {
    T <: TimeSeriesParameter,
    U <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    # Check if model has time series names configured
    ts_names = get_time_series_names(model)
    if haskey(ts_names, T)
        return ts_names[T]
    end

    # Default: use parameter type name without "TimeSeriesParameter" suffix
    param_name = string(T)
    return replace(param_name, "TimeSeriesParameter" => "")
end
