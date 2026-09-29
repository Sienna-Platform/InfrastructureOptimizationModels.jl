"""
Convert the internal `Dates.Millisecond` interval (where `UNSET_INTERVAL` means
unset) to the `Union{Nothing, Dates.Period}` form the IS / PSY time-series API
expects.
"""
_to_is_interval(interval::Dates.Millisecond) =
    interval == UNSET_INTERVAL ? nothing : interval

"""
Convert the internal `Dates.Millisecond` resolution (where `UNSET_RESOLUTION`
means unset) to the `Union{Nothing, Dates.Period}` form the IS / PSY
time-series API expects.
"""
_to_is_resolution(resolution::Dates.Millisecond) =
    resolution == UNSET_RESOLUTION ? nothing : resolution

function get_available_components(
    model::DeviceModel{T, <:AbstractDeviceFormulation},
    sys::IS.InfrastructureSystemsContainer,
) where {T <: IS.InfrastructureSystemsComponent}
    subsystem = get_subsystem(model)
    filter_function = get_attribute(model, "filter_function")
    if filter_function === nothing
        return IS.get_components(
            IS.get_available,
            T,
            sys;
            subsystem_name = subsystem,
        )
    else
        return IS.get_components(
            x -> IS.get_available(x) && filter_function(x),
            T,
            sys;
            subsystem_name = subsystem,
        )
    end
end

function get_available_components(
    model::ServiceModel{T, <:AbstractServiceFormulation},
    sys::IS.InfrastructureSystemsContainer,
) where {T <: IS.InfrastructureSystemsComponent}
    subsystem = get_subsystem(model)
    filter_function = get_attribute(model, "filter_function")
    if filter_function === nothing
        return IS.get_components(
            IS.get_available,
            T,
            sys;
            subsystem_name = subsystem,
        )
    else
        return IS.get_components(
            x -> IS.get_available(x) && filter_function(x),
            T,
            sys;
            subsystem_name = subsystem,
        )
    end
end

function get_available_components(
    model::NetworkModel,
    ::Type{T},
    sys::IS.InfrastructureSystemsContainer,
) where {T <: IS.InfrastructureSystemsComponent}
    subsystem = get_subsystem(model)
    return get_subsystem_components(T, sys; subsystem_name = subsystem)
end

##################################################
########### Cost Function Utilities ##############
##################################################

# Cost curves are in natural units (x in MW); one system per-unit of power is
# `system_base_power` MW.

"""
Proportional (slope) cost coefficient normalized to system base.
"""
get_proportional_cost_per_system_unit(cost_term::Float64, system_base_power::Float64) =
    cost_term * system_base_power

"""
Quadratic cost coefficient normalized to system base.
"""
get_quadratic_cost_per_system_unit(cost_term::Float64, system_base_power::Float64) =
    cost_term * system_base_power^2

"""
PiecewiseLinearData normalized to system base. x-coords (MW) become system per-unit;
y-coords (\$/h) do not depend on the power base.
"""
get_piecewise_pointcurve_per_system_unit(
    cost_component::IS.PiecewiseLinearData,
    system_base_power::Float64,
) = IS.PiecewiseLinearData([
    (x = point.x / system_base_power, y = point.y) for
    point in IS.get_points(cost_component)
])

"""
PiecewiseStepData normalized to system base. x-coords (MW) become system per-unit;
y-coords (\$/MWh) scale up by the same factor.
"""
get_piecewise_curve_per_system_unit(
    cost_component::IS.PiecewiseStepData,
    system_base_power::Float64,
) = IS.PiecewiseStepData(
    get_piecewise_curve_per_system_unit(
        IS.get_x_coords(cost_component),
        IS.get_y_coords(cost_component),
        system_base_power,
    )...,
)

get_piecewise_curve_per_system_unit(
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    system_base_power::Float64,
) = (x_coords ./ system_base_power, y_coords .* system_base_power)

is_time_variant(x) = IS.is_time_series_backed(x)

function get_forecast_intervals(sys::IS.InfrastructureSystemsContainer)
    table = get_forecast_summary_table(sys)
    return Set(row.interval for row in eachrow(table) if row.interval !== nothing)
end

function auto_transform_time_series!(
    sys::IS.InfrastructureSystemsContainer,
    settings::Settings,
)
    model_interval = get_interval(settings)
    model_horizon = get_horizon(settings)
    if model_interval == UNSET_INTERVAL || model_horizon == UNSET_HORIZON
        return
    end

    counts = get_time_series_counts(sys)
    if counts.static_time_series_count < 1
        return
    end
    if counts.forecast_count > 0 && model_interval in get_forecast_intervals(sys)
        return
    end

    model_resolution = get_resolution(settings)
    resolution_kwarg =
        model_resolution == UNSET_RESOLUTION ? (;) : (; resolution = model_resolution)

    @info "Auto-transforming SingleTimeSeries to DeterministicSingleTimeSeries" horizon =
        Dates.canonicalize(model_horizon) interval = Dates.canonicalize(model_interval)
    transform_single_time_series!(
        sys,
        model_horizon,
        model_interval;
        delete_existing = false,
        resolution_kwarg...,
    )
    return
end
