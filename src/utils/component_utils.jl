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
    return IS.get_components(
        T,
        sys;
        subsystem_name = subsystem,
    )
end

##################################################
########### Cost Function Utilities ##############
##################################################

"""
Obtain proportional (marginal or slope) cost data in system base per unit
depending on the specified power units
"""
function get_proportional_cost_per_system_unit(
    cost_term::Float64,
    unit_system::IS.UnitSystem,
    system_base_power::Float64,
    device_base_power::Float64,
)
    return _get_proportional_cost_per_system_unit(
        cost_term,
        Val{unit_system}(),
        system_base_power,
        device_base_power,
    )
end

function _get_proportional_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.SYSTEM_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term
end

function _get_proportional_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.DEVICE_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term * (system_base_power / device_base_power)
end

function _get_proportional_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.NATURAL_UNITS},
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term * system_base_power
end

"""
Obtain quadratic cost data in system base per unit
depending on the specified power units
"""
function get_quadratic_cost_per_system_unit(
    cost_term::Float64,
    unit_system::IS.UnitSystem,
    system_base_power::Float64,
    device_base_power::Float64,
)
    return _get_quadratic_cost_per_system_unit(
        cost_term,
        Val{unit_system}(),
        system_base_power,
        device_base_power,
    )
end

function _get_quadratic_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.SYSTEM_BASE}, # SystemBase Unit
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term
end

function _get_quadratic_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.DEVICE_BASE}, # DeviceBase Unit
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term * (system_base_power / device_base_power)^2
end

function _get_quadratic_cost_per_system_unit(
    cost_term::Float64,
    ::Val{IS.UnitSystem.NATURAL_UNITS}, # Natural Units
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_term * system_base_power^2
end

"""
Obtain the normalized PiecewiseLinear cost data in system base per unit
depending on the specified power units.

Note that the costs (y-axis) are always in \$/h so
they do not require transformation
"""
function get_piecewise_pointcurve_per_system_unit(
    cost_component::IS.PiecewiseLinearData,
    unit_system::IS.UnitSystem,
    system_base_power::Float64,
    device_base_power::Float64,
)
    return _get_piecewise_pointcurve_per_system_unit(
        cost_component,
        Val{unit_system}(),
        system_base_power,
        device_base_power,
    )
end

function _get_piecewise_pointcurve_per_system_unit(
    cost_component::IS.PiecewiseLinearData,
    ::Val{IS.UnitSystem.SYSTEM_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    return cost_component
end

function _get_piecewise_pointcurve_per_system_unit(
    cost_component::IS.PiecewiseLinearData,
    ::Val{IS.UnitSystem.DEVICE_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    points = cost_component.points
    points_normalized = Vector{NamedTuple{(:x, :y)}}(undef, length(points))
    for (ix, point) in enumerate(points)
        points_normalized[ix] =
            (x = point.x * (device_base_power / system_base_power), y = point.y)
    end
    return IS.PiecewiseLinearData(points_normalized)
end

function _get_piecewise_pointcurve_per_system_unit(
    cost_component::IS.PiecewiseLinearData,
    ::Val{IS.UnitSystem.NATURAL_UNITS},
    system_base_power::Float64,
    device_base_power::Float64,
)
    points = cost_component.points
    points_normalized = Vector{NamedTuple{(:x, :y)}}(undef, length(points))
    for (ix, point) in enumerate(points)
        points_normalized[ix] = (x = point.x / system_base_power, y = point.y)
    end
    return IS.PiecewiseLinearData(points_normalized)
end

"""
Obtain the normalized PiecewiseStepData in system base per unit depending on the specified
power units.

Note that the costs (y-axis) are in \$/MWh, \$/(sys pu h) or \$/(device pu h), so they also
require transformation.
"""
function get_piecewise_curve_per_system_unit(
    cost_component::IS.PiecewiseStepData,
    unit_system::IS.UnitSystem,
    system_base_power::Float64,
    device_base_power::Float64,
)
    return IS.PiecewiseStepData(
        get_piecewise_curve_per_system_unit(
            IS.get_x_coords(cost_component),
            IS.get_y_coords(cost_component),
            unit_system,
            system_base_power,
            device_base_power,
        )...,
    )
end

function get_piecewise_curve_per_system_unit(
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    unit_system::IS.UnitSystem,
    system_base_power::Float64,
    device_base_power::Float64,
)
    return _get_piecewise_curve_per_system_unit(
        x_coords,
        y_coords,
        Val{unit_system}(),
        system_base_power,
        device_base_power,
    )
end

function _get_piecewise_curve_per_system_unit(
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    ::Val{IS.UnitSystem.SYSTEM_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    return x_coords, y_coords
end

function _get_piecewise_curve_per_system_unit(
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    ::Val{IS.UnitSystem.DEVICE_BASE},
    system_base_power::Float64,
    device_base_power::Float64,
)
    ratio = device_base_power / system_base_power
    x_coords_normalized = x_coords .* ratio
    y_coords_normalized = y_coords ./ ratio
    return x_coords_normalized, y_coords_normalized
end

function _get_piecewise_curve_per_system_unit(
    x_coords::AbstractVector,
    y_coords::AbstractVector,
    ::Val{IS.UnitSystem.NATURAL_UNITS},
    system_base_power::Float64,
    device_base_power::Float64,
)
    x_coords_normalized = x_coords ./ system_base_power
    y_coords_normalized = y_coords .* system_base_power
    return x_coords_normalized, y_coords_normalized
end

is_time_variant(::IS.TimeSeriesKey) = true
is_time_variant(::IS.ValueCurve{<:IS.TimeSeriesFunctionData}) = true
is_time_variant(
    ::IS.ProductionVariableCostCurve{<:IS.ValueCurve{<:IS.TimeSeriesFunctionData}},
) = true
is_time_variant(::IS.TupleTimeSeries) = true
is_time_variant(::Any) = false
