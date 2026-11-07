function _check_service_formulation(::Type{D}) where {D}
    # Reject genuinely abstract supertypes (e.g. `Reserve`). A parametric component
    # family that left a trailing type parameter free — such as
    # `ReserveDemandCurve{ReserveUp}` once PSY parameterized it on a unit-system type —
    # is *not* concrete but *is* a valid dispatch/selection key, so it must be allowed.
    if isabstracttype(D)
        throw(
            ArgumentError(
                "The service model must contain only concrete types, $(D) is an Abstract Type",
            ),
        )
    end
end

"""
Establishes the model for all services of a particular type. A `ServiceModel` represents
every service of its type in the system. Uses the keyword argument `feedforwards` to
enable passing values between operation models at simulation time.

# Arguments

-`::Type{D}`: Power System Service Type
-`::Type{B}`: Abstract Service Formulation

# Accepted Key Words

  - `feedforwards::Vector{<:AbstractAffectFeedforward}` : use to pass parameters between models
  - `slack_penalty::Float64 = SERVICES_SLACK_COST` : Objective cost of one unit of slack per hour, applied when `use_slacks` is `true`

# Example

reserves = ServiceModel(PSY.VariableReserve{PSY.ReserveUp}, RangeReserve)
"""
mutable struct ServiceModel{D <: IS.InfrastructureSystemsComponent, B}
    # Heterogeneous by design: concrete Vector of the abstract type, not a UnionAll field.
    feedforwards::Vector{AbstractAffectFeedforward}
    use_slacks::Bool
    slack_penalty::Float64
    duals::Vector{DataType}
    time_series_names::Dict{Type{<:TimeSeriesParameter}, String}
    attributes::Dict{String, Any}
    # Per service: service name -> device type -> contributing devices.
    contributing_devices_map::Dict{
        String,
        Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}},
    }
    subsystem::Union{Nothing, String}
    # Maps outage UUIDs to monitored components grouped by device type. PNM indexes DF matrices with UUIDs.
    outages::Dict{Int, Dict{DataType, Set{String}}}
    function ServiceModel(
        ::Type{D},
        ::Type{B};
        use_slacks = false,
        slack_penalty::Float64 = SERVICES_SLACK_COST,
        feedforwards = Vector{AbstractAffectFeedforward}(),
        duals = Vector{DataType}(),
        time_series_names = get_default_time_series_names(D, B),
        attributes = Dict{String, Any}(),
        contributing_devices_map = Dict{
            String,
            Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}},
        }(),
    ) where {D <: IS.InfrastructureSystemsComponent, B}
        attributes_for_model = get_default_attributes(D, B)
        for (k, v) in attributes
            attributes_for_model[k] = v
        end

        _check_service_formulation(D)
        _check_service_formulation(B)
        new{D, B}(
            convert(Vector{AbstractAffectFeedforward}, feedforwards),
            use_slacks,
            slack_penalty,
            duals,
            time_series_names,
            attributes_for_model,
            contributing_devices_map,
            nothing,
            Dict{Int, Dict{DataType, Set{String}}}(),
        )
    end
end

get_component_type(
    ::ServiceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B} = D
get_formulation(
    ::ServiceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B} = B
get_feedforwards(m::ServiceModel) = m.feedforwards
get_use_slacks(m::ServiceModel) = m.use_slacks
get_slack_penalty(m::ServiceModel) = m.slack_penalty
get_duals(m::ServiceModel) = m.duals
get_time_series_names(m::ServiceModel) = m.time_series_names
get_attributes(m::ServiceModel) = m.attributes
get_attribute(m::ServiceModel, key::String) = get(m.attributes, key, nothing)
# Whole nested map: service name -> device type -> contributing devices.
get_contributing_devices_map(m::ServiceModel) = m.contributing_devices_map
# Returned for a service with no entry in the map. Callers treat it as read-only (shared).
const _EMPTY_CONTRIBUTING_DEVICES_MAP =
    Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}}()
# One service's inner `device type -> devices` map (the empty const if the service is absent).
get_contributing_devices_map(m::ServiceModel, service_name::AbstractString) =
    get(m.contributing_devices_map, service_name, _EMPTY_CONTRIBUTING_DEVICES_MAP)
# The flatten accessors below mix device types, so the element type widens.
# Build code must iterate `get_contributing_devices_map` groups instead.
get_contributing_devices(m::ServiceModel) =
    [z for inner in values(m.contributing_devices_map) for x in values(inner) for z in x]
get_contributing_devices(m::ServiceModel, service_name::AbstractString) =
    [z for x in values(get_contributing_devices_map(m, service_name)) for z in x]

"""
Return the device types that contribute to the service `service_name`.
Return an empty collection if the service has no entry.
"""
contributing_device_types(m::ServiceModel, service_name::AbstractString) =
    keys(get_contributing_devices_map(m, service_name))

"""
Add `device` to the contributing devices of the service `service_name`.
The function creates the entry for the service and for the type `D` if they do not exist.
Each device type keeps its own `Vector{D}`.
"""
function add_contributing_device!(
    m::ServiceModel,
    service_name::AbstractString,
    device::D,
) where {D <: IS.InfrastructureSystemsComponent}
    inner = get!(
        Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}},
        m.contributing_devices_map,
        String(service_name),
    )
    devices = get!(Vector{D}, inner, D)::Vector{D}
    push!(devices, device)
    return
end
get_subsystem(m::ServiceModel) = m.subsystem
get_outages(m::ServiceModel) = m.outages

set_subsystem!(m::ServiceModel, id::String) = m.subsystem = id

function set_model!(dict::Dict, key::Symbol, model::ServiceModel)
    if haskey(dict, key)
        @warn "Overwriting $(key) existing model"
    end
    dict[key] = model
    return
end

function set_model!(
    dict::Dict,
    model::ServiceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B}
    set_model!(dict, Symbol(D), model)
    return
end
