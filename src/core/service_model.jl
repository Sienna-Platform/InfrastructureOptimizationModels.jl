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
every service of its type in the system; it has no single service name. Uses the keyword
argument `feedforwards` to enable passing values between operation models at simulation
time.

# Arguments

-`::Type{D}`: Power System Service Type
-`::Type{B}`: Abstract Service Formulation

# Accepted Key Words

  - `feedforwards::Vector{<:AbstractAffectFeedforward}` : use to pass parameters between models

# Example

reserves = ServiceModel(PSY.VariableReserve{PSY.ReserveUp}, RangeReserve)
"""
mutable struct ServiceModel{D <: IS.InfrastructureSystemsComponent, B}
    # Heterogeneous by design: concrete Vector of the abstract type, not a UnionAll field.
    feedforwards::Vector{AbstractAffectFeedforward}
    use_slacks::Bool
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
    outages::Dict{Base.UUID, Dict{DataType, Set{String}}}
    function ServiceModel(
        ::Type{D},
        ::Type{B};
        use_slacks = false,
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
        if !haskey(attributes_for_model, "aggregated_service_model")
            push!(attributes_for_model, "aggregated_service_model" => true)
        end

        _check_service_formulation(D)
        _check_service_formulation(B)
        new{D, B}(
            convert(Vector{AbstractAffectFeedforward}, feedforwards),
            use_slacks,
            duals,
            time_series_names,
            attributes_for_model,
            contributing_devices_map,
            nothing,
            Dict{Base.UUID, Dict{DataType, Set{String}}}(),
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
get_duals(m::ServiceModel) = m.duals
get_time_series_names(m::ServiceModel) = m.time_series_names
get_attributes(m::ServiceModel) = m.attributes
get_attribute(m::ServiceModel, key::String) = get(m.attributes, key, nothing)
# Whole nested map: service name -> device type -> contributing devices.
get_contributing_devices_map(m::ServiceModel) = m.contributing_devices_map
# One service's inner `Dict{DataType, Vector}` (empty Dict if the service is absent).
get_contributing_devices_map(m::ServiceModel, service_name::AbstractString) =
    get(
        m.contributing_devices_map,
        service_name,
        Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}}(),
    )
# All contributing devices across ALL services (flatten the nested map).
get_contributing_devices(m::ServiceModel) =
    [z for inner in values(m.contributing_devices_map) for x in values(inner) for z in x]
# One service's contributing devices (flattened Vector).
get_contributing_devices(m::ServiceModel, service_name::AbstractString) =
    [z for x in values(get_contributing_devices_map(m, service_name)) for z in x]
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
