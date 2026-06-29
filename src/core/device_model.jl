"""
Formulation type to augment the power balance constraint expression with a time series parameter
"""
struct FixedOutput <: AbstractDeviceFormulation end

"""
Domain-neutral supertype for the per-device contingency-event model. The concrete,
PowerSystems-specific `EventModel{<:Contingency, <:AbstractEventCondition}` lives in
`PowerOperationsModels`; IOM only needs the abstract to type the `DeviceModel.events`
field (mirroring how `AbstractAffectFeedforward` types the `feedforwards` field).
"""
abstract type AbstractEventModel end

"""
Domain-neutral supertype for the key under which an `AbstractEventModel` is stored in a
`DeviceModel`. The concrete `EventKey{<:Contingency, <:Component}` lives in
`PowerOperationsModels`.
"""
abstract type AbstractEventKey end

function _check_device_formulation(
    ::Type{D},
) where {D <: Union{AbstractDeviceFormulation, IS.InfrastructureSystemsComponent}}
    if !isconcretetype(D)
        throw(
            ArgumentError(
                "The device model must contain only concrete types, $(D) is an Abstract Type",
            ),
        )
    end
end

"""
    DeviceModel(
        ::Type{D},
        ::Type{B};
        feedforwards = Vector{AbstractAffectFeedforward}(),
        use_slacks = false,
        duals = Vector{DataType}(),
        time_series_names = get_default_time_series_names(D, B),
        attributes = Dict{String, Any}(),
        outages = IS.InfrastructureSystemsComponent[],
    )

Establishes the model for a particular device specified by type. Uses the keyword argument
`feedforwards` to enable passing values between operation models at simulation time.

# Arguments

  - `::Type{D}`: Device Type (e.g., PSY.ThermalStandard or a mock device type)
  - `::Type{B} where B<:AbstractDeviceFormulation`: Abstract Device Formulation
  - `feedforwards::Vector{<:AbstractAffectFeedforward} = Vector{AbstractAffectFeedforward}()` : use to pass parameters between models
  - `use_slacks::Bool = false` : Add slacks to the device model. Implementation is model dependent and not all models feature slacks
  - `duals::Vector{DataType} = Vector{DataType}()`: use to pass constraint type to calculate the duals. The DataType needs to be a valid ConstraintType
  - `time_series_names::Dict{Type{<:TimeSeriesParameter}, String} = get_default_time_series_names(D, B)` : use to specify time series names associated to the device`
  - `attributes::Dict{String, Any} = get_default_attributes(D, B)` : use to specify attributes to the device
  - `outages::AbstractVector{<:IS.InfrastructureSystemsComponent} = IS.InfrastructureSystemsComponent[]` :
    N-1 contingencies to model when the formulation is security-constrained. The
    constructor stores the `IS.get_uuid(outage)` of each entry as a key in the model's
    `outages::Dict{UUID, Dict{DataType, Set{String}}}` field with empty inner maps;
    template validation in downstream packages fills the inner maps with the per-type
    set of monitored component names that each outage carries. Power-specific
    validation (e.g. checking that entries are `PSY.Outage` subtypes) lives in
    `PowerOperationsModels`. If `B` is not security-constrained, a non-empty value is
    dropped with a warning.

# Example
```julia
thermal_gens = DeviceModel(ThermalStandard, ThermalBasicUnitCommitment)
```
"""
mutable struct DeviceModel{
    D <: IS.InfrastructureSystemsComponent,
    B <: AbstractDeviceFormulation,
}
    # Heterogeneous by design: concrete Vector of the abstract type, not a UnionAll field.
    feedforwards::Vector{AbstractAffectFeedforward}
    use_slacks::Bool
    duals::Vector{DataType}
    services::Vector{ServiceModel}
    time_series_names::Dict{Type{<:ParameterType}, String}
    attributes::Dict{String, Any}
    subsystem::Union{Nothing, String}
    # Contingency-event models keyed by an `AbstractEventKey`. Concrete key/value types
    # are power-specific and live in POM; stored abstractly here like `feedforwards`.
    events::Dict{AbstractEventKey, AbstractEventModel}
    # Keyed by UUID to match PNM's `get_registered_contingencies(::VirtualMODF) ::
    # Dict{UUID, ContingencySpec}` so the consolidation step in network_model.jl can
    # set-diff directly. UUIDs are also stable across (de)serialization in a way that
    # live component references aren't.
    outages::Dict{Base.UUID, Dict{DataType, Set{String}}}
    device_cache::Vector{D}
    function DeviceModel(
        ::Type{D},
        ::Type{B};
        feedforwards = Vector{AbstractAffectFeedforward}(),
        use_slacks = false,
        duals = Vector{DataType}(),
        time_series_names = get_default_time_series_names(D, B),
        attributes = Dict{String, Any}(),
        outages::AbstractVector{<:IS.InfrastructureSystemsComponent} =
        IS.InfrastructureSystemsComponent[],
    ) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
        attributes_ = get_default_attributes(D, B)
        for (k, v) in attributes
            attributes_[k] = v
        end

        _check_device_formulation(D)
        _check_device_formulation(B)
        outages_field = _add_device_model_outages(D, B, outages)
        new{D, B}(
            convert(Vector{AbstractAffectFeedforward}, feedforwards),
            use_slacks,
            duals,
            Vector{ServiceModel}(),
            time_series_names,
            attributes_,
            nothing,
            Dict{AbstractEventKey, AbstractEventModel}(),
            outages_field,
            Vector{D}(),
        )
    end
end

function _add_device_model_outages(
    ::Type{D},
    ::Type{B},
    outages::AbstractVector{<:IS.InfrastructureSystemsComponent},
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
    field = Dict{Base.UUID, Dict{DataType, Set{String}}}()
    isempty(outages) && return field
    if !supports_outages(B)
        @warn "DeviceModel{$D, $B}: 'outages' kwarg ignored — formulation does \
               not support N-1 contingencies."
        return field
    end
    for outage in outages
        field[IS.get_uuid(outage)] = Dict{DataType, Set{String}}()
    end
    return field
end

"""
    supports_outages(::Type{<:AbstractDeviceFormulation}) -> Bool

Trait declaring whether a device formulation consumes `DeviceModel.outages`.
Defaults to `false`. POM specializes this to `true` for security-constrained branch
formulations (`AbstractSecurityConstrainedStaticBranch`).
"""
supports_outages(::Type{<:AbstractDeviceFormulation}) = false

get_component_type(
    ::DeviceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation} = D
get_formulation(
    ::DeviceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation} = B
get_feedforwards(m::DeviceModel) = m.feedforwards
get_services(m::DeviceModel) = m.services
get_services(::Nothing) = nothing
get_use_slacks(m::DeviceModel) = m.use_slacks
get_duals(m::DeviceModel) = m.duals
get_time_series_names(m::DeviceModel) = m.time_series_names
get_attributes(m::DeviceModel) = m.attributes
get_attribute(::Nothing, ::String) = nothing
get_attribute(m::DeviceModel, key::String) = get(m.attributes, key, nothing)
get_subsystem(m::DeviceModel) = m.subsystem
get_events(m::DeviceModel) = m.events
get_outages(m::DeviceModel) = m.outages
get_device_cache(m::DeviceModel) = m.device_cache

set_subsystem!(m::DeviceModel, id::String) = m.subsystem = id

"""
    set_event_model!(model::DeviceModel, key::AbstractEventKey, event_model::AbstractEventModel)

Attach an event (contingency) model to `model` under `key`. Errors if `key` is already
present. The concrete `EventKey`/`EventModel` types and the convenience method that derives
the key from the event model live in `PowerOperationsModels`.
"""
function set_event_model!(
    model::DeviceModel{D, B},
    key::AbstractEventKey,
    event_model::AbstractEventModel,
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
    if haskey(model.events, key)
        error("EventModel $key already exists in model for device $D")
    end
    model.events[key] = event_model
    return
end

function set_model!(
    dict::Dict,
    model::DeviceModel{D, B},
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
    key = nameof(D)
    if haskey(dict, key)
        @warn "Overwriting $(nameof(D)) existing model"
    end
    dict[key] = model
    return
end

has_service_model(model::DeviceModel) = !isempty(get_services(model))
