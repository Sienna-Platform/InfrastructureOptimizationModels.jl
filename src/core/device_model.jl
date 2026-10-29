"""
Formulation type to augment the power balance constraint expression with a time series parameter
"""
struct FixedOutput <: AbstractDeviceFormulation end

"""
Abstracts for event contingencies
"""
abstract type AbstractEventModel end
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
        selected_outages = nothing,
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
  - `selected_outages::Union{Nothing, AbstractVector{<:IS.SupplementalAttribute}} = nothing` :
    N-1 contingencies to model when the formulation is security-constrained. `nothing`
    means no selection, so every outage in the system is modeled; a vector models exactly
    those outages, so an empty vector models none. The constructor stores
    `IS.get_id(outage)` of each entry, readable with [`get_selected_outage_ids`](@ref), so
    each outage must already be attached to the system. Power-specific validation (e.g.
    checking that entries are `PSY.Outage` subtypes) and the resolution of the selection
    into the model's `outages` field live in `PowerOperationsModels`. If `B` is not security-constrained, a non-empty value is
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
    events::Dict{AbstractEventKey, AbstractEventModel}
    # Ids of the outages to model; `nothing` models every outage in the system.
    selected_outage_ids::Union{Nothing, Set{Int}}
    # Maps outage ids to monitored components grouped by device type. PNM indexes DF matrices with these ids.
    outages::Dict{Int, Dict{DataType, Set{String}}}
    device_cache::Vector{D}
    function DeviceModel(
        ::Type{D},
        ::Type{B};
        feedforwards = Vector{AbstractAffectFeedforward}(),
        use_slacks = false,
        duals = Vector{DataType}(),
        time_series_names = get_default_time_series_names(D, B),
        attributes = Dict{String, Any}(),
        selected_outages::Union{Nothing, AbstractVector{<:IS.SupplementalAttribute}} = nothing,
    ) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
        attributes_ = get_default_attributes(D, B)
        for (k, v) in attributes
            attributes_[k] = v
        end

        _check_device_formulation(D)
        _check_device_formulation(B)
        new{D, B}(
            convert(Vector{AbstractAffectFeedforward}, feedforwards),
            use_slacks,
            duals,
            Vector{ServiceModel}(),
            time_series_names,
            attributes_,
            nothing,
            Dict{AbstractEventKey, AbstractEventModel}(),
            _selected_outage_ids(D, B, selected_outages),
            Dict{Int, Dict{DataType, Set{String}}}(),
            Vector{D}(),
        )
    end
end

_selected_outage_ids(::Type, ::Type, ::Nothing) = nothing

function _selected_outage_ids(
    ::Type{D},
    ::Type{B},
    selected_outages::AbstractVector{<:IS.SupplementalAttribute},
) where {D <: IS.InfrastructureSystemsComponent, B <: AbstractDeviceFormulation}
    if !supports_outages(B)
        isempty(selected_outages) ||
            @warn "DeviceModel{$D, $B}: 'selected_outages' kwarg ignored — \
                   formulation does not support N-1 contingencies."
        return nothing
    end
    ids = Set{Int}()
    for outage in selected_outages
        id = IS.get_id(outage)
        if id == IS.UNASSIGNED_ID
            throw(
                ArgumentError(
                    "DeviceModel{$D, $B}: outage $(summary(outage)) is not attached \
                     to a system. Add it to the system before passing it to `selected_outages`.",
                ),
            )
        end
        push!(ids, id)
    end
    return ids
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
"""
    get_selected_outage_ids(m::DeviceModel) -> Union{Nothing, Set{Int}}

Ids of the outages passed to the `selected_outages` constructor kwarg, or `nothing` when
no outages were selected and every outage in the system is modeled.
"""
get_selected_outage_ids(m::DeviceModel) = m.selected_outage_ids
get_outages(m::DeviceModel) = m.outages
get_device_cache(m::DeviceModel) = m.device_cache

set_subsystem!(m::DeviceModel, id::String) = m.subsystem = id

"""
    set_event_model!(model::DeviceModel, key::AbstractEventKey, event_model::AbstractEventModel)

Attach an event (contingency) model to `model` under `key`. Errors if `key` is already
present.
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
