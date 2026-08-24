const DeviceModelForBranches =
    DeviceModel{<:IS.InfrastructureSystemsComponent, <:AbstractDeviceFormulation}
const BranchModelContainer = Dict{Symbol, DeviceModelForBranches}

"""
Abstract anchor for the branch-reduction bookkeeping carried by a `NetworkModel`.
The concrete tracker (and all reduction machinery) lives in the matrix-aware
downstream package (POM), which constructs it during network model instantiation
via [`set_reduced_branch_tracker!`](@ref).
"""
abstract type AbstractBranchReductionTracker end

"""
Abstract anchor for the declaration of which network a `NetworkModel` is built on and
how it is reduced. Concrete sources (a reduction specification, a prebuilt matrix, a
prebuilt factorization core) live in the matrix-aware downstream package (POM), so
IOM carries no dependency on any matrix implementation.
"""
abstract type AbstractNetworkSource end

"""
Abstract anchor for the network artifacts a build derives from an
[`AbstractNetworkSource`](@ref): the reduction data plus whichever matrices the
network formulation needs. Concrete containers live in POM.
"""
abstract type AbstractNetworkData end

"""
Source used when a `NetworkModel` is constructed without an explicit one: build the
network from the system with no reductions applied.
"""
struct DefaultNetworkSource <: AbstractNetworkSource end

"Abstract supertype for network model formulations; neutral anchor for NetworkModel{T}."
abstract type AbstractNetworkModel <: IS.Optimization.AbstractInfrastructureModel end

function _check_network_formulation(::Type{T}) where {T <: AbstractNetworkModel}
    if !isconcretetype(T)
        throw(
            ArgumentError(
                "The network model must contain only concrete types, $(T) is an Abstract Type",
            ),
        )
    end
end

"""
Establishes the NetworkModel for a given AC network formulation type.

# Arguments
- `::Type{T}` where `T <: AbstractNetworkModel`: the network formulation type.

# Accepted keyword arguments
- `use_slacks::Bool` = false
    Adds slack buses to the network modeling.
- `network_source::AbstractNetworkSource` = `DefaultNetworkSource()`
    Declares which network the model is built on and how it is reduced. The default
    builds it from the system with no reductions. Concrete sources (a reduction
    specification, a prebuilt matrix, a prebuilt factorization core) live in the
    matrix-aware downstream package.
- `reduction_exceptions::Vector{Int}` = `Int[]`
    Bus numbers the reduction must not eliminate, on top of those the template
    itself pins.
- `duals::Vector{DataType}` = Vector{DataType}()
    Constraint types for which duals should be recorded.
- `evaluations::EvaluationContainer`
    External evaluators (e.g. power-flow) keyed by concrete evaluator type.
    Default is an empty container — no evaluator runs.

# Notes
- `network_data` holds every matrix and the reduction data derived from
  `network_source` during `instantiate_network_model!`; it is `nothing` before then.
- `subnetworks`, `modeled_branch_types` and `reduced_branch_tracker` are internal
  fields managed by the model.
- `subsystem` can be set after construction via `set_subsystem!(model, id)`.

# Examples (concrete types like PTDFPowerModel, CopperPlatePowerModel are defined in PowerSimulations)
# ec = EvaluationContainer()
# add_evaluator!(ec, PFS.PowerFlowEvaluationModel, PFS.PowerFlowEvaluationModel())
# nw = NetworkModel(PTDFPowerModel;
#                   network_source = NetworkReductionSpec(RadialReduction()),
#                   evaluations = ec)
#
# nw2 = NetworkModel(CopperPlatePowerModel)
"""
mutable struct NetworkModel{T <: AbstractNetworkModel}
    use_slacks::Bool
    network_source::AbstractNetworkSource
    reduction_exceptions::Vector{Int}
    subnetworks::Dict{Int, Set{Int}}
    bus_area_map::Dict{IS.InfrastructureSystemsComponent, Int}
    duals::Vector{DataType}
    network_data::Union{Nothing, AbstractNetworkData}
    evaluations::EvaluationContainer
    subsystem::Union{Nothing, String}
    hvdc_network_model::Union{Nothing, AbstractHVDCNetworkModel}
    modeled_branch_types::Vector{DataType}
    reduced_branch_tracker::Union{Nothing, AbstractBranchReductionTracker}

    function NetworkModel(
        ::Type{T};
        use_slacks = false,
        network_source = DefaultNetworkSource(),
        reduction_exceptions = Int[],
        duals = Vector{DataType}(),
        evaluations = EvaluationContainer(),
        hvdc_network_model = nothing,
    ) where {T <: AbstractNetworkModel}
        _check_network_formulation(T)
        new{T}(
            use_slacks,
            network_source,
            reduction_exceptions,
            Dict{Int, Set{Int}}(),
            Dict{IS.InfrastructureSystemsComponent, Int}(),
            duals,
            # Populated by the network-matrix-aware instantiation code (POM); IOM
            # holds it behind an abstract type so it carries no PNM dependency.
            nothing,
            evaluations,
            nothing,
            hvdc_network_model,
            Vector{DataType}(),
            nothing,
        )
    end
end

get_use_slacks(m::NetworkModel) = m.use_slacks
get_network_source(m::NetworkModel) = m.network_source
get_reduction_exceptions(m::NetworkModel) = m.reduction_exceptions
get_network_data(m::NetworkModel) = m.network_data
get_duals(m::NetworkModel) = m.duals

"""
The network matrix derived during instantiation. Implemented in the matrix-aware
downstream package, which owns the concrete `AbstractNetworkData`.
"""
function get_network_matrix end

"""The contingency matrix derived during instantiation. Implemented downstream."""
function get_contingency_matrix end

"""The network reduction derived during instantiation. Implemented downstream."""
function get_network_reduction end

get_network_formulation(::NetworkModel{T}) where {T} = T
get_reduced_branch_tracker(m::NetworkModel) = m.reduced_branch_tracker
get_reference_buses(m::NetworkModel{T}) where {T <: AbstractNetworkModel} =
    collect(keys(m.subnetworks))
get_subnetworks(m::NetworkModel) = m.subnetworks
get_bus_area_map(m::NetworkModel) = m.bus_area_map
get_evaluations(m::NetworkModel) = m.evaluations
has_subnetworks(m::NetworkModel) = !isempty(m.bus_area_map)
get_subsystem(m::NetworkModel) = m.subsystem
get_hvdc_network_model(m::NetworkModel) = m.hvdc_network_model

set_subsystem!(m::NetworkModel, id::String) = m.subsystem = id
set_hvdc_network_model!(m::NetworkModel, val::Union{Nothing, AbstractHVDCNetworkModel}) =
    m.hvdc_network_model = val
function set_reduced_branch_tracker!(m::NetworkModel, val::AbstractBranchReductionTracker)
    m.reduced_branch_tracker = val
    return
end

function set_network_data!(m::NetworkModel, val::Union{Nothing, AbstractNetworkData})
    m.network_data = val
    return
end

function add_dual!(model::NetworkModel, dual)
    dual in model.duals && error("dual = $dual is already stored")
    push!(model.duals, dual)
    @debug "Added dual" dual _group = LOG_GROUP_NETWORK_CONSTRUCTION
    return
end

"""
True if any branch DeviceModel in `branch_models` uses a formulation that
consumes `DeviceModel.outages` (per `supports_outages`). POM's
`AbstractSecurityConstrainedStaticBranch` specialization makes that trait
return `true`; non-SC formulations default to `false`.

`BranchModelContainer` (`Dict{Symbol, DeviceModelForBranches}`) is defined at
the top of this file and exported from IOM.
"""
function _template_has_outage_aware_branch(branch_models::BranchModelContainer)
    for v in values(branch_models)
        if supports_outages(get_formulation(v))
            return true
        end
    end
    return false
end

# Default implementations for network model compatibility checks
# These can be extended in PowerOperationsModels for specific network formulations
requires_all_branch_models(::Type{<:AbstractNetworkModel}) = true
supports_branch_filtering(::Type{<:AbstractNetworkModel}) = false
ignores_branch_filtering(::Type{<:AbstractNetworkModel}) = false
branches_modeled(::Type{<:AbstractNetworkModel}) = true

function _check_branch_network_compatibility(
    ::NetworkModel{T},
    unmodeled_branch_types::Vector{DataType},
) where {T <: AbstractNetworkModel}
    if requires_all_branch_models(T) && !isempty(unmodeled_branch_types)
        for d in unmodeled_branch_types
            @error "The system has a branch branch type $(d) but the DeviceModel is not included in the Template."
        end
        throw(
            IS.ConflictingInputsError(
                "Network model $(T) requires all AC Transmission devices have a model",
            ),
        )
    end
    return
end

function _validate_branch_models(
    ::Type{T},
    model_has_branch_filters::Bool,
) where {T <: AbstractNetworkModel}
    if supports_branch_filtering(T) || !model_has_branch_filters
        return
    elseif model_has_branch_filters
        if ignores_branch_filtering(T)
            @warn "Branch filtering is ignored for network model $(T)"
        else
            throw(
                IS.ConflictingInputsError(
                    "Branch filtering is not supported for network model $(T). Remove branch \\
                    filter functions from branch models or use a different network model.",
                ),
            )
        end
    else
        throw(
            IS.ConflictingInputsError(
                "Network model $(T) can't be validated against branch models",
            ),
        )
    end
    return
end

function validate_network_model(
    network_model::NetworkModel{T},
    unmodeled_branch_types::Vector{DataType},
    model_has_branch_filters::Bool,
) where {T <: AbstractNetworkModel}
    _check_branch_network_compatibility(network_model, unmodeled_branch_types)
    _validate_branch_models(T, model_has_branch_filters)
    return
end

function _get_filters(branch_models::BranchModelContainer)
    filters = Dict{DataType, Function}()
    for v in values(branch_models)
        filter_func = get_attribute(v, "filter_function")
        if filter_func !== nothing
            filters[get_component_type(v)] = filter_func
        end
    end
    return filters
end

# NOTE: instantiate_network_model! implementations have been moved to
# PowerOperationsModels/src/network_models/instantiate_network_model.jl
# IOM retains only the generic dispatch entry point in operation_model_interface.jl
