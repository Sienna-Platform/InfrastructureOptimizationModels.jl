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

function _check_pm_formulation(::Type{T}) where {T <: AbstractPowerModel}
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
- `::Type{T}` where `T <: AbstractPowerModel`: the power-system formulation type.

# Accepted keyword arguments
- `use_slacks::Bool` = false
    Adds slack buses to the network modeling.
- `PTDF_matrix::Union{AbstractInfrastructureNetworkMatrix, Nothing}` = nothing
    PTDF/VirtualPTDF matrix (e.g. produced by PowerNetworkMatrices; optional).
- `MODF_matrix::Union{AbstractInfrastructureNetworkMatrix, Nothing}` = nothing
    VirtualMODF matrix for security-constrained models (N-k contingencies).
    If `nothing` and the template includes a security-constrained branch
    formulation, the matrix is constructed from the system during
    `instantiate_network_model!` (same pattern as PTDF).
- `reduce_radial_branches::Bool` = false
    Enable radial branch reduction when building network matrices.
- `reduce_degree_two_branches::Bool` = false
    Enable degree-two branch reduction when building network matrices.
- `subnetworks::Dict{Int, Set{Int}}` = Dict()
    Optional mapping of reference bus → set of mapped buses. If not provided,
    subnetworks are inferred from PTDF/VirtualPTDF or discovered from the system.
- `duals::Vector{DataType}` = Vector{DataType}()
    Constraint types for which duals should be recorded.
- `evaluations::EvaluationContainer`
    External evaluators (e.g. power-flow) keyed by concrete evaluator type.
    Default is an empty container — no evaluator runs.

# Notes
- `modeled_branch_types` and `reduced_branch_tracker` are internal fields managed by the model.
- `subsystem` can be set after construction via `set_subsystem!(model, id)`.
- PTDF and MODF inputs are validated against the requested reduction flags and
  may raise a ConflictingInputsError if they are inconsistent with
  `reduce_radial_branches` or `reduce_degree_two_branches`.

# Examples (concrete types like PTDFPowerModel, CopperPlatePowerModel are defined in PowerSimulations)
# ptdf = PowerNetworkMatrices.VirtualPTDF(system)
# ec = EvaluationContainer()
# add_evaluator!(ec, PFS.PowerFlowEvaluationModel, PFS.PowerFlowEvaluationModel())
# nw = NetworkModel(PTDFPowerModel; PTDF_matrix = ptdf, reduce_radial_branches = true,
#                   evaluations = ec)
#
# nw2 = NetworkModel(CopperPlatePowerModel; subnetworks = Dict(1 => Set([1,2,3])))
"""
mutable struct NetworkModel{T <: AbstractPowerModel}
    use_slacks::Bool
    PTDF_matrix::Union{Nothing, AbstractInfrastructureNetworkMatrix}
    MODF_matrix::Union{Nothing, AbstractInfrastructureNetworkMatrix}
    subnetworks::Dict{Int, Set{Int}}
    bus_area_map::Dict{IS.InfrastructureSystemsComponent, Int}
    duals::Vector{DataType}
    network_reduction::Union{Nothing, AbstractInfrastructureNetworkReductionData}
    reduce_radial_branches::Bool
    reduce_degree_two_branches::Bool
    evaluations::EvaluationContainer
    subsystem::Union{Nothing, String}
    hvdc_network_model::Union{Nothing, AbstractHVDCNetworkModel}
    modeled_branch_types::Vector{DataType}
    reduced_branch_tracker::Union{Nothing, AbstractBranchReductionTracker}

    function NetworkModel(
        ::Type{T};
        use_slacks = false,
        PTDF_matrix = nothing,
        MODF_matrix = nothing,
        reduce_radial_branches = false,
        reduce_degree_two_branches = false,
        subnetworks = Dict{Int, Set{Int}}(),
        duals = Vector{DataType}(),
        evaluations = EvaluationContainer(),
        hvdc_network_model = nothing,
    ) where {T <: AbstractPowerModel}
        _check_pm_formulation(T)
        new{T}(
            use_slacks,
            PTDF_matrix,
            MODF_matrix,
            subnetworks,
            Dict{IS.InfrastructureSystemsComponent, Int}(),
            duals,
            # Populated by the network-matrix-aware instantiation code (POM); IOM
            # holds it behind the IS abstraction so it carries no PNM dependency.
            nothing,
            reduce_radial_branches,
            reduce_degree_two_branches,
            evaluations,
            nothing,
            hvdc_network_model,
            Vector{DataType}(),
            nothing,
        )
    end
end

get_use_slacks(m::NetworkModel) = m.use_slacks
get_PTDF_matrix(m::NetworkModel) = m.PTDF_matrix
get_MODF_matrix(m::NetworkModel) = m.MODF_matrix
get_reduce_radial_branches(m::NetworkModel) = m.reduce_radial_branches
get_network_reduction(m::NetworkModel) = m.network_reduction
get_duals(m::NetworkModel) = m.duals
get_network_formulation(::NetworkModel{T}) where {T} = T
get_reduced_branch_tracker(m::NetworkModel) = m.reduced_branch_tracker
get_reference_buses(m::NetworkModel{T}) where {T <: AbstractPowerModel} =
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
requires_all_branch_models(::Type{<:AbstractPowerModel}) = true
supports_branch_filtering(::Type{<:AbstractPowerModel}) = false
ignores_branch_filtering(::Type{<:AbstractPowerModel}) = false
branches_modeled(::Type{<:AbstractPowerModel}) = true

function _check_branch_network_compatibility(
    ::NetworkModel{T},
    unmodeled_branch_types::Vector{DataType},
) where {T <: AbstractPowerModel}
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
) where {T <: AbstractPowerModel}
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
) where {T <: AbstractPowerModel}
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
