# Device model
function add_constraint_dual!(
    container::OptimizationContainer,
    sys::IS.InfrastructureSystemsContainer,
    model::DeviceModel{T, D},
) where {T <: IS.InfrastructureSystemsComponent, D <: AbstractDeviceFormulation}
    if !isempty(get_duals(model))
        devices = get_available_components(model, sys)
        for constraint_type in get_duals(model)
            assign_dual_variable!(container, constraint_type, devices, D)
        end
    end
    return
end

# Network model
function add_constraint_dual!(
    container::OptimizationContainer,
    sys::IS.InfrastructureSystemsContainer,
    model::NetworkModel{T},
) where {T <: AbstractPowerModel}
    if !isempty(get_duals(model))
        # component is ACBus, but we don't have PSY as a dependency.
        devices = get_available_components(model, component_for_network_dual(nothing), sys)
        for constraint_type in get_duals(model)
            assign_dual_variable!(container, constraint_type, devices, model)
        end
    end
    return
end

# Service model
function add_constraint_dual!(
    container::OptimizationContainer,
    sys::IS.InfrastructureSystemsContainer,
    model::ServiceModel{T, D},
) where {T <: IS.InfrastructureSystemsComponent, D <: AbstractServiceFormulation}
    if !isempty(get_duals(model))
        service = get_available_components(model, sys)
        for constraint_type in get_duals(model)
            assign_dual_variable!(container, constraint_type, service, D)
        end
    end
    return
end

# service formulation
function assign_dual_variable!(
    container::OptimizationContainer,
    constraint_type::Type{<:ConstraintType},
    service::D,
    ::Type{<:AbstractServiceFormulation},
) where {D <: IS.InfrastructureSystemsComponent}
    time_steps = get_time_steps(container)
    service_name = IS.get_name(service)
    add_dual_container!(
        container,
        constraint_type,
        D,
        [service_name],
        time_steps;
        meta = service_name,
    )
    return
end

# device formulation
function assign_dual_variable!(
    container::OptimizationContainer,
    constraint_type::Type{<:ConstraintType},
    devices::U,
    ::Type{<:AbstractDeviceFormulation},
) where {
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    time_steps = get_time_steps(container)
    metas = _existing_constraint_metas(container, constraint_type, D)
    if isempty(metas)
        device_names = IS.get_name.(devices)
        add_dual_container!(container, constraint_type, D, device_names, time_steps)
        return
    end
    for meta in metas
        key = ConstraintKey(constraint_type, D, meta)
        existing = get_constraint(container, key)
        _assign_dual_from_existing!(container, key, existing, D, time_steps)
    end
    return
end

# Sparse constraints (e.g. post-contingency flow-rate constraints keyed by
# (outage_id, name, t)) have no `axes`. Mirror the constraint's exact sparse keys
# into a Float64 dual container so the dual matches the constraint storage one-to-one.
function _assign_dual_from_existing!(
    container::OptimizationContainer,
    key::ConstraintKey,
    existing::SparseAxisArray,
    ::Type{D},
    time_steps,
) where {D}
    dual_container =
        SparseAxisArray(Dict(k => zero(Float64) for k in keys(existing.data)))
    _assign_container!(container.duals, key, dual_container)
    return
end

# Reuse the existing constraint container's row axis so the dual axis matches the
# constraint exactly. Network reductions (radial / degree-two) drop branches that
# pass the device-model filter, so the constraint axis is a strict subset of
# IS.get_name.(devices). Sizing the dual from the device list would leave the dual
# broadcast in process_duals incompatible with the constraint matrix.
function _assign_dual_from_existing!(
    container::OptimizationContainer,
    key::ConstraintKey,
    existing::DenseAxisArray,
    ::Type{D},
    time_steps,
) where {D}
    row_axis = axes(existing)[1]
    add_dual_container!(
        container,
        get_entry_type(key),
        D,
        row_axis,
        time_steps;
        meta = key.meta,
    )
    return
end

function _existing_constraint_metas(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{D},
) where {T <: ConstraintType, D}
    metas = String[]
    for key in get_constraint_keys(container)
        if get_entry_type(key) === T &&
           get_component_type(key) === D
            push!(metas, key.meta)
        end
    end
    return metas
end

# network model with buses
function assign_dual_variable!(
    container::OptimizationContainer,
    constraint_type::Type{<:ConstraintType},
    devices::U,
    ::NetworkModel{<:AbstractPowerModel},
) where {
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    time_steps = get_time_steps(container)
    add_dual_container!(
        container,
        constraint_type,
        D,
        IS.get_name.(devices),
        time_steps,
    )
    return
end
