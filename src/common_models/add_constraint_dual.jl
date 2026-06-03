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

_existing_constraint_keys(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{D},
) where {T <: ConstraintType, D} = filter(
    key -> get_entry_type(key) === T && get_component_type(key) === D,
    get_constraint_keys(container),
)

_validate_keys(keys) =
    isempty(keys) && throw(
        IS.InvalidValue(
            "No constraint of type $constraint_type for $D is stored; cannot assign a dual variable.",
        ),
    )

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
    constraint_keys = _existing_constraint_keys(container, constraint_type, D)
    _validate_keys(constraint_keys)
    for key in constraint_keys
        existing = get_constraint(container, key)
        _assign_dual_from_existing!(container, key, existing, D, time_steps)
    end
    return
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
    IS.@assert_op !isempty(devices)
    time_steps = get_time_steps(container)
    constraint_keys = _existing_constraint_keys(container, constraint_type, D)
    _validate_keys(constraint_keys)
    for key in constraint_keys
        existing = get_constraint(container, key)
        _assign_dual_from_existing!(container, key, existing, D, time_steps)
    end
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

function _assign_dual_from_existing!(
    container::OptimizationContainer,
    key::ConstraintKey,
    existing::SparseAxisArray,
    ::Type{D},
    time_steps,
) where {D}
    add_dual_container!(
        container,
        get_entry_type(key),
        D,
        keys(existing.data);
        sparse = true,
    )
    return
end
