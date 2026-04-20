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
        devices = get_available_components(model, IS.InfrastructureSystemsComponent, sys)
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
    add_dual_container!(
        container,
        constraint_type,
        D,
        IS.get_name.(devices),
        time_steps,
    )
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
