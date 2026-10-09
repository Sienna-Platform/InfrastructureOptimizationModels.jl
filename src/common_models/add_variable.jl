"""
Set `value` as the start value of `variable` when warm start is enabled in the container
settings. A `nothing` value leaves the variable without a start value.

Every start value in a model built on this package should go through this function, so a
model built with `warm_start = false` reaches the solver with no start values at all. Some
solvers treat a partial start as a sub-problem to complete before presolve, so a single
stray start value can cost as much as a full warm start.
"""
function set_start_value!(
    container::OptimizationContainer,
    variable::JuMP.VariableRef,
    value::Union{Nothing, Real},
)
    value === nothing && return
    get_warm_start(get_settings(container)) || return
    JuMP.set_start_value(variable, value)
    return
end

@doc raw"""
Adds a variable to the optimization model.

Bounds determined from get_variable_lower_bound, get_variable_upper_bound. Binary determined from
get_variable_binary. Start value determined from get_variable_warm_start_value.

# Arguments
* container::OptimizationContainer : the optimization container model built in InfrastructureOptimizationModels
* ::Type{T} : variable type
* devices::U : vector or iterator of devices
* ::Type{F} : device formulation

# Keyword arguments
* relax_binaries::Bool : When true, binary variables are instead continuous in [0, 1].
"""
function add_variables!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    ::Type{F};
    relax_binaries::Bool = false,
) where {
    T <: VariableType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    F,
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    time_steps = get_time_steps(container)
    relaxed = relax_binaries && get_variable_binary(T, D, F)
    binary = !relax_binaries && get_variable_binary(T, D, F)
    included = [d for d in devices if !skip_variable(T, d, F)]

    variable = add_variable_container!(
        container,
        T,
        D,
        String[get_name(d) for d in included],
        time_steps,
    )

    for t in time_steps, d in included
        name = get_name(d)
        var =
            variable[name, t] = JuMP.@variable(
                get_jump_model(container),
                base_name = "$(T)_$(D)_{$(name), $(t)}",
                binary = binary
            )
        if relaxed
            JuMP.set_lower_bound(var, 0.0)
            JuMP.set_upper_bound(var, 1.0)
        else
            ub = get_variable_upper_bound(T, d, F)
            ub !== nothing && JuMP.set_upper_bound(var, ub)
            lb = get_variable_lower_bound(T, d, F)
            lb !== nothing && JuMP.set_lower_bound(var, lb)
        end
        set_start_value!(container, var, get_variable_warm_start_value(T, d, F))
    end

    return
end

"""
Add variables to the OptimizationContainer for every service of a type and their contributing
devices.

Each `(device type, service type)` pair gets its own sparse container keyed on
`ComponentPairKey{D, U}` and indexed by `(service_name, device_name, time)`, holding every
service of that type. Keying on the pair keeps devices of different types that share a name,
and services of different types that share a name, apart.
"""
function add_service_variables!(
    container::OptimizationContainer,
    ::Type{T},
    services::Vector{U},
    model::ServiceModel,
    ::Type{F},
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    F <: AbstractServiceFormulation,
}
    by_device_type = Dict{DataType, Vector{Tuple{U, Vector}}}()
    for service in services
        for (device_type, devices) in
            get_contributing_devices_map(model, IS.get_name(service))
            isempty(devices) && continue
            push!(
                get!(Vector{Tuple{U, Vector}}, by_device_type, device_type),
                (service, devices),
            )
        end
    end
    for (device_type, entries) in by_device_type
        _add_service_variables!(container, T, U, device_type, entries, F)
    end
    return
end

function _add_service_variables!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{D},
    entries::Vector{Tuple{U, Vector}},
    ::Type{F},
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    D <: IS.InfrastructureSystemsComponent,
    F <: AbstractServiceFormulation,
}
    variable = add_variable_container!(
        container,
        T,
        ComponentPairKey{D, U},
        String[],
        String[],
        Int[];
        sparse = true,
    )
    for (service, devices) in entries
        _add_service_device_variables!(container, variable, T, service, devices, F)
    end
    return
end

function _add_service_device_variables!(
    container::OptimizationContainer,
    variable::SparseAxisArray,
    ::Type{T},
    service::U,
    devices::Vector{D},
    ::Type{F},
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    D <: IS.InfrastructureSystemsComponent,
    F <: AbstractServiceFormulation,
}
    binary = get_variable_binary(T, U, F)
    service_name = IS.get_name(service)
    jump_model = get_jump_model(container)
    for d in devices, t in get_time_steps(container)
        device_name = IS.get_name(d)
        var = JuMP.@variable(
            jump_model,
            base_name = "$(T)_$(D)_$(U)_{$(service_name), $(device_name), $(t)}",
            binary = binary,
        )
        variable[service_name, device_name, t] = var
        ub = get_variable_upper_bound(T, service, d, F)
        ub !== nothing && JuMP.set_upper_bound(var, ub)
        lb = get_variable_lower_bound(T, service, d, F)
        lb !== nothing && !binary && JuMP.set_lower_bound(var, lb)
        set_start_value!(container, var, get_variable_warm_start_value(T, d, F))
    end
    return
end
