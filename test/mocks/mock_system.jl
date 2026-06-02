"""
Minimal mock for PSY.System.
Implements only the interface required by OptimizationContainer and models.
"""

using InfrastructureSystems
const IS = InfrastructureSystems

mutable struct MockSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
    components::Dict{DataType, Vector{Any}}
    time_series::Dict{Any, Any}
    stores_in_memory::Bool
end

# Convenience constructors
MockSystem() = MockSystem(100.0, Dict{DataType, Vector{Any}}(), Dict{Any, Any}(), false)
MockSystem(base_power::Float64) =
    MockSystem(base_power, Dict{DataType, Vector{Any}}(), Dict{Any, Any}(), false)
MockSystem(base_power::Float64, stores_in_memory::Bool) =
    MockSystem(
        base_power,
        Dict{DataType, Vector{Any}}(),
        Dict{Any, Any}(),
        stores_in_memory,
    )

# Required interface methods - extend InfrastructureOptimizationModels functions for duck-typing
InfrastructureOptimizationModels.get_base_power(sys::MockSystem) = sys.base_power
InfrastructureOptimizationModels.stores_time_series_in_memory(sys::MockSystem) =
    sys.stores_in_memory

function IOM.get_available_components(::NetworkModel, ::Type{T}, sys::MockSystem) where {T}
    return get_components(T, sys)
end

function get_components(::Type{T}, sys::MockSystem) where {T}
    return get(sys.components, T, T[])
end

function get_component(::Type{T}, sys::MockSystem, name::String) where {T}
    for component in get_components(T, sys)
        if get_name(component) == name
            return component
        end
    end
    @error "Component with name '$(name)' not found."
end

function add_component!(sys::MockSystem, component)
    comp_type = typeof(component)
    if !haskey(sys.components, comp_type)
        sys.components[comp_type] = []
    end
    push!(sys.components[comp_type], component)
    return
end

function get_time_series(
    ::Type{T},
    sys::MockSystem,
    component,
    args...;
    kwargs...,
) where {T}
    return get(sys.time_series, (T, component), nothing)
end

function add_time_series!(sys::MockSystem, component, ts)
    sys.time_series[(typeof(ts), component)] = ts
    return
end
