"""
Minimal mock for PSY.System.
Implements only the interface required by OptimizationContainer and models.
"""

#=
function stores_time_series_in_memory end
function get_time_series_counts_by_type end
function get_time_series_counts end
function get_forecast_interval end
function get_time_series_resolutions end
function get_forecast_horizon end
function get_uuid end
=#

using InfrastructureSystems
const IS = InfrastructureSystems

mutable struct MockSystemData
    base_power::Float64
    components::Dict{DataType, Vector{Any}}
    time_series::Dict{Any, Any}
    stores_in_memory::Bool
    internal::IS.InfrastructureSystemsInternal
end

mutable struct MockSystem <: IS.InfrastructureSystemsContainer
    data::MockSystemData
end

MockSystem() = MockSystem(MockSystemData())
MockSystem(base_power::Float64) = MockSystem(MockSystemData(base_power))
MockSystem(base_power::Float64, stores_in_memory::Bool) =
    MockSystem(MockSystemData(base_power, stores_in_memory))

# Convenience constructors
MockSystemData() = MockSystemData(
    100.0,
    Dict{DataType, Vector{Any}}(),
    Dict{Any, Any}(),
    false,
    IS.InfrastructureSystemsInternal(),
)
MockSystemData(base_power::Float64) = MockSystemData(
    base_power,
    Dict{DataType, Vector{Any}}(),
    Dict{Any, Any}(),
    false,
    IS.InfrastructureSystemsInternal(),
)
MockSystemData(base_power::Float64, stores_in_memory::Bool) = MockSystemData(
    base_power,
    Dict{DataType, Vector{Any}}(),
    Dict{Any, Any}(),
    stores_in_memory,
    IS.InfrastructureSystemsInternal(),
)

# Required interface methods - extend InfrastructureOptimizationModels functions for duck-typing
IOM.get_base_power(sys::MockSystem) = sys.data.base_power
IOM.stores_time_series_in_memory(sys::MockSystem) = sys.data.stores_in_memory
IOM.stores_time_series_in_memory(data::MockSystemData) = data.stores_in_memory

function IOM.get_available_components(::NetworkModel, ::Type{T}, sys::MockSystem) where {T}
    return get_components(T, sys)
end

function get_components(::Type{T}, sys::MockSystem) where {T}
    return get(sys.data.components, T, T[])
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
    if !haskey(sys.data.components, comp_type)
        sys.data.components[comp_type] = []
    end
    push!(sys.data.components[comp_type], component)
    return
end

function get_time_series(
    ::Type{T},
    sys::MockSystem,
    component,
    args...;
    kwargs...,
) where {T}
    return get(sys.data.time_series, (T, component), nothing)
end

function add_time_series!(sys::MockSystem, component, ts)
    sys.data.time_series[(typeof(ts), component)] = ts
    return
end
