"""
Per-container component statuses for the mock device types.

`set_test_statuses!` registers a status table for one container and component type; every other
container, and every unregistered type, keeps the default `AllComponentsActive()`. Keying on
the container keeps the override from leaking into tests that build their own containers.
"""
const _TEST_COMPONENT_STATUSES =
    WeakKeyDict{IOM.OptimizationContainer, Dict{DataType, IOM.AbstractComponentStatuses}}()

function set_test_statuses!(
    container::IOM.OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    values::Matrix{Float64},
) where {C <: IS.InfrastructureSystemsComponent}
    statuses = DenseAxisArray(values, names, IOM.get_time_steps(container))
    by_type = get!(
        Dict{DataType, IOM.AbstractComponentStatuses},
        _TEST_COMPONENT_STATUSES,
        container,
    )
    by_type[C] = IOM.ComponentStatusArray(statuses)
    return
end

function IOM.get_component_statuses(
    container::IOM.OptimizationContainer,
    ::Type{C},
) where {C <: AbstractMockDevice}
    by_type = get(_TEST_COMPONENT_STATUSES, container, nothing)
    isnothing(by_type) && return IOM.AllComponentsActive()
    return get(by_type, C, IOM.AllComponentsActive())
end

# The bounds that the range builders scale by the status are those on variables fixed while
# their component is inactive.
IOM.fixed_when_inactive(::Type{TestVariableType}) = true

"""
The LP text of the container's JuMP model, for comparing two builds term by term.
"""
function lp_text(container::IOM.OptimizationContainer)
    return mktempdir() do dir
        path = joinpath(dir, "model.lp")
        JuMP.write_to_file(IOM.get_jump_model(container), path)
        return read(path, String)
    end
end
