@doc raw"""
Default implementation of adding auxiliary variable to the model.
"""
function add_variables!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    formulation,
) where {
    T <: AuxVariableType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    time_steps = get_time_steps(container)
    add_aux_variable_container!(
        container,
        T,
        D,
        IS.get_name.(devices),
        time_steps,
    )
    return
end
