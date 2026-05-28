
################## ic updates from store for emulation problems simulation #################

"""
    update_initial_conditions!(model, key, source)

Update initial conditions for a specific key from the model store.
Dispatches to the per-IC-type `update_initial_conditions!(ics, store, resolution)` method.
"""
function update_initial_conditions!(
    model::AbstractOptimizationModel,
    key::InitialConditionKey{T, U},
    source,
) where {T <: InitialConditionType, U <: IS.InfrastructureSystemsComponent}
    if get_execution_count(model) < 1
        return
    end
    container = get_optimization_container(model)
    model_resolution = get_resolution(get_store_params(model))
    ini_conditions_vector = get_initial_condition(container, key)
    update_initial_conditions!(ini_conditions_vector, source, model_resolution)
    return
end

"""
    update_initial_conditions!(ics, store, resolution)

Update initial conditions from the emulation model store.
This is an extension point - concrete implementations for specific initial condition
types should be defined in PowerOperationsModels.

# Arguments
- `ics`: Vector of InitialCondition objects to update
- `store`: EmulationModelStore containing the recorded values
- `resolution`: Time resolution (Dates.Millisecond)
"""
function update_initial_conditions!(
    ics::Vector{<:InitialCondition},
    store::EmulationModelStore{InMemoryDataset},
    resolution::Dates.Millisecond,
)
    # This is a stub - concrete implementations for specific initial condition types
    # (InitialTimeDurationOn, InitialTimeDurationOff, DevicePower, DeviceStatus, etc.)
    # should be defined in PowerOperationsModels.
    error(
        "update_initial_conditions! not implemented for initial condition type " *
        "$(eltype(ics)). Implement this method in PowerOperationsModels.",
    )
end
