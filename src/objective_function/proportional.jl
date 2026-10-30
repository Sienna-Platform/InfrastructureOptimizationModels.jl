# add_proportional_cost! is used for Thermals (a bunch) and ControllableLoads (once) in POM
# add_proportional_cost_maybe_time_variant! is used to define a add_proportional_cost! in
# POM, for Thermals and ControllableLoads with certain formulations.

# this is only used for ControllableLoads with non-PowerLoadInterruptible formulations.
# The rest go through a thin wrapper around the maybe-variant version.
"""
Default implementation for proportional cost, where the cost term is not time variant.
See also: `add_proportional_cost_maybe_time_variant!` for a common basis for devices that
might have time-variant proportional costs.

A component whose cost is skipped (see [`skip_proportional_cost`](@ref)) is committed
whenever it is active: its cost is booked in `FixedCostExpression`, outside the objective,
scaled by its status (see [`get_component_statuses`](@ref)), so nothing is booked at an
inactive step.
"""
function add_proportional_cost!(
    container::OptimizationContainer,
    ::Type{U},
    devices::Union{Vector{T}, IS.FlattenIteratorWrapper{T}},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(U, V)
    # Scale by the timestep to match variable/VOM cost dt-scaling (else sub-hourly
    # resolutions over-weight proportional costs).
    dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR
    statuses = get_component_statuses(container, T)
    for d in devices
        op_cost_data = get_operation_cost(d)
        cost_term = proportional_cost(op_cost_data, U, d, V)
        iszero(cost_term) && continue
        name = get_name(d)
        rate = cost_term * multiplier * dt
        skip = skip_proportional_cost(d, U)
        for t in get_time_steps(container)
            if skip
                # Always committed: booked in FixedCostExpression at the active steps only,
                # outside the objective.
                active_rate = rate * get_component_status(statuses, name, t)
                iszero(active_rate) && continue
                add_cost_to_expression!(
                    container, FixedCostExpression, active_rate, T, name, t)
            else
                variable = get_variable(container, U, T)[name, t]
                add_cost_term_invariant!(
                    container, variable, rate,
                    FixedCostExpression, T, name, t,
                )
            end
        end
    end
    return
end

"""
Common basis for maybe time variant proportional costs for devices that might be always
committed (see [`skip_proportional_cost`](@ref)); their cost is booked as in
[`add_proportional_cost!`](@ref), scaled by the component status.
Currently used for `(ThermalGen, AbstractThermal)` and `(ControllableLoad, PowerLoadInterruption)`
device, formulation pairs.
"""
function add_proportional_cost_maybe_time_variant!(
    container::OptimizationContainer,
    ::Type{U},
    devices::Union{Vector{T}, IS.FlattenIteratorWrapper{T}},
    ::Type{V},
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(U, V)
    # Same dt scaling as the time-invariant proportional path above.
    dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR
    statuses = get_component_statuses(container, T)
    for d in devices
        op_cost_data = get_operation_cost(d)
        name = get_name(d)
        # is_time_variant_proportional depends only on op_cost_data; hoist out of the time loop.
        add_as_time_variant = is_time_variant_proportional(op_cost_data)
        skip = skip_proportional_cost(d, U)
        for t in get_time_steps(container)
            cost_term = proportional_cost(container, op_cost_data, U, d, V, t)
            iszero(cost_term) && continue
            rate = cost_term * multiplier * dt

            if skip
                # Always committed: booked in FixedCostExpression at the active steps only,
                # outside the objective.
                active_rate = rate * get_component_status(statuses, name, t)
                iszero(active_rate) && continue
                add_cost_to_expression!(
                    container,
                    FixedCostExpression,
                    active_rate,
                    T,
                    name,
                    t,
                )
            else
                variable = get_variable(container, U, T)[name, t]
                if add_as_time_variant
                    add_cost_term_variant!(
                        container, variable, rate, FixedCostExpression, T, name, t)
                else
                    add_cost_term_invariant!(
                        container, variable, rate, FixedCostExpression, T, name, t)
                end
            end
        end
    end
    return
end
