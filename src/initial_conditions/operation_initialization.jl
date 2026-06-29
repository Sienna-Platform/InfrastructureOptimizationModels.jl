# Called `initialize!` in PSI (lived in operation_model_interface.jl).
function solve_and_write_initial_conditions!(model::AbstractOptimizationModel)
    container = get_optimization_container(model)
    if get_initial_conditions_model_container(get_internal(model)) === nothing
        return
    end
    @info "Solving Initialization Model for $(get_name(model))"
    status = execute_optimizer!(
        get_initial_conditions_model_container(get_internal(model)),
        get_system(model),
    )
    if status == RunStatus.FAILED
        error("Model failed to initialize")
    end

    write_initial_conditions_data!(
        container,
        get_initial_conditions_model_container(get_internal(model)),
    )
    init_file = get_initial_conditions_file(model)
    Serialization.serialize(init_file, get_initial_conditions_data(container))
    @info "Serialized initial conditions to $init_file"
    return
end
