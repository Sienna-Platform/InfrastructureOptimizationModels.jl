# Default implementations of getter/setter functions for AbstractOptimizationModel.
is_built(model::AbstractOptimizationModel) =
    get_status(get_internal(model)) == ModelBuildStatus.BUILT
isempty(model::AbstractOptimizationModel) =
    get_status(get_internal(model)) == ModelBuildStatus.EMPTY
warm_start_enabled(model::AbstractOptimizationModel) =
    get_warm_start(get_optimization_container(model).settings)
built_for_recurrent_solves(model::AbstractOptimizationModel) =
    get_optimization_container(model).built_for_recurrent_solves
get_constraints(model::AbstractOptimizationModel) =
    get_constraints(get_internal(model))
get_execution_count(model::AbstractOptimizationModel) =
    get_execution_count(get_internal(model))
get_executions(model::AbstractOptimizationModel) = get_executions(get_internal(model))
get_initial_time(model::AbstractOptimizationModel) = get_initial_time(get_settings(model))
get_internal(model::AbstractOptimizationModel) = model.internal

function get_jump_model(model::AbstractOptimizationModel)
    return get_jump_model(get_container(get_internal(model)))
end

get_name(model::AbstractOptimizationModel) = model.name
get_store(model::AbstractOptimizationModel) = model.store
is_synchronized(model::AbstractOptimizationModel) = is_synchronized(get_optimization_container(model))

function get_rebuild_model(model::AbstractOptimizationModel)
    sim_info = model.simulation_info
    if sim_info === nothing
        error("Model not part of a simulation")
    end
    return get_rebuild_model(get_optimization_container(model).settings)
end

function get_optimization_container(model::AbstractOptimizationModel)
    return get_optimization_container(get_internal(model))
end

function get_resolution(model::AbstractOptimizationModel)
    resolution = get_resolution(get_settings(model))
    return resolution
end

get_problem_base_power(model::AbstractOptimizationModel) = get_base_power(model.sys)
get_settings(model::AbstractOptimizationModel) = get_optimization_container(model).settings

get_optimizer_stats(model::AbstractOptimizationModel) =
# This deepcopy is important because the optimization container is overwritten
# at each solve in a simulation.
    deepcopy(get_optimizer_stats(get_optimization_container(model)))

get_simulation_info(model::AbstractOptimizationModel) = model.simulation_info

function get_simulation_number(model::AbstractOptimizationModel)
    sim_info = get_simulation_info(model)
    isnothing(sim_info) &&
        error("Model is not part of a simulation. Cannot get simulation number.")
    return get_number(sim_info)
end

function set_simulation_number!(model::AbstractOptimizationModel, val)
    sim_info = get_simulation_info(model)
    isnothing(sim_info) &&
        error("Model is not part of a simulation. Cannot set simulation number.")
    return set_number!(sim_info, val)
end

function get_sequence_uuid(model::AbstractOptimizationModel)
    sim_info = get_simulation_info(model)
    isnothing(sim_info) &&
        error("Model is not part of a simulation. Cannot get sequence UUID.")
    return get_sequence_uuid(sim_info)
end

function set_sequence_uuid!(model::AbstractOptimizationModel, val)
    sim_info = get_simulation_info(model)
    isnothing(sim_info) &&
        error("Model is not part of a simulation. Cannot set sequence UUID.")
    return set_sequence_uuid!(sim_info, val)
end
get_status(model::AbstractOptimizationModel) = get_status(get_internal(model))
get_system(model::AbstractOptimizationModel) = model.sys
get_template(model::AbstractOptimizationModel) = model.template
get_log_file(model::AbstractOptimizationModel) = joinpath(get_output_dir(model), PROBLEM_LOG_FILENAME)
get_store_params(model::AbstractOptimizationModel) =
    get_store_params(get_internal(model))
get_output_dir(model::AbstractOptimizationModel) = get_output_dir(get_internal(model))
get_initial_conditions_file(model::AbstractOptimizationModel) =
    joinpath(get_output_dir(model), "initial_conditions.bin")
get_recorder_dir(model::AbstractOptimizationModel) =
    joinpath(get_output_dir(model), "recorder")
get_variables(model::AbstractOptimizationModel) = get_variables(get_optimization_container(model))
get_parameters(model::AbstractOptimizationModel) = get_parameters(get_optimization_container(model))
get_duals(model::AbstractOptimizationModel) = get_duals(get_optimization_container(model))
get_initial_conditions(model::AbstractOptimizationModel) =
    get_initial_conditions(get_optimization_container(model))

get_interval(model::AbstractOptimizationModel) = get_store_params(model).interval

get_run_status(model::AbstractOptimizationModel) = get_run_status(get_simulation_info(model))

set_run_status!(model::AbstractOptimizationModel, status) =
    set_run_status!(get_simulation_info(model), status)
get_time_series_cache(model::AbstractOptimizationModel) =
    get_time_series_cache(get_internal(model))
empty_time_series_cache!(x::AbstractOptimizationModel) = empty!(get_time_series_cache(x))

function get_current_timestamp(model::AbstractOptimizationModel)
    # For EmulationModel interval and resolution are the same.
    return get_initial_time(model) + get_execution_count(model) * get_interval(model)
end

function get_timestamps(model::AbstractOptimizationModel)
    optimization_container = get_optimization_container(model)
    start_time = get_initial_time(optimization_container)
    resolution = get_resolution(model)
    horizon_count = get_time_steps(optimization_container)[end]
    return range(start_time; length = horizon_count, step = resolution)
end

function write_data(model::AbstractOptimizationModel, output_dir::AbstractString; kwargs...)
    write_data(get_optimization_container(model), output_dir; kwargs...)
    return
end

function get_initial_conditions(
    model::AbstractOptimizationModel,
    ::T,
    ::U,
) where {T <: InitialConditionType, U <: IS.InfrastructureSystemsComponent}
    return get_initial_conditions(get_optimization_container(model), T, U)
end

# Called `solve_impl!(model)` in PSI.
function solve_model!(model::AbstractOptimizationModel)
    container = get_optimization_container(model)
    model_name = get_name(model)
    ts = get_current_timestamp(model)
    output_dir = get_output_dir(model)

    if get_export_optimization_model(get_settings(model))
        model_output_dir = joinpath(output_dir, "optimization_model_exports")
        mkpath(model_output_dir)
        tss = replace("$(ts)", ":" => "_")
        model_export_path = joinpath(model_output_dir, "exported_$(model_name)_$(tss).json")
        serialize_optimization_model(container, model_export_path)
        write_lp_file(
            get_jump_model(container),
            replace(model_export_path, ".json" => ".lp"),
        )
    end

    status = execute_optimizer!(container, get_system(model))
    set_run_status!(model, status)
    if status != RunStatus.SUCCESSFULLY_FINALIZED
        settings = get_settings(model)
        infeasible_opt_path = joinpath(output_dir, "infeasible_$(model_name).json")
        @error("Serializing Infeasible Problem at $(infeasible_opt_path)")
        serialize_optimization_model(container, infeasible_opt_path)
        if !get_allow_fails(settings)
            error("Solving model $(model_name) failed at $(ts)")
        else
            @error "Solving model $(model_name) failed at $(ts). Failure Allowed"
        end
    end
    return
end

set_console_level!(model::AbstractOptimizationModel, val) =
    set_console_level!(get_internal(model), val)
set_file_level!(model::AbstractOptimizationModel, val) =
    set_file_level!(get_internal(model), val)
function set_executions!(model::AbstractOptimizationModel, val::Int)
    set_executions!(get_internal(model), val)
    return
end

function set_execution_count!(model::AbstractOptimizationModel, val::Int)
    set_execution_count!(get_internal(model), val)
    return
end

set_initial_time!(model::AbstractOptimizationModel, val::Dates.DateTime) =
    set_initial_time!(get_settings(model), val)

get_simulation_info(model::AbstractOptimizationModel, val) = model.simulation_info = val

function set_status!(model::AbstractOptimizationModel, status::ModelBuildStatus)
    set_status!(get_internal(model), status)
    return
end

function set_output_dir!(model::AbstractOptimizationModel, path::AbstractString)
    set_output_dir!(get_internal(model), path)
    return
end

function advance_execution_count!(model::AbstractOptimizationModel)
    internal = get_internal(model)
    internal.execution_count += 1
    return
end

function _check_numerical_bounds(model::AbstractOptimizationModel)
    variable_bounds = get_variable_numerical_bounds(model)
    if variable_bounds.bounds.max - variable_bounds.bounds.min > 1e9
        @warn "Variable bounds range is $(variable_bounds.bounds.max - variable_bounds.bounds.min) and can result in numerical problems for the solver. \\
        max_bound_variable = $(encode_key_as_string(variable_bounds.bounds.max_index)) \\
        min_bound_variable = $(encode_key_as_string(variable_bounds.bounds.min_index)) \\
        Run get_detailed_variable_numerical_bounds on the model for a deeper analysis"
    else
        @info "Variable bounds range is [$(variable_bounds.bounds.min) $(variable_bounds.bounds.max)]"
    end

    constraint_bounds = get_constraint_numerical_bounds(model)
    if constraint_bounds.coefficient.max - constraint_bounds.coefficient.min > 1e9
        @warn "Constraint coefficient bounds range is $(constraint_bounds.coefficient.max - constraint_bounds.coefficient.min) and can result in numerical problems for the solver. \\
        max_bound_constraint = $(encode_key_as_string(constraint_bounds.coefficient.max_index)) \\
        min_bound_constraint = $(encode_key_as_string(constraint_bounds.coefficient.min_index)) \\
        Run get_detailed_constraint_numerical_bounds on the model for a deeper analysis"
    else
        @info "Constraint coefficient bounds range is [$(constraint_bounds.coefficient.min) $(constraint_bounds.coefficient.max)]"
    end

    if constraint_bounds.rhs.max - constraint_bounds.rhs.min > 1e9
        @warn "Constraint right-hand-side bounds range is $(constraint_bounds.rhs.max - constraint_bounds.rhs.min) and can result in numerical problems for the solver. \\
        max_bound_constraint = $(encode_key_as_string(constraint_bounds.rhs.max_index)) \\
        min_bound_constraint = $(encode_key_as_string(constraint_bounds.rhs.min_index)) \\
        Run get_detailed_constraint_numerical_bounds on the model for a deeper analysis"
    else
        @info "Constraint right-hand-side bounds [$(constraint_bounds.rhs.min) $(constraint_bounds.rhs.max)]"
    end
    return
end

function _pre_solve_model_checks(model::AbstractOptimizationModel, optimizer = nothing)
    jump_model = get_jump_model(model)
    if optimizer !== nothing
        JuMP.set_optimizer(jump_model, optimizer)
    end

    if JuMP.mode(jump_model) != JuMP.DIRECT
        if JuMP.backend(jump_model).state == MOIU.NO_OPTIMIZER
            error("No Optimizer has been defined, can't solve the operational problem")
        end
    else
        @assert get_direct_mode_optimizer(get_settings(model))
    end

    optimizer_name = JuMP.solver_name(jump_model)
    @info "$(get_name(model)) optimizer set to: $optimizer_name"
    settings = get_settings(model)
    if get_check_numerical_bounds(settings)
        @info "Checking Numerical Bounds"
        TimerOutputs.@timeit BUILD_PROBLEMS_TIMER "Numerical Bounds Check" begin
            _check_numerical_bounds(model)
        end
    end
    return
end

function list_names(model::AbstractOptimizationModel, ::Type{T}) where {T <: OptimizationKeyType}
    return encode_keys_as_strings(
        list_keys(get_store(model), T),
    )
end

read_dual(model::AbstractOptimizationModel, key::ConstraintKey) = _read_outputs(model, key)
read_parameter(model::AbstractOptimizationModel, key::ParameterKey) = _read_outputs(model, key)
read_aux_variable(model::AbstractOptimizationModel, key::AuxVarKey) = _read_outputs(model, key)
read_variable(model::AbstractOptimizationModel, key::VariableKey) = _read_outputs(model, key)
read_expression(model::AbstractOptimizationModel, key::ExpressionKey) = _read_outputs(model, key)

function _read_outputs(model::AbstractOptimizationModel, key::OptimizationContainerKey)
    array = read_outputs(get_store(model), key)
    return to_outputs_dataframe(array, nothing, Val(TableFormat.LONG))
end

read_optimizer_stats(model::AbstractOptimizationModel) = read_optimizer_stats(get_store(model))

function add_recorders!(model::AbstractOptimizationModel, recorders)
    internal = get_internal(model)
    for name in recorders
        add_recorder!(internal, name)
    end
end

function register_recorders!(model::AbstractOptimizationModel, file_mode)
    recorder_dir = get_recorder_dir(model)
    mkpath(recorder_dir)
    for name in get_recorders(get_internal(model))
        IS.register_recorder!(name; mode = file_mode, directory = recorder_dir)
    end
end

function unregister_recorders!(model::AbstractOptimizationModel)
    for name in get_recorders(get_internal(model))
        IS.unregister_recorder!(name)
    end
end

const _JUMP_MODEL_FILENAME = "jump_model.json"

function serialize_optimization_model(model::AbstractOptimizationModel)
    serialize_optimization_model(
        get_optimization_container(model),
        joinpath(get_output_dir(model), _JUMP_MODEL_FILENAME),
    )
    return
end

function instantiate_network_model!(model::AbstractOptimizationModel)
    template = get_template(model)
    network_model = get_network_model(template)
    branch_models = get_branch_models(template)
    number_of_steps = get_time_steps(get_optimization_container(model))[end]
    instantiate_network_model!(
        network_model,
        branch_models,
        number_of_steps,
        get_system(model),
    )
    return
end

list_aux_variable_keys(x::AbstractOptimizationModel) = list_keys(get_store(x), AuxVariableType)
list_aux_variable_names(x::AbstractOptimizationModel) = list_names(x, AuxVariableType)
list_variable_keys(x::AbstractOptimizationModel) = list_keys(get_store(x), VariableType)
list_variable_names(x::AbstractOptimizationModel) = list_names(x, VariableType)
list_parameter_keys(x::AbstractOptimizationModel) = list_keys(get_store(x), ParameterType)
list_parameter_names(x::AbstractOptimizationModel) = list_names(x, ParameterType)
list_dual_keys(x::AbstractOptimizationModel) = list_keys(get_store(x), ConstraintType)
list_dual_names(x::AbstractOptimizationModel) = list_names(x, ConstraintType)
list_expression_keys(x::AbstractOptimizationModel) = list_keys(get_store(x), ExpressionType)
list_expression_names(x::AbstractOptimizationModel) = list_names(x, ExpressionType)

function list_all_keys(x::AbstractOptimizationModel)
    return Iterators.flatten(
        list_fields(get_store(x), T) for T in STORE_CONTAINER_TYPES
    )
end

function serialize_optimization_model(model::AbstractOptimizationModel, save_path::String)
    serialize_jump_optimization_model(
        get_jump_model(get_optimization_container(model)),
        save_path,
    )
    return
end
