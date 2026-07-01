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
is_synchronized(model::AbstractOptimizationModel) =
    is_synchronized(get_optimization_container(model))

function get_rebuild_model(model::AbstractOptimizationModel)
    sim_info = model.simulation_info
    if isnothing(sim_info)
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
get_log_file(model::AbstractOptimizationModel) =
    joinpath(get_output_dir(model), PROBLEM_LOG_FILENAME)
get_store_params(model::AbstractOptimizationModel) =
    get_store_params(get_internal(model))
get_output_dir(model::AbstractOptimizationModel) = get_output_dir(get_internal(model))
get_initial_conditions_file(model::AbstractOptimizationModel) =
    joinpath(get_output_dir(model), "initial_conditions.bin")
get_recorder_dir(model::AbstractOptimizationModel) =
    joinpath(get_output_dir(model), "recorder")
get_variables(model::AbstractOptimizationModel) =
    get_variables(get_optimization_container(model))
get_parameters(model::AbstractOptimizationModel) =
    get_parameters(get_optimization_container(model))
get_duals(model::AbstractOptimizationModel) = get_duals(get_optimization_container(model))
get_initial_conditions(model::AbstractOptimizationModel) =
    get_initial_conditions(get_optimization_container(model))

get_interval(model::AbstractOptimizationModel) = get_store_params(model).interval

get_run_status(model::AbstractOptimizationModel) =
    get_run_status(get_simulation_info(model))

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

    fmt = get_export_optimization_model(get_settings(model))
    if fmt != OptimizationModelExportFormat.NONE
        model_output_dir = joinpath(output_dir, "optimization_model_exports")
        mkpath(model_output_dir)
        tss = replace("$(ts)", ":" => "_")
        ext = fmt == OptimizationModelExportFormat.LP ? "lp" : "json"
        model_export_path =
            joinpath(model_output_dir, "exported_$(model_name)_$(tss).$(ext)")
        serialize_optimization_model(model, model_export_path, fmt)
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

function set_simulation_info!(model::AbstractOptimizationModel, val)
    model.simulation_info = val
    return
end

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
        max_bound_variable = $(_bound_index_string(variable_bounds.bounds.max_index)) \\
        min_bound_variable = $(_bound_index_string(variable_bounds.bounds.min_index)) \\
        Run get_detailed_variable_numerical_bounds on the model for a deeper analysis"
    else
        @info "Variable bounds range is [$(variable_bounds.bounds.min) $(variable_bounds.bounds.max)]"
    end

    constraint_bounds = get_constraint_numerical_bounds(model)
    if constraint_bounds.coefficient.max - constraint_bounds.coefficient.min > 1e9
        @warn "Constraint coefficient bounds range is $(constraint_bounds.coefficient.max - constraint_bounds.coefficient.min) and can result in numerical problems for the solver. \\
        max_bound_constraint = $(_bound_index_string(constraint_bounds.coefficient.max_index)) \\
        min_bound_constraint = $(_bound_index_string(constraint_bounds.coefficient.min_index)) \\
        Run get_detailed_constraint_numerical_bounds on the model for a deeper analysis"
    else
        @info "Constraint coefficient bounds range is [$(constraint_bounds.coefficient.min) $(constraint_bounds.coefficient.max)]"
    end

    if constraint_bounds.rhs.max - constraint_bounds.rhs.min > 1e9
        @warn "Constraint right-hand-side bounds range is $(constraint_bounds.rhs.max - constraint_bounds.rhs.min) and can result in numerical problems for the solver. \\
        max_bound_constraint = $(_bound_index_string(constraint_bounds.rhs.max_index)) \\
        min_bound_constraint = $(_bound_index_string(constraint_bounds.rhs.min_index)) \\
        Run get_detailed_constraint_numerical_bounds on the model for a deeper analysis"
    else
        @info "Constraint right-hand-side bounds [$(constraint_bounds.rhs.min) $(constraint_bounds.rhs.max)]"
    end
    return
end

function _pre_solve_model_checks(model::AbstractOptimizationModel, optimizer = nothing)
    jump_model = get_jump_model(model)
    if !isnothing(optimizer)
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

function list_names(
    model::AbstractOptimizationModel,
    ::Type{T},
) where {T <: OptimizationKeyType}
    return encode_keys_as_strings(
        list_keys(get_store(model), T),
    )
end

read_dual(model::AbstractOptimizationModel, key::ConstraintKey) = _read_outputs(model, key)
read_parameter(model::AbstractOptimizationModel, key::ParameterKey) =
    _read_outputs(model, key)
read_aux_variable(model::AbstractOptimizationModel, key::AuxVarKey) =
    _read_outputs(model, key)
read_variable(model::AbstractOptimizationModel, key::VariableKey) =
    _read_outputs(model, key)
read_expression(model::AbstractOptimizationModel, key::ExpressionKey) =
    _read_outputs(model, key)

function _read_outputs(model::AbstractOptimizationModel, key::OptimizationContainerKey)
    array = read_outputs(get_store(model), key)
    return to_outputs_dataframe(array, nothing, Val(TableFormat.LONG))
end

read_optimizer_stats(model::AbstractOptimizationModel) =
    read_optimizer_stats(get_store(model))

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

list_aux_variable_keys(x::AbstractOptimizationModel) =
    list_keys(get_store(x), AuxVariableType)
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

wait_for_serialization!(model::AbstractOptimizationModel) =
    wait_for_serialization!(get_optimization_container(model))

function serialize_optimization_model(
    model::AbstractOptimizationModel,
    save_path::String,
    fmt::OptimizationModelExportFormat,
)
    container = get_optimization_container(model)
    dest = _copy_jump_model_for_export(get_jump_model(container), fmt)
    if Threads.nthreads() > 1
        wait_for_serialization!(container)
        task = Threads.@spawn _write_export_model(dest, save_path)
        set_serialization_task!(container, task)
    else
        _write_export_model(dest, save_path)
    end
    return
end

#################################################################################
# Generic build/solve/run lifecycle (domain-neutral; dispatched on the wrapper).
# The two `DecisionModel`/`EmulationModel` behavioral differences are factored into
# the `_mark_recurrent_solves!`, `_apply_build_executions!`, and `reset_time_series_type`
# hooks (defined per-wrapper in decision_model.jl / emulation_model.jl). Power-specific
# steps go through the `build_problem!`/`build_initial_conditions!` extension points.

# EmulationModels always solve recurrently; DecisionModels leave the container flag as-is
# (so a DecisionModel built inside a simulation keeps its externally-set value).
_mark_recurrent_solves!(::AbstractOptimizationModel) = nothing

# Standalone `build!` only applies an execution count for models that solve recurrently.
_apply_build_executions!(::AbstractOptimizationModel, executions) = nothing

function build_pre_step!(model::AbstractOptimizationModel)
    TimerOutputs.@timeit BUILD_PROBLEMS_TIMER "Build pre-step" begin
        validate_template(model)
        if !isempty(model)
            @info "OptimizationProblem status not ModelBuildStatus.EMPTY. Resetting"
            reset!(model)
        end
        _mark_recurrent_solves!(model)
        @info "Initializing Optimization Container"
        init_optimization_container!(
            get_optimization_container(model),
            get_network_model(get_template(model)),
            get_system(model),
        )
        @info "Initializing ModelStoreParams"
        init_model_store_params!(model)
        set_status!(model, ModelBuildStatus.IN_PROGRESS)
    end
    return
end

# Called `build_impl!(model)` in PSI.
function build_model!(model::AbstractOptimizationModel)
    build_pre_step!(model)
    @info "Instantiating Network Model"
    instantiate_network_model!(model)
    handle_initial_conditions!(model)
    build_problem!(
        get_optimization_container(model),
        get_template(model),
        get_system(model),
    )
    serialize_metadata!(get_optimization_container(model), get_output_dir(model))
    log_values(get_settings(model))
    return
end

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

function handle_initial_conditions!(model::AbstractOptimizationModel)
    TimerOutputs.@timeit BUILD_PROBLEMS_TIMER "Model Initialization" begin
        if isempty(get_template(model))
            return
        end
        settings = get_settings(model)
        initialize_model = get_initialize_model(settings)
        deserialize_initial_conditions = get_deserialize_initial_conditions(settings)
        serialized_initial_conditions_file = get_initial_conditions_file(model)
        custom_init_file = get_initialization_file(settings)

        if !initialize_model && deserialize_initial_conditions
            throw(
                IS.ConflictingInputsError(
                    "!initialize_model && deserialize_initial_conditions",
                ),
            )
        elseif !initialize_model && !isempty(custom_init_file)
            throw(IS.ConflictingInputsError("!initialize_model && initialization_file"))
        end

        if !initialize_model
            @info "Skip build of initial conditions"
            return
        end

        if !isempty(custom_init_file)
            if !isfile(custom_init_file)
                error("initialization_file = $custom_init_file does not exist")
            end
            if abspath(custom_init_file) != abspath(serialized_initial_conditions_file)
                cp(custom_init_file, serialized_initial_conditions_file; force = true)
            end
        end

        if deserialize_initial_conditions && isfile(serialized_initial_conditions_file)
            set_initial_conditions_data!(
                get_optimization_container(model),
                Serialization.deserialize(serialized_initial_conditions_file),
            )
            @info "Deserialized initial_conditions_data"
        else
            @info "Make Initial Conditions Model"
            build_initial_conditions!(model)
            solve_and_write_initial_conditions!(model)
        end
        set_initial_conditions_model_container!(get_internal(model), nothing)
    end
    return
end

function reset!(model::AbstractOptimizationModel)
    was_built_for_recurrent_solves = built_for_recurrent_solves(model)
    if was_built_for_recurrent_solves
        set_execution_count!(model, 0)
    end
    set_container!(
        get_internal(model),
        OptimizationContainer(
            get_system(model),
            get_settings(model),
            nothing,
            reset_time_series_type(model),
        ),
    )
    get_optimization_container(model).built_for_recurrent_solves =
        was_built_for_recurrent_solves
    set_initial_conditions_model_container!(get_internal(model), nothing)
    empty_time_series_cache!(model)
    empty!(get_store(model))
    set_status!(model, ModelBuildStatus.EMPTY)
    return
end

"""
Build an operation model. `executions` is honored only by models that solve
recurrently (`EmulationModel`).

# Arguments

  - `model::AbstractOptimizationModel`: the operation model
  - `output_dir::String`: Output directory for results
  - `recorders::Vector{Symbol} = []`: recorder names to register
  - `console_level = Logging.Error`
  - `file_level = Logging.Info`
  - `disable_timer_outputs = false`: Enable/Disable timing outputs
  - `store_system_in_results::Bool = true`: store the system as JSON in the results HDF5 file
"""
function build!(
    model::AbstractOptimizationModel;
    executions = 1,
    output_dir::String,
    recorders = Symbol[],
    console_level = Logging.Error,
    file_level = Logging.Info,
    disable_timer_outputs = false,
    store_system_in_results = true,
)
    mkpath(output_dir)
    set_output_dir!(model, output_dir)
    set_console_level!(model, console_level)
    set_file_level!(model, file_level)
    TimerOutputs.reset_timer!(BUILD_PROBLEMS_TIMER)
    disable_timer_outputs && TimerOutputs.disable_timer!(BUILD_PROBLEMS_TIMER)
    file_mode = "w"
    add_recorders!(model, recorders)
    register_recorders!(model, file_mode)
    logger = configure_logging(get_internal(model), PROBLEM_LOG_FILENAME, file_mode)
    if store_system_in_results
        @warn "store_system_in_results is set to true. This will do nothing unless a Simulation is being built."
    end
    try
        Logging.with_logger(logger) do
            try
                _apply_build_executions!(model, executions)
                TimerOutputs.@timeit BUILD_PROBLEMS_TIMER "Problem $(get_name(model))" begin
                    build_model!(model)
                end
                set_status!(model, ModelBuildStatus.BUILT)
                @info "\n$(BUILD_PROBLEMS_TIMER)\n"
            catch e
                set_status!(model, ModelBuildStatus.FAILED)
                bt = catch_backtrace()
                @error "$(nameof(typeof(model))) Build Failed" exception = e, bt
            end
        end
    finally
        unregister_recorders!(model)
        close(logger)
    end
    return get_status(model)
end

# Execution-time kwargs consumed by `_execute_model!`, not by `build!`.
const _EXECUTION_KWARGS = (:optimizer, :enable_progress_bar)

function build_if_not_already_built!(model::AbstractOptimizationModel; kwargs...)
    status = get_status(model)
    if status == ModelBuildStatus.EMPTY
        if !haskey(kwargs, :output_dir)
            error(
                "'output_dir' must be provided as a kwarg if the model build status is $status",
            )
        else
            new_kwargs = Dict(k => v for (k, v) in kwargs if k ∉ _EXECUTION_KWARGS)
            status = build!(model; new_kwargs...)
        end
    end
    if status != ModelBuildStatus.BUILT
        error("build! of the $(typeof(model)) $(get_name(model)) failed: $status")
    end
    return
end

function execute_model!(
    model::AbstractOptimizationModel;
    export_problem_outputs = false,
    console_level = Logging.Error,
    file_level = Logging.Info,
    disable_timer_outputs = false,
    export_optimization_problem = true,
    store_system_in_results = true,
    kwargs...,
)
    if store_system_in_results
        @warn "store_system_in_results is set to true. This will do nothing unless a Simulation is being built."
    end
    build_if_not_already_built!(
        model;
        console_level = console_level,
        file_level = file_level,
        disable_timer_outputs = disable_timer_outputs,
        kwargs...,
    )
    set_console_level!(model, console_level)
    set_file_level!(model, file_level)
    TimerOutputs.reset_timer!(RUN_OPERATION_MODEL_TIMER)
    disable_timer_outputs && TimerOutputs.disable_timer!(RUN_OPERATION_MODEL_TIMER)

    file_mode = "a"
    register_recorders!(model, file_mode)
    logger = configure_logging(get_internal(model), PROBLEM_LOG_FILENAME, file_mode)
    try
        Logging.with_logger(logger) do
            try
                initialize_storage!(
                    get_store(model),
                    get_optimization_container(model),
                    get_store_params(model),
                )
                _execute_model!(model; kwargs...)
                if export_optimization_problem
                    TimerOutputs.@timeit RUN_OPERATION_MODEL_TIMER "Serialize" begin
                        serialize_optimization_model(model)
                    end
                end
                TimerOutputs.@timeit RUN_OPERATION_MODEL_TIMER "Outputs processing" begin
                    outputs = OptimizationProblemOutputs(model)
                    serialize_outputs(outputs, get_output_dir(model))
                    export_problem_outputs && export_outputs(outputs)
                end
                @info "\n$(RUN_OPERATION_MODEL_TIMER)\n"
            catch e
                @error "$(get_name(model)) failed" exception = (e, catch_backtrace())
                set_run_status!(model, RunStatus.FAILED)
            end
        end
    finally
        wait_for_serialization!(model)
        unregister_recorders!(model)
        close(logger)
    end
    return get_run_status(model)
end
