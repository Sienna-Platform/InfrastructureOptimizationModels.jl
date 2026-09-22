mutable struct InvestmentModel{S <: AbstractOptimizationProblem} <:
               AbstractOptimizationModel
    name::Symbol
    template::AbstractProblemTemplate
    portfolio::IS.InfrastructureSystemsContainer
    internal::Union{Nothing, ModelInternal}
    simulation_info::Union{Nothing, SimulationInfo}
    store::InvestmentModelStore
    ext::Dict{String, Any}
end

function InvestmentModel{M}(
    template::AbstractProblemTemplate,
    ::Type{M},
    portfolio::IS.InfrastructureSystemsContainer,
    settings::Settings,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
) where {M <: AbstractOptimizationProblem}
    internal = ModelInternal(OptimizationContainer(portfolio, settings, jump_model))

    model = InvestmentModel{M}(
        :CEM,
        template,
        portfolio,
        internal,
        SimulationInfo(),
        InvestmentModelStore(),
        Dict{String, Any}(),
    )
    return model
end

function InvestmentModel{M}(
    template::AbstractProblemTemplate,
    portfolio::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
    optimizer = nothing,
    horizon = UNSET_HORIZON,
    resolution = UNSET_RESOLUTION,
    portfolio_to_file = true,
    optimizer_solve_log_print = false,
    detailed_optimizer_stats = false,
    calculate_conflict = false,
    direct_mode_optimizer = false,
    store_variable_names = false,
    check_numerical_bounds = true,
    initial_time = UNSET_INI_TIME,
    time_series_cache_size::Int = IS.TIME_SERIES_CACHE_SIZE_BYTES,
) where {M <: AbstractOptimizationProblem}
    settings = Settings(
        portfolio;
        initial_time = initial_time,
        time_series_cache_size = time_series_cache_size,
        horizon = horizon,
        resolution = resolution,
        optimizer = optimizer,
        direct_mode_optimizer = direct_mode_optimizer,
        optimizer_solve_log_print = optimizer_solve_log_print,
        detailed_optimizer_stats = detailed_optimizer_stats,
        calculate_conflict = calculate_conflict,
        system_to_file = portfolio_to_file,
        check_numerical_bounds = check_numerical_bounds,
        store_variable_names = store_variable_names,
    )
    return InvestmentModel{M}(template, M, portfolio, settings, jump_model)
end

# Default implementations of getter/setter functions for InvestmentModel.
is_built(model::InvestmentModel) =
    get_status(get_internal(model)) == ModelBuildStatus.BUILT
isempty(model::InvestmentModel) =
    get_status(get_internal(model)) == ModelBuildStatus.EMPTY

get_constraints(model::InvestmentModel) =
    get_constraints(get_internal(model))
get_internal(model::InvestmentModel) = model.internal

function get_jump_model(model::InvestmentModel)
    return get_jump_model(get_container(get_internal(model)))
end

get_name(model::InvestmentModel) = model.name
get_store(model::InvestmentModel) = model.store

function get_optimization_container(model::InvestmentModel)
    return get_optimization_container(get_internal(model))
end

function get_timestamps(model::InvestmentModel)
    optimization_container = get_optimization_container(model)
    start_time = get_initial_time(optimization_container)
    resolution = get_resolution(model)
    horizon_count = get_time_steps(optimization_container)[end]
    return range(start_time; length = horizon_count, step = resolution)
end

# No Base Power for Portfolio models. Always in Natural Units.
get_problem_base_power(model::InvestmentModel) = 1.0
get_settings(model::InvestmentModel) = get_optimization_container(model).settings
get_optimizer_stats(model::InvestmentModel) =
    get_optimizer_stats(get_optimization_container(model))

get_status(model::InvestmentModel) = get_status(get_internal(model))
get_portfolio(model::InvestmentModel) = model.portfolio
get_template(model::InvestmentModel) = model.template
get_time_stamps(model::InvestmentModel) =
    get_time_stamps(get_time_mapping(get_optimization_container(model)))

get_store_params(model::InvestmentModel) =
    get_store_params(get_internal(model))
get_output_dir(model::InvestmentModel) = get_output_dir(get_internal(model))
get_recorder_dir(model::InvestmentModel) = joinpath(get_output_dir(model), "recorder")

get_variables(model::InvestmentModel) = get_variables(get_optimization_container(model))
get_duals(model::InvestmentModel) = get_duals(get_optimization_container(model))
get_initial_conditions(model::InvestmentModel) =
    get_initial_conditions(get_optimization_container(model))

get_simulation_info(model::InvestmentModel) = model.simulation_info
get_executions(model::InvestmentModel) = get_executions(get_internal(model))

get_run_status(model::InvestmentModel) = get_run_status(get_simulation_info(model))
set_run_status!(model::InvestmentModel, status) =
    set_run_status!(get_simulation_info(model), status)

get_initial_time(model::InvestmentModel) = get_initial_time(get_settings(model))
get_resolution(model::InvestmentModel) = get_resolution(get_settings(model))

set_console_level!(model::InvestmentModel, val) =
    set_console_level!(get_internal(model), val)
set_file_level!(model::InvestmentModel, val) =
    set_file_level!(get_internal(model), val)

function set_status!(model::InvestmentModel, status::ModelBuildStatus)
    set_status!(get_internal(model), status)
    return
end

function set_output_dir!(model::InvestmentModel, path::AbstractString)
    set_output_dir!(get_internal(model), path)
    return
end

# Portfolio-specific alias for Setttings
get_portfolio_to_file(settings::Settings) = get_system_to_file(settings)
