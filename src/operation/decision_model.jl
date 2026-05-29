function get_deterministic_time_series_type(sys::IS.InfrastructureSystemsContainer)
    time_series_types = get_time_series_counts_by_type(sys)
    existing_types = Set(d["type"] for d in time_series_types)
    if ("Deterministic" in existing_types) &&
       ("DeterministicSingleTimeSeries" in existing_types)
        error(
            "The System contains a combination of forecast data and transformed time series data. Currently this is not supported.",
        )
    end
    if "Deterministic" ∈ existing_types
        return IS.Deterministic
    elseif "DeterministicSingleTimeSeries" ∈ existing_types
        return IS.DeterministicSingleTimeSeries
    else
        error(
            "The System does not contain any forecast data or transformed time series data.",
        )
    end
end

mutable struct DecisionModel{M <: AbstractOptimizationProblem} <: AbstractOptimizationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::IS.InfrastructureSystemsContainer
    internal::Union{Nothing, ModelInternal}
    simulation_info::Union{Nothing, SimulationInfo}
    store::DecisionModelStore
    ext::Dict{String, Any}
end

"""
    DecisionModel{M}(
        template::AbstractProblemTemplate,
        sys::IS.InfrastructureSystemsContainer,
        settings::Settings,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        name = nothing) where {M<:AbstractOptimizationProblem}

Settings-taking constructor — builds the `OptimizationContainer`, finalizes
the template, and validates time series. Both `finalize_template!` and
`validate_time_series!` are extension points: downstream packages provide
methods for their concrete `AbstractProblemTemplate` and model types.
"""
function DecisionModel{M}(
    template::AbstractProblemTemplate,
    sys::IS.InfrastructureSystemsContainer,
    settings::Settings,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
) where {M <: AbstractOptimizationProblem}
    if isnothing(name)
        name = nameof(M)
    elseif name isa String
        name = Symbol(name)
    end
    auto_transform_time_series!(sys, settings)
    ts_type = get_deterministic_time_series_type(sys)
    internal = ModelInternal(
        OptimizationContainer(sys, settings, jump_model, ts_type),
    )

    template_ = deepcopy(template)
    finalize_template!(template_, sys)
    model = DecisionModel{M}(
        name,
        template_,
        sys,
        internal,
        SimulationInfo(),
        DecisionModelStore(),
        Dict{String, Any}(),
    )
    validate_time_series!(model)
    return model
end

"""
    DecisionModel{M}(
        template::AbstractProblemTemplate,
        sys::IS.InfrastructureSystemsContainer,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        kwargs...) where {M<:AbstractOptimizationProblem}

Kwargs constructor — accepts horizon/resolution/interval and all the standard
solver/model settings, builds a `Settings`, and delegates to the
settings-taking constructor.
"""
function DecisionModel{M}(
    template::AbstractProblemTemplate,
    sys::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
    optimizer = nothing,
    horizon = UNSET_HORIZON,
    resolution = UNSET_RESOLUTION,
    interval = UNSET_INTERVAL,
    warm_start = true,
    check_components = true,
    initialize_model = true,
    initialization_file = "",
    deserialize_initial_conditions = false,
    export_pwl_vars = false,
    allow_fails = false,
    optimizer_solve_log_print = false,
    detailed_optimizer_stats = false,
    calculate_conflict = false,
    direct_mode_optimizer = false,
    store_variable_names = false,
    rebuild_model = false,
    export_optimization_model = false,
    check_numerical_bounds = true,
    initial_time = UNSET_INI_TIME,
    time_series_cache_size::Int = IS.TIME_SERIES_CACHE_SIZE_BYTES,
) where {M <: AbstractOptimizationProblem}
    settings = Settings(
        sys;
        horizon = horizon,
        resolution = resolution,
        interval = interval,
        initial_time = initial_time,
        optimizer = optimizer,
        time_series_cache_size = time_series_cache_size,
        warm_start = warm_start,
        check_components = check_components,
        initialize_model = initialize_model,
        initialization_file = initialization_file,
        deserialize_initial_conditions = deserialize_initial_conditions,
        export_pwl_vars = export_pwl_vars,
        allow_fails = allow_fails,
        calculate_conflict = calculate_conflict,
        optimizer_solve_log_print = optimizer_solve_log_print,
        detailed_optimizer_stats = detailed_optimizer_stats,
        direct_mode_optimizer = direct_mode_optimizer,
        check_numerical_bounds = check_numerical_bounds,
        store_variable_names = store_variable_names,
        rebuild_model = rebuild_model,
        export_optimization_model = export_optimization_model,
    )
    return DecisionModel{M}(template, sys, settings, jump_model; name = name)
end

"""
    DecisionModel(::Type{M}, template, sys, jump_model=nothing; kwargs...)
        where {M <: AbstractOptimizationProblem}

Type-first dispatch variant.
"""
function DecisionModel(
    ::Type{M},
    template::AbstractProblemTemplate,
    sys::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
) where {M <: AbstractOptimizationProblem}
    return DecisionModel{M}(template, sys, jump_model; kwargs...)
end

get_problem_type(::DecisionModel{M}) where {M <: AbstractOptimizationProblem} = M

function validate_template(::DecisionModel{M}) where {M <: AbstractOptimizationProblem}
    error("validate_template is not implemented for DecisionModel{$M}")
end

# Probably could be more efficient by storing the info in the internal
function get_current_time(model::DecisionModel)
    execution_count = get_execution_count(model)
    initial_time = get_initial_time(model)
    interval = get_interval(model)
    return initial_time + interval * execution_count
end

function init_model_store_params!(model::DecisionModel)
    num_executions = get_executions(model)
    horizon = get_horizon(model)
    system = get_system(model)
    settings = get_settings(model)
    model_interval = get_interval(settings)
    if model_interval != UNSET_INTERVAL
        interval = model_interval
    else
        interval = get_forecast_interval(system)
    end
    resolution = get_resolution(model)
    base_power = get_base_power(system)
    sys_uuid = get_system_uuid(system)
    store_params = ModelStoreParams(
        num_executions,
        horizon,
        iszero(interval) ? resolution : interval,
        resolution,
        base_power,
        sys_uuid,
        get_metadata(get_optimization_container(model)),
    )
    set_store_params!(get_internal(model), store_params)
    return
end

get_horizon(model::DecisionModel) = get_horizon(get_settings(model))
