"""
Abstract type for models that use default InfrastructureOptimizationModels formulations. For custom decision problems
    use DecisionProblem as the super type.
"""
abstract type DefaultDecisionProblem <: DecisionProblem end

"""
Generic InfrastructureOptimizationModels Operation Problem Type for unspecified models
"""
struct GenericOpProblem <: DefaultDecisionProblem end

mutable struct DecisionModel{M <: DecisionProblem} <: OperationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::PSY.System
    internal::Union{Nothing, ModelInternal}
    simulation_info::Union{Nothing, SimulationInfo}
    store::DecisionModelStore
    ext::Dict{String, Any}
end

"""
    DecisionModel{M}(
        template::AbstractProblemTemplate,
        sys::PSY.System,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        kwargs...) where {M<:DecisionProblem}

Build the optimization problem of type M with the specific system and template.

# Arguments

  - `::Type{M} where M<:DecisionProblem`: The abstract operation model type
  - `template::AbstractProblemTemplate`: The model reference made up of transmission, devices, branches, and services.
  - `sys::PSY.System`: the system created using Power Systems
  - `jump_model::Union{Nothing, JuMP.Model}`: Enables passing a custom JuMP model. Use with care
  - `name = nothing`: name of model, string or symbol; defaults to the type of template converted to a symbol.
  - `optimizer::Union{Nothing,MOI.OptimizerWithAttributes} = nothing` : The optimizer does
    not get serialized. Callers should pass whatever they passed to the original problem.
  - `horizon::Dates.Period = UNSET_HORIZON`: Manually specify the length of the forecast Horizon
  - `resolution::Dates.Period = UNSET_RESOLUTION`: Manually specify the model's resolution
  - `warm_start::Bool = true`: True will use the current operation point in the system to initialize variable values. False initializes all variables to zero. Default is true
  - `check_components::Bool = true`: True to check the components valid fields when building
  - `initialize_model::Bool = true`: Option to decide to initialize the model or not.
  - `initialization_file::String = ""`: This allows to pass pre-existing initialization values to avoid the solution of an optimization problem to find feasible initial conditions.
  - `deserialize_initial_conditions::Bool = false`: Option to deserialize conditions
  - `export_pwl_vars::Bool = false`: True to export all the pwl intermediate variables. It can slow down significantly the build and solve time.
  - `allow_fails::Bool = false`: True to allow the simulation to continue even if the optimization step fails. Use with care.
  - `optimizer_solve_log_print::Bool = false`: Uses JuMP.unset_silent() to print the optimizer's log. By default all solvers are set to MOI.Silent()
  - `detailed_optimizer_stats::Bool = false`: True to save detailed optimizer stats log.
  - `calculate_conflict::Bool = false`: True to use solver to calculate conflicts for infeasible problems. Only specific solvers are able to calculate conflicts.
  - `direct_mode_optimizer::Bool = false`: True to use the solver in direct mode. Creates a [JuMP.direct_model](https://jump.dev/JuMP.jl/dev/reference/models/#JuMP.direct_model).
  - `store_variable_names::Bool = false`: to store variable names in optimization model. Decreases the build times.
  - `rebuild_model::Bool = false`: It will force the rebuild of the underlying JuMP model with each call to update the model. It increases solution times, use only if the model can't be updated in memory.
  - `initial_time::Dates.DateTime = UNSET_INI_TIME`: Initial Time for the model solve.
  - `time_series_cache_size::Int = IS.TIME_SERIES_CACHE_SIZE_BYTES`: Size in bytes to cache for each time array. Default is 1 MiB. Set to 0 to disable.

# Example

```julia
template = ProblemTemplate(CopperPlatePowerModel, devices, branches, services)
OpModel = DecisionModel(MockOperationProblem, template, system)
```
"""
function DecisionModel{M}(
    template::AbstractProblemTemplate,
    sys::PSY.System,
    settings::Settings,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
) where {M <: DecisionProblem}
    if name === nothing
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

function DecisionModel{M}(
    template::AbstractProblemTemplate,
    sys::PSY.System,
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
) where {M <: DecisionProblem}
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
Build the optimization problem of type M with the specific system and template

# Arguments

  - `::Type{M} where M<:DecisionProblem`: The abstract operation model type
  - `template::AbstractProblemTemplate`: The model reference made up of transmission, devices, branches, and services.
  - `sys::PSY.System`: the system created using Power Systems
  - `jump_model::Union{Nothing, JuMP.Model}` = nothing: Enables passing a custom JuMP model. Use with care.

# Example

```julia
template = ProblemTemplate(CopperPlatePowerModel, devices, branches, services)
problem = DecisionModel(MyOpProblemType, template, system, optimizer)
```
"""
function DecisionModel(
    ::Type{M},
    template::AbstractProblemTemplate,
    sys::PSY.System,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
) where {M <: DecisionProblem}
    return DecisionModel{M}(template, sys, jump_model; kwargs...)
end

function DecisionModel(
    template::AbstractProblemTemplate,
    sys::PSY.System,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
)
    return DecisionModel{GenericOpProblem}(template, sys, jump_model; kwargs...)
end

function DecisionModel{M}(
    sys::PSY.System,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
) where {M <: DefaultDecisionProblem}
    IS.ArgumentError(
        "DefaultDecisionProblem subtypes require a template. Use DecisionModel subtyping instead.",
    )
end

get_problem_type(::DecisionModel{M}) where {M <: DecisionProblem} = M

function validate_template(::DecisionModel{M}) where {M <: DecisionProblem}
    error("validate_template is not implemented for DecisionModel{$M}")
end

function validate_template(::DecisionModel{<:DefaultDecisionProblem})
    return nothing
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
        interval = PSY.get_forecast_interval(system)
    end
    resolution = get_resolution(model)
    base_power = PSY.get_base_power(system)
    sys_uuid = IS.get_uuid(system)
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

function validate_time_series!(model::DecisionModel{<:DefaultDecisionProblem})
    sys = get_system(model)
    settings = get_settings(model)
    available_resolutions = PSY.get_time_series_resolutions(sys)

    if get_resolution(settings) == UNSET_RESOLUTION && length(available_resolutions) != 1
        throw(
            IS.ConflictingInputsError(
                "Data contains multiple resolutions, the resolution keyword argument must be added to the Model. Time Series Resolutions: $(available_resolutions)",
            ),
        )
    elseif get_resolution(settings) != UNSET_RESOLUTION && length(available_resolutions) > 1
        if get_resolution(settings) ∉ available_resolutions
            throw(
                IS.ConflictingInputsError(
                    "Resolution $(get_resolution(settings)) is not available in the system data. Time Series Resolutions: $(available_resolutions)",
                ),
            )
        end
    else
        set_resolution!(settings, first(available_resolutions))
    end

    model_interval = get_interval(settings)
    available_intervals = get_forecast_intervals(sys)
    if model_interval == UNSET_INTERVAL && length(available_intervals) > 1
        throw(
            IS.ConflictingInputsError(
                "The system contains multiple forecast intervals $(available_intervals). " *
                "The `interval` keyword argument must be provided to the DecisionModel constructor " *
                "to select which interval to use.",
            ),
        )
    elseif model_interval != UNSET_INTERVAL && !isempty(available_intervals)
        if model_interval ∉ available_intervals
            throw(
                IS.ConflictingInputsError(
                    "Interval $(Dates.canonicalize(model_interval)) is not available in the system data. " *
                    "Available forecast intervals: $(available_intervals)",
                ),
            )
        end
    end
    interval_kwarg =
        model_interval == UNSET_INTERVAL ? (;) : (; interval = model_interval)
    if get_horizon(settings) == UNSET_HORIZON
        set_horizon!(settings, PSY.get_forecast_horizon(sys; interval_kwarg...))
    end

    counts = PSY.get_time_series_counts(sys)
    if counts.forecast_count < 1
        error(
            "The system does not contain forecast data. A DecisionModel can't be built.",
        )
    end
    return
end

get_horizon(model::DecisionModel) = get_horizon(get_settings(model))
