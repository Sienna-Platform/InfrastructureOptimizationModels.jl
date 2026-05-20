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

mutable struct DecisionModel{M <: AbstractOptimizationProblem} <: OperationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::IS.InfrastructureSystemsContainer
    internal::Union{Nothing, ModelInternal}
    simulation_info::Union{Nothing, SimulationInfo}
    store::DecisionModelStore
    ext::Dict{String, Any}
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
