mutable struct EmulationModel{M <: AbstractOptimizationProblem} <: AbstractOptimizationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::IS.InfrastructureSystemsContainer
    internal::ModelInternal
    simulation_info::SimulationInfo
    store::EmulationModelStore # might be extended to other stores for simulation
    ext::Dict{String, Any}
end

"""
    EmulationModel{M}(
        template::AbstractProblemTemplate,
        sys::IS.InfrastructureSystemsContainer,
        settings::Settings,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        name = nothing) where {M<:AbstractOptimizationProblem}

Settings-taking constructor — finalizes the template, builds the
`OptimizationContainer` with single-time-series data, and validates time
series. Both `finalize_template!` and `validate_time_series!` are extension
points implemented by downstream packages for their concrete template/model
types.
"""
function EmulationModel{M}(
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
    template_ = _deepcopy_template(template)
    finalize_template!(template_, sys)
    internal = ModelInternal(
        OptimizationContainer(sys, settings, jump_model, IS.SingleTimeSeries),
    )
    model = EmulationModel{M}(
        name,
        template_,
        sys,
        internal,
        SimulationInfo(),
        EmulationModelStore(),
        Dict{String, Any}(),
    )
    validate_time_series!(model)
    return model
end

"""
    EmulationModel{M}(
        template::AbstractProblemTemplate,
        sys::IS.InfrastructureSystemsContainer,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        kwargs...) where {M<:AbstractOptimizationProblem}

Kwargs constructor — builds a `Settings` (with `horizon == resolution`, since
emulation models solve one step at a time) and delegates to the Settings-taking
constructor.
"""
function EmulationModel{M}(
    template::AbstractProblemTemplate,
    sys::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    resolution = UNSET_RESOLUTION,
    name = nothing,
    optimizer = nothing,
    warm_start = true,
    initialize_model = true,
    initialization_file = "",
    deserialize_initial_conditions = false,
    export_pwl_vars = false,
    allow_fails = false,
    calculate_conflict = false,
    optimizer_solve_log_print = false,
    detailed_optimizer_stats = false,
    direct_mode_optimizer = false,
    check_numerical_bounds = true,
    store_variable_names = false,
    rebuild_model = false,
    initial_time = UNSET_INI_TIME,
    time_series_cache_size::Int = IS.TIME_SERIES_CACHE_SIZE_BYTES,
) where {M <: AbstractOptimizationProblem}
    settings = Settings(
        sys;
        initial_time = initial_time,
        optimizer = optimizer,
        time_series_cache_size = time_series_cache_size,
        warm_start = warm_start,
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
        horizon = resolution,
        resolution = resolution,
    )
    return EmulationModel{M}(template, sys, settings, jump_model; name = name)
end

"""
    EmulationModel(::Type{M}, template, sys, jump_model=nothing; kwargs...)
        where {M <: AbstractOptimizationProblem}

Type-first dispatch variant.
"""
function EmulationModel(
    ::Type{M},
    template::AbstractProblemTemplate,
    sys::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
) where {M <: AbstractOptimizationProblem}
    return EmulationModel{M}(template, sys, jump_model; kwargs...)
end

get_problem_type(::EmulationModel{M}) where {M <: AbstractOptimizationProblem} = M

function validate_template(::EmulationModel{M}) where {M <: AbstractOptimizationProblem}
    error("validate_template is not implemented for EmulationModel{$M}")
end

function get_current_time(model::EmulationModel)
    execution_count = get_execution_count(model)
    initial_time = get_initial_time(model)
    resolution = get_resolution(model)
    return initial_time + resolution * execution_count
end

function init_model_store_params!(model::EmulationModel)
    num_executions = get_executions(model)
    system = get_system(model)
    settings = get_settings(model)
    horizon = interval = resolution = get_resolution(settings)
    base_power = get_base_power(system)
    sys_uuid = get_system_uuid(system)
    set_store_params!(
        get_internal(model),
        ModelStoreParams(
            num_executions,
            horizon,
            interval,
            resolution,
            base_power,
            sys_uuid,
            get_metadata(get_optimization_container(model)),
        ),
    )
    return
end
