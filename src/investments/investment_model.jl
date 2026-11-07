mutable struct InvestmentModel{S <: AbstractOptimizationProblem} <:
               AbstractOptimizationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::IS.InfrastructureSystemsContainer
    internal::Union{Nothing, ModelInternal}
    simulation_info::Union{Nothing, SimulationInfo}
    store::InvestmentModelStore
    ext::Dict{String, Any}
end

"""
    InvestmentModel{M}(
        template::AbstractProblemTemplate,
        portfolio::IS.InfrastructureSystemsContainer,
        settings::Settings,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        name = nothing) where {M<:AbstractOptimizationProblem}

Settings-taking constructor — builds the `OptimizationContainer`, finalizes a copy of the
template, and validates time series. Both `finalize_template!` and `validate_time_series!`
are extension points: downstream packages provide methods for their concrete
`AbstractProblemTemplate` and model types. `name` defaults to the problem type's name.
"""
function InvestmentModel{M}(
    template::AbstractProblemTemplate,
    portfolio::IS.InfrastructureSystemsContainer,
    settings::Settings,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
) where {M <: AbstractOptimizationProblem}
    if isnothing(name)
        name = nameof(M)
    elseif name isa String
        name = Symbol(name)
    end

    internal = ModelInternal(
        OptimizationContainer(portfolio, settings, jump_model, IS.SingleTimeSeries),
    )

    template_ = deepcopy(template)
    share_template_references!(template_, template)
    finalize_template!(template_, portfolio)
    model = InvestmentModel{M}(
        name,
        template_,
        portfolio,
        internal,
        SimulationInfo(),
        InvestmentModelStore(),
        Dict{String, Any}(),
    )
    validate_time_series!(model)
    return model
end

"""
    InvestmentModel{M}(
        template::AbstractProblemTemplate,
        portfolio::IS.InfrastructureSystemsContainer,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        kwargs...) where {M<:AbstractOptimizationProblem}

Kwargs constructor — builds a `Settings` and delegates to the settings-taking constructor.
`name` defaults to the problem type's name.

Accepts the settings that apply to investment models. The operational-only
settings (`interval`, `allow_fails`, `rebuild_model`, `initialize_model`,
`initialization_file`, `deserialize_initial_conditions`, `check_components`,
`export_pwl_vars`) are not accepted. `portfolio_to_file` is the investment name for
`system_to_file`.
"""
function InvestmentModel{M}(
    template::AbstractProblemTemplate,
    portfolio::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    name = nothing,
    optimizer = nothing,
    horizon = UNSET_HORIZON,
    resolution = UNSET_RESOLUTION,
    warm_start = true,
    portfolio_to_file = true,
    optimizer_solve_log_print = false,
    detailed_optimizer_stats = false,
    calculate_conflict = false,
    direct_mode_optimizer = false,
    store_variable_names = false,
    export_optimization_model = OptimizationModelExportFormat.NONE,
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
        warm_start = warm_start,
        direct_mode_optimizer = direct_mode_optimizer,
        optimizer_solve_log_print = optimizer_solve_log_print,
        detailed_optimizer_stats = detailed_optimizer_stats,
        calculate_conflict = calculate_conflict,
        system_to_file = portfolio_to_file,
        check_numerical_bounds = check_numerical_bounds,
        store_variable_names = store_variable_names,
        export_optimization_model = export_optimization_model,
    )
    return InvestmentModel{M}(template, portfolio, settings, jump_model; name = name)
end

"""
    InvestmentModel(
        ::Type{M},
        template::AbstractProblemTemplate,
        portfolio::IS.InfrastructureSystemsContainer,
        jump_model::Union{Nothing, JuMP.Model}=nothing;
        kwargs...) where {M<:AbstractOptimizationProblem}

Type-first dispatch variant.
"""
function InvestmentModel(
    ::Type{M},
    template::AbstractProblemTemplate,
    portfolio::IS.InfrastructureSystemsContainer,
    jump_model::Union{Nothing, JuMP.Model} = nothing;
    kwargs...,
) where {M <: AbstractOptimizationProblem}
    return InvestmentModel{M}(template, portfolio, jump_model; kwargs...)
end

# No Base Power for Portfolio models. Always in Natural Units.
get_problem_base_power(model::InvestmentModel) = 1.0
get_portfolio(model::InvestmentModel) = get_system(model)
get_timestamps(model::InvestmentModel) =
    get_timestamps(get_time_mapping(get_optimization_container(model)))

# Portfolio-specific alias for Settings
get_portfolio_to_file(settings::Settings) = get_system_to_file(settings)
