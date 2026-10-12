"""
Abstract type for the configuration of an evaluator.

An evaluator is a calculation that runs after each solve, outside the optimization model.
The configuration is part of the problem template. It holds no runtime data.

At build time, [`initialize_evaluation_data`](@ref) makes the runtime data from the
configuration. Downstream packages define the concrete subtypes.
"""
abstract type AbstractEvaluator end

"""
Abstract type for the runtime data of an evaluator.

[`initialize_evaluation_data`](@ref) makes the runtime data at build time. The
`OptimizationContainer` keeps it in an [`EvaluationDataContainer`](@ref) for one build.
`calculate_aux_variables!` uses it after each solve.
"""
abstract type AbstractEvaluationData end

"""
Holds the evaluator configurations of a problem template, keyed by the concrete evaluator
type. It holds configuration only. The runtime data is in an
[`EvaluationDataContainer`](@ref).
"""
mutable struct EvaluationContainer
    evaluators::Dict{DataType, AbstractEvaluator}
end

EvaluationContainer() = EvaluationContainer(Dict{DataType, AbstractEvaluator}())

get_evaluators(ec::EvaluationContainer) = ec.evaluators
get_evaluator(ec::EvaluationContainer, T::DataType) = ec.evaluators[T]

function add_evaluator!(ec::EvaluationContainer, T::DataType, ev::AbstractEvaluator)
    ec.evaluators[T] = ev
    return
end

Base.isempty(ec::EvaluationContainer) = isempty(ec.evaluators)
Base.length(ec::EvaluationContainer) = length(ec.evaluators)
Base.haskey(ec::EvaluationContainer, T::DataType) = haskey(ec.evaluators, T)

"""
Holds the runtime data of the evaluators of one build, keyed by the concrete evaluator
type. The `OptimizationContainer` owns it, and a rebuild empties it.
"""
struct EvaluationDataContainer
    evaluation_data::Dict{DataType, AbstractEvaluationData}
end

EvaluationDataContainer() =
    EvaluationDataContainer(Dict{DataType, AbstractEvaluationData}())

function add_evaluation_data!(
    edc::EvaluationDataContainer,
    T::DataType,
    data::AbstractEvaluationData,
)
    edc.evaluation_data[T] = data
    return
end

Base.getindex(edc::EvaluationDataContainer, T::DataType) = edc.evaluation_data[T]
Base.haskey(edc::EvaluationDataContainer, T::DataType) = haskey(edc.evaluation_data, T)
Base.isempty(edc::EvaluationDataContainer) = isempty(edc.evaluation_data)
Base.length(edc::EvaluationDataContainer) = length(edc.evaluation_data)
Base.keys(edc::EvaluationDataContainer) = keys(edc.evaluation_data)
Base.values(edc::EvaluationDataContainer) = values(edc.evaluation_data)
Base.iterate(edc::EvaluationDataContainer, state...) =
    iterate(edc.evaluation_data, state...)
function Base.empty!(edc::EvaluationDataContainer)
    empty!(edc.evaluation_data)
    return edc
end

#=
=========================================================================
AbstractEvaluator interface (config-side)
=========================================================================
=#

"""
Make the runtime data for the evaluator configuration `ev` and return an
`AbstractEvaluationData`.

The domain package calls this function once for each evaluator at build time. It stores
the result in the `OptimizationContainer` with [`add_evaluation_data!`](@ref).
Downstream packages define the concrete methods.
"""
function initialize_evaluation_data(
    ev::AbstractEvaluator,
    container,
    system,
)
    error(
        "initialize_evaluation_data not implemented for evaluator type $(typeof(ev)). " *
        "Define `initialize_evaluation_data(::$(typeof(ev)), container, system) ::AbstractEvaluationData` " *
        "in the package that owns the concrete type.",
    )
end

#=
=========================================================================
AbstractEvaluationData interface (runtime-side)
=========================================================================
=#

"""
Run the evaluation: read state from `container`/`system`, write aux-variable
inputs back into `container`, and mark `data` as solved. Concrete methods are
registered in downstream packages.
"""
function evaluate!(data::AbstractEvaluationData, container, system)
    error(
        "evaluate! not implemented for evaluation data type $(typeof(data)). " *
        "Define `evaluate!(::$(typeof(data)), container, system)` " *
        "in the package that owns the concrete type.",
    )
end

"""
Mark `data` as not-yet-solved for the current iteration. Called by
`calculate_aux_variables!` at the top of each pass.
"""
function reset!(data::AbstractEvaluationData)
    error(
        "reset! not implemented for evaluation data type $(typeof(data)). " *
        "Define `reset!(::$(typeof(data)))` " *
        "in the package that owns the concrete type.",
    )
end

"""
Return `true` if `data` has been solved in the current iteration.
"""
function is_solved(data::AbstractEvaluationData)
    error(
        "is_solved not implemented for evaluation data type $(typeof(data)). " *
        "Define `is_solved(::$(typeof(data))) ::Bool` " *
        "in the package that owns the concrete type.",
    )
end

"""
Return the domain data that `data` wraps. Aux variable calculations use it to read the
output of the evaluator.
"""
function get_inner_data(data::AbstractEvaluationData)
    error(
        "get_inner_data not implemented for evaluation data type $(typeof(data)). " *
        "Define `get_inner_data(::$(typeof(data)))` " *
        "in the package that owns the concrete type.",
    )
end
