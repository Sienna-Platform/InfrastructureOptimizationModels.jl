"""
Abstract type for evaluator configurations attached to a `NetworkModel`.

An `AbstractEvaluator` is stateless configuration that, when used to build an
`OptimizationContainer`, produces an `AbstractEvaluationData` (its runtime
counterpart) via [`initialize_evaluation_data`](@ref).

Concrete subtypes (e.g. `PowerFlowEvaluationModel`) are defined in downstream
packages such as PowerOperationsModels and its PowerFlows extension.
"""
abstract type AbstractEvaluator end

"""
Abstract type for runtime evaluator state stored inside the
`OptimizationContainer`'s [`EvaluationContainer`](@ref).

Concrete subtypes wrap whatever domain-specific data (power-flow solver state,
exporter handles, etc.) the evaluator needs across iterations of
`calculate_aux_variables!`.
"""
abstract type AbstractEvaluationData end

"""
Holds the evaluators registered on a `NetworkModel` and their associated
runtime data, keyed by concrete evaluator type.

`evaluators` is populated by the user when constructing a `NetworkModel`;
`evaluation_data` is populated during `OptimizationContainer` construction by
calling [`initialize_evaluation_data`](@ref) for each registered evaluator.

The split between the two dictionaries may evolve; for now we keep them as
parallel keyed dictionaries so that `evaluation_data[T]` is the runtime state
produced from `evaluators[T]`.
"""
mutable struct EvaluationContainer
    evaluators::Dict{DataType, Any}
    evaluation_data::Dict{DataType, AbstractEvaluationData}
end

function EvaluationContainer()
    return EvaluationContainer(
        Dict{DataType, Any}(),
        Dict{DataType, AbstractEvaluationData}(),
    )
end

get_evaluators(ec::EvaluationContainer) = ec.evaluators
get_evaluation_data(ec::EvaluationContainer) = ec.evaluation_data

get_evaluator(ec::EvaluationContainer, T::DataType) = ec.evaluators[T]
get_evaluation_data(ec::EvaluationContainer, T::DataType) = ec.evaluation_data[T]

add_evaluator!(ec::EvaluationContainer, T::DataType, ev) =
    (ec.evaluators[T] = ev)
add_evaluation_data!(
    ec::EvaluationContainer,
    T::DataType,
    d::AbstractEvaluationData,
) = (ec.evaluation_data[T] = d)

Base.isempty(ec::EvaluationContainer) = isempty(ec.evaluators)
Base.length(ec::EvaluationContainer) = length(ec.evaluators)
Base.haskey(ec::EvaluationContainer, T::DataType) = haskey(ec.evaluators, T)

#=
=========================================================================
AbstractEvaluator interface (config-side)
=========================================================================
=#

"""
Build the runtime state for `ev` and return an `AbstractEvaluationData`.
Called once during `OptimizationContainer` construction. Concrete methods are
registered in downstream packages.
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
Return the underlying domain-specific data wrapped by `data` (e.g., a
`PowerFlowData` for a power-flow evaluator). Used by aux-variable calculators
that need direct read access to the evaluator's output.
"""
function get_inner_data(data::AbstractEvaluationData)
    error(
        "get_inner_data not implemented for evaluation data type $(typeof(data)). " *
        "Define `get_inner_data(::$(typeof(data)))` " *
        "in the package that owns the concrete type.",
    )
end
