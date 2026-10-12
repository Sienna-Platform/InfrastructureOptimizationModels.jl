"""
Abstract type for an evaluator configuration.

The user adds an `AbstractEvaluator` to an [`EvaluationContainer`](@ref) and gives the
container to a `NetworkModel`. The configuration lives on the network model for the life of
the template. It holds no runtime data.

At build time, [`initialize_evaluation_data`](@ref) makes the runtime data from the
configuration. Downstream packages, such as PowerOperationsModels, define the concrete
subtypes.
"""
abstract type AbstractEvaluator end

"""
Abstract type for the runtime data of an evaluator.

[`initialize_evaluation_data`](@ref) makes the runtime data at build time. The
`OptimizationContainer` keeps it for one build, keyed by the concrete evaluator type. Use
[`get_evaluation_data`](@ref) and [`add_evaluation_data!`](@ref) on the container to read
and write it. `calculate_aux_variables!` uses it at each solve.
"""
abstract type AbstractEvaluationData end

"""
Holds the evaluator configurations of a `NetworkModel`, keyed by the concrete evaluator
type.

The user fills the container and gives it to `NetworkModel(...; evaluations = ec)`. The
container holds configuration only. The `OptimizationContainer` keeps the runtime data.
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

#=
=========================================================================
AbstractEvaluator interface (config-side)
=========================================================================
=#

"""
Make the runtime data for the evaluator configuration `ev` and return an
`AbstractEvaluationData`.

The domain package calls this function once for each evaluator at build time. It reads
`ev` from the network model and writes the result to the `OptimizationContainer` with
[`add_evaluation_data!`](@ref). Downstream packages define the concrete methods.
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
