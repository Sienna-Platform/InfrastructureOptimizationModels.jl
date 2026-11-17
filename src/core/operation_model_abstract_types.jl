"""
Abstract type for any optimization problem. Concrete subtypes are provided by
downstream domain libraries (e.g. PowerOperationsModels). `DecisionModel{M}` and
`EmulationModel{M}` are parameterized over this type so any optimization problem
can be wrapped without IOM knowing its domain.

# Example

import InfrastructureOptimizationModels
const IOM = InfrastructureOptimizationModels
struct MyCustomProblem <: IOM.AbstractOptimizationProblem end
"""
abstract type AbstractOptimizationProblem end

"""
Abstract supertype for `DecisionModel` and `EmulationModel`. Concrete subtypes
are parameterized with an `AbstractOptimizationProblem` subtype.
"""
abstract type AbstractOptimizationModel end

"""
Abstract type for the data that one solve gives to the post-solve calculations.

The domain package makes one context for each solve with [`make_solve_context`](@ref).
`execute_optimizer!` gives the context to `calculate_aux_variable_value!` for each
standalone aux variable. No container, template, or model keeps the context.
"""
abstract type AbstractSolveContext end
