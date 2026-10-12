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
