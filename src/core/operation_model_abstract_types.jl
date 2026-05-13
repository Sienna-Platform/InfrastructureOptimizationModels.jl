"""
Common supertype for problems that an `OperationModel` can solve.
Both `DecisionProblem` and `EmulationProblem` are subtypes.
"""
abstract type OperationProblem end

"""
Abstract type for Decision Model and Emulation Model. `OperationModel`
subtypes are parameterized by the problem they solve (`<: OperationProblem`).
"""
abstract type OperationModel{T <: OperationProblem} end

#TODO: Document the required interfaces for custom types
"""
Abstract type for Decision Problems

# Example

import InfrastructureOptimizationModels
const POM = InfrastructureOptimizationModels
struct MyCustomProblem <: POM.DecisionProblem
"""
abstract type DecisionProblem <: OperationProblem end

"""
Abstract type for Emulation Problems

# Example

import InfrastructureOptimizationModels
const POM = InfrastructureOptimizationModels
struct MyCustomEmulator <: POM.EmulationProblem
"""
abstract type EmulationProblem <: OperationProblem end

"""
Return the concrete `OperationProblem` subtype that parameterizes a model.
"""
get_problem_type(::OperationModel{M}) where {M <: OperationProblem} = M

#################################################################################
# Simulation Models Container
# Holds references to models in a simulation
# Used for display/printing purposes
struct SimulationModels
    decision_models::Vector{<:OperationModel}
    emulation_model::Union{Nothing, OperationModel}
end

function SimulationModels(;
    decision_models,
    emulation_model::Union{Nothing, OperationModel} = nothing,
)
    return SimulationModels(decision_models, emulation_model)
end

#################################################################################
# Simulation Sequence
# Holds the execution sequence information for a simulation
# This is a placeholder struct - concrete implementation in PowerSimulations
struct SimulationSequence
    executions_by_model::Dict
    horizons::Dict
    intervals::Dict
    SimulationSequence() = new(Dict(), Dict(), Dict())
end

# Placeholder accessor function for simulation sequence
get_step_resolution(::SimulationSequence) = Dates.Hour(1)

#################################################################################
# Simulation Type
# Abstract type for simulation objects
# Concrete implementation should be in PowerSimulations
abstract type Simulation end

#################################################################################
# Simulation Outputs Type
# Abstract type for simulation outputs
# Concrete implementation should be in PowerSimulations
abstract type SimulationOutputs end

#################################################################################
# Simulation Problem Outputs Type
# Abstract type for individual problem outputs within a simulation
# Concrete implementation should be in PowerSimulations
abstract type SimulationProblemOutputs end
