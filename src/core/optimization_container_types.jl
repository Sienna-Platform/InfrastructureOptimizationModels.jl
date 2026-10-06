# Key types are imported from InfrastructureSystems.Optimization in the main module:
# - AbstractOptimizationContainer, OptimizationKeyType
# - VariableType, ConstraintType, AuxVariableType, ParameterType, InitialConditionType, ExpressionType
# - ObjectiveFunctionParameter, TimeSeriesParameter
# - ConstructStage, ArgumentConstructStage, ModelConstructStage

# Utility functions for the imported types
convert_output_to_natural_units(::Type{<:OptimizationKeyType}) = false

should_write_resulting_value(::Type{<:VariableType}) = true
should_write_resulting_value(::Type{<:ConstraintType}) = true
should_write_resulting_value(::Type{<:AuxVariableType}) = true
should_write_resulting_value(::Type{<:ExpressionType}) = false
# TODO: Piecewise linear parameter are broken to write
should_write_resulting_value(::Type{<:ParameterType}) = false
