"""
Time-series parameter whose value multiplies a decision variable. Its containers always hold
`Float64`, and its values reach the model only through
[`add_parameterized_product_constraint!`](@ref).
"""
abstract type TimeSeriesLHSParameter <: LeftHandSideParameter end

"""
Time series parameter types for optimization models.
These are simple type markers that indicate which time series data to use.
"""

# Standard power system parameters
struct ActivePowerTimeSeriesParameter <: TimeSeriesParameter end
struct ReactivePowerTimeSeriesParameter <: TimeSeriesParameter end
struct ActivePowerInTimeSeriesParameter <: TimeSeriesParameter end
struct ActivePowerOutTimeSeriesParameter <: TimeSeriesParameter end
struct RequirementTimeSeriesParameter <: TimeSeriesParameter end
