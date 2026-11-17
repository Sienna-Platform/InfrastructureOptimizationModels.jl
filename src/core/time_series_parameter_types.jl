"""
Time-series parameter whose value multiplies or selects a decision variable. Its containers
always hold `Float64`, and formulations read its values with [`get_lhs_parameter_values`](@ref)
and write them into constraints as fixed coefficients or use them to choose the variables of a
row. A model holding one is rebuilt every simulation step, so the rows follow the refreshed
values.
"""
abstract type LeftHandSideTimeSeriesParameter <: TimeSeriesParameter end

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
