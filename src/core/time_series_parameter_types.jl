"""
Time-series parameter whose value multiplies or selects a decision variable. Its containers
always hold `Float64`, and formulations read its values with [`get_lhs_parameter_values`](@ref)
and write them into constraints as fixed coefficients or use them to choose the variables of a
row. A model holding one is rebuilt every simulation step, so the rows follow the refreshed
values.
"""
abstract type LeftHandSideTimeSeriesParameter <: TimeSeriesParameter end

"""
Parameter driven by an event (e.g. an outage) rather than a time series. Its values are written
by a simulation runtime. Formulations may read one as a left-hand-side coefficient with
[`get_lhs_parameter_values`](@ref), which requires a model rebuilt every step.
"""
abstract type EventParameter <: ParameterType end

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
