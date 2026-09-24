"""
Minimal service mocks.
"""

struct MockUp end
struct MockDown end

struct MockReserve{D} <: IS.InfrastructureSystemsComponent
    name::String
    requirement::Float64
    contributing_devices::Vector{Any}
end

get_name(r::MockReserve) = r.name
get_requirement(r::MockReserve) = r.requirement

struct MockReserveFormulation <: IOM.AbstractServiceFormulation end

IOM.get_default_attributes(::Type{<:MockReserve}, ::Type{MockReserveFormulation}) =
    Dict{String, Any}()
IOM.get_default_time_series_names(::Type{<:MockReserve}, ::Type{MockReserveFormulation}) =
    Dict{Type{<:IOM.ParameterType}, String}()
IOM.get_variable_binary(
    ::Type{<:IOM.VariableType},
    ::Type{<:MockReserve},
    ::Type{MockReserveFormulation},
) = false
IOM.get_variable_upper_bound(
    ::Type{<:IOM.VariableType},
    r::MockReserve,
    ::IS.InfrastructureSystemsComponent,
    ::Type{MockReserveFormulation},
) = r.requirement
IOM.get_variable_lower_bound(
    ::Type{<:IOM.VariableType},
    ::MockReserve,
    ::IS.InfrastructureSystemsComponent,
    ::Type{MockReserveFormulation},
) = 0.0
