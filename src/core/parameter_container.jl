#################################################################################
# Parameter Attributes - metadata types for parameter containers
#################################################################################

abstract type ParameterAttributes end

struct NoAttributes <: ParameterAttributes end

struct TimeSeriesAttributes{T <: IS.TimeSeriesData} <: ParameterAttributes
    name::String
    multiplier_id::Base.RefValue{Int}
    component_name_to_ts_uuid::Dict{String, String}
    subsystem::Base.RefValue{String}
end

function TimeSeriesAttributes(
    ::Type{T},
    name::String,
    multiplier_id::Int = 1,
    component_name_to_ts_uuid = Dict{String, String}(),
) where {T <: IS.TimeSeriesData}
    return TimeSeriesAttributes{T}(
        name,
        Base.RefValue{Int}(multiplier_id),
        component_name_to_ts_uuid,
        Base.RefValue{String}(""),
    )
end

get_time_series_type(::TimeSeriesAttributes{T}) where {T <: IS.TimeSeriesData} = T
get_time_series_name(attr::TimeSeriesAttributes) = attr.name
get_time_series_multiplier_id(attr::TimeSeriesAttributes) = attr.multiplier_id[]

get_subsystem(attr::TimeSeriesAttributes) = attr.subsystem[]
function set_subsystem!(attr::TimeSeriesAttributes, val::String)
    attr.subsystem[] = val
    return
end
set_subsystem!(::TimeSeriesAttributes, ::Nothing) = nothing

function add_component_name!(attr::TimeSeriesAttributes, name::String, uuid::String)
    if haskey(attr.component_name_to_ts_uuid, name)
        throw(ArgumentError("$name is already stored"))
    end

    attr.component_name_to_ts_uuid[name] = uuid
    return
end

get_component_names(attr::TimeSeriesAttributes) = keys(attr.component_name_to_ts_uuid)
function _get_ts_uuid(attr::TimeSeriesAttributes, name::String)
    if !haskey(attr.component_name_to_ts_uuid, name)
        throw(
            ArgumentError(
                "No time series UUID found for in attributes for component $name: available names are $(keys(attr.component_name_to_ts_uuid))",
            ),
        )
    end
    return attr.component_name_to_ts_uuid[name]
end

struct VariableValueAttributes{T <: OptimizationContainerKey} <: ParameterAttributes
    attribute_key::T
    affected_keys::Set
end

function VariableValueAttributes(key::T) where {T <: OptimizationContainerKey}
    return VariableValueAttributes{T}(key, Set())
end

get_attribute_key(attr::VariableValueAttributes) = attr.attribute_key

struct CostFunctionAttributes{T} <: ParameterAttributes
    variable_types::Tuple{Vararg{Type}}
    sos_status::SOSStatusVariable
    uses_compact_power::Bool
end

get_sos_status(attr::CostFunctionAttributes) = attr.sos_status
get_variable_types(attr::CostFunctionAttributes) = attr.variable_types
get_uses_compact_power(attr::CostFunctionAttributes) = attr.uses_compact_power

struct EventParametersAttributes{T <: IS.InfrastructureSystemsComponent, U <: ParameterType} <: ParameterAttributes
    affected_devices::Vector{<:IS.InfrastructureSystemsComponent}
end

function get_param_type(
    ::EventParametersAttributes{T, U},
) where {T <: IS.InfrastructureSystemsComponent, U <: ParameterType}
    return U
end

#################################################################################
# Parameter Container - holds parameter arrays and their attributes
#################################################################################

struct ParameterContainer{T <: AbstractArray, U <: AbstractArray, A <: ParameterAttributes}
    attributes::A
    parameter_array::T
    multiplier_array::U
end

function ParameterContainer(parameter_array, multiplier_array)
    return ParameterContainer(NoAttributes(), parameter_array, multiplier_array)
end

get_parameter_array(c::ParameterContainer) = c.parameter_array
get_multiplier_array(c::ParameterContainer) = c.multiplier_array
get_attributes(c::ParameterContainer) = c.attributes
Base.length(c::ParameterContainer) = length(c.parameter_array)
Base.size(c::ParameterContainer) = size(c.parameter_array)

function get_column_names(key::ParameterKey, c::ParameterContainer)
    return get_column_names_from_axis_array(key, get_multiplier_array(c))
end

#################################################################################
# Parameter value calculation methods
#################################################################################

function calculate_parameter_values(container::ParameterContainer)
    return calculate_parameter_values(
        container.attributes,
        container.parameter_array,
        container.multiplier_array,
    )
end

function calculate_parameter_values(
    attributes::ParameterAttributes,
    param_array::DenseAxisArray,
    multiplier_array::DenseAxisArray,
)
    return get_parameter_values(attributes, param_array, multiplier_array) .*
           multiplier_array
end

function calculate_parameter_values(
    ::ParameterAttributes,
    param_array::SparseAxisArray,
    multiplier_array::SparseAxisArray,
)
    p_array = jump_value.(to_matrix(param_array))
    m_array = to_matrix(multiplier_array)
    return p_array .* m_array
end

function get_parameter_column_refs(container::ParameterContainer, column::AbstractString)
    return get_parameter_column_refs(
        container.attributes,
        container.parameter_array,
        column,
    )
end

function get_parameter_column_refs(::ParameterAttributes, param_array, column)
    return param_array
end

function get_parameter_column_refs(
    attributes::TimeSeriesAttributes{T},
    param_array::DenseAxisArray,
    column,
) where {T <: IS.TimeSeriesData}
    expand_ixs((_get_ts_uuid(attributes, column),), param_array)
    return param_array[expand_ixs((_get_ts_uuid(attributes, column),), param_array)...]
end

function get_parameter_column_values(container::ParameterContainer, column::AbstractString)
    return jump_value.(get_parameter_column_refs(container, column))
end

function get_parameter_values(container::ParameterContainer)
    return get_parameter_values(
        container.attributes,
        container.parameter_array,
        container.multiplier_array,
    )
end

# TODO: SparseAxisArray versions of these functions

function get_parameter_values(
    ::ParameterAttributes,
    param_array::DenseAxisArray,
    multiplier_array::DenseAxisArray,
)
    return (.*).(jump_value.(param_array), multiplier_array)
end

function get_parameter_values(
    attr::EventParametersAttributes,
    param_array::DenseAxisArray,
    multiplier_array::DenseAxisArray,
)
    return jump_value.(param_array)
end

function get_parameter_values(
    attributes::TimeSeriesAttributes{T},
    param_array::DenseAxisArray,
    multiplier_array::DenseAxisArray,
) where {T <: IS.TimeSeriesData}
    exploded_param_array = DenseAxisArray{Float64}(undef, axes(multiplier_array)...)
    for name in axes(multiplier_array)[1]
        param_col = param_array[_get_ts_uuid(attributes, name), axes(param_array)[2:end]...]
        device_axes = axes(multiplier_array)[2:end]
        exploded_param_array[name, device_axes...] = jump_value.(param_col)
    end

    return exploded_param_array
end

#################################################################################
# Parameter setting methods
#################################################################################

const ValidDataParamEltypes = Union{Float64, Tuple{Vararg{Float64}}}

function _set_parameter!(
    array::AbstractArray{T},
    ::JuMP.Model,
    value::Union{T, AbstractVector{T}},
    ixs::Tuple,
) where {T <: ValidDataParamEltypes}
    assign_maybe_broadcast!(array, value, ixs)
    return
end

function _set_parameter!(
    array::AbstractArray{JuMP.VariableRef},
    model::JuMP.Model,
    value::Union{T, AbstractVector{T}},
    ixs::Tuple,
) where {T <: ValidDataParamEltypes}
    assign_maybe_broadcast!(array, add_jump_parameter.(Ref(model), value), ixs)
    return
end

function _set_parameter!(
    array::SparseAxisArray{Union{Nothing, JuMP.VariableRef}},
    model::JuMP.Model,
    value::Union{T, AbstractVector{T}},
    ixs::Tuple,
) where {T <: ValidDataParamEltypes}
    assign_maybe_broadcast!(array, add_jump_parameter.(Ref(model), value), ixs)
    return
end

function set_multiplier!(
    container::ParameterContainer,
    multiplier::Float64,
    ixs::Vararg{Any, N},
) where {N}
    assign_maybe_broadcast!(get_multiplier_array(container), multiplier, ixs)
    return
end

function set_parameter!(
    container::ParameterContainer,
    jump_model::JuMP.Model,
    parameter::Union{ValidDataParamEltypes, AbstractVector{<:ValidDataParamEltypes}},
    ixs::Vararg{Any, N},
) where {N}
    param_array = get_parameter_array(container)
    _set_parameter!(param_array, jump_model, parameter, ixs)
    return
end

# Overload for when a JuMP parameter VariableRef is passed directly (recurrent-solve
# path where a parallel branch type reuses the VariableRef created by the first type).
function set_parameter!(
    container::ParameterContainer,
    ::JuMP.Model,
    parameter::JuMP.VariableRef,
    ixs::Vararg{Any, N},
) where {N}
    assign_maybe_broadcast!(get_parameter_array(container), parameter, ixs)
    return
end
