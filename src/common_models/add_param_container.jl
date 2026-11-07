"""
Thin wrappers around `add_param_container_split_axes!` and `add_param_container_shared_axes!`.
Each one constructs one `ParameterAttributes` subtype and is named after it; the `T <: ...`
constraints are primarily sanity checks (each builder has a single method) rather than selecting
among multiple overloads of a single function.
"""

_time_series_param_eltype(container::OptimizationContainer, ::Type{<:TimeSeriesParameter}) =
    get_param_eltype(container)
_time_series_param_eltype(
    ::OptimizationContainer,
    ::Type{<:LeftHandSideTimeSeriesParameter},
) =
    Float64

"""
Allocate a time-series parameter container (`TimeSeriesAttributes`). Parameter and multiplier
arrays may have different first axes, so this is the only builder using the split-axes allocator.

A [`LeftHandSideTimeSeriesParameter`](@ref) container always holds `Float64`: its values are constraint
coefficients, never JuMP parameters.
"""
function add_time_series_parameter_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{V},
    name::String,
    param_axs,
    multiplier_axs,
    additional_axs,
    time_steps::UnitRange{Int};
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: TimeSeriesParameter,
    U <: IS.InfrastructureSystemsComponent,
    V <: IS.TimeSeriesData,
}
    param_key = ParameterKey(T, U, meta)
    if isabstracttype(V)
        error("$V can't be abstract: $param_key")
    end
    attributes = TimeSeriesAttributes(V, name)
    return add_param_container_split_axes!(
        container,
        param_key,
        attributes,
        _time_series_param_eltype(container, T),
        param_axs,
        multiplier_axs,
        additional_axs,
        time_steps;
        sparse = sparse,
    )
end

"""
Allocate a time-series parameter container whose components are summed per row key.
`row_of` maps each component name to its row key, for example the bus of a load. The member
array holds `Float64`; one parameter per row and time step is the sum (see
[`RowAggregation`](@ref)). Call [`build_row_aggregates!`](@ref) after the members are set.
"""
function add_aggregated_time_series_parameter_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{V},
    name::String,
    param_axs,
    multiplier_axs,
    row_of::Dict{String, K},
    time_steps::UnitRange{Int};
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: TimeSeriesParameter,
    U <: IS.InfrastructureSystemsComponent,
    V <: IS.TimeSeriesData,
    K,
}
    param_key = ParameterKey(T, U, meta)
    if isabstracttype(V)
        error("$V can't be abstract: $param_key")
    end
    row_keys = unique!(sort!(collect(values(row_of))))
    aggregation = RowAggregation(
        row_of,
        [Tuple{Int, Float64}[] for _ in row_keys],
        DenseAxisArray{get_param_eltype(container)}(undef, row_keys, time_steps),
    )
    attributes = TimeSeriesAttributes(V, name, 1, Dict{String, String}(), aggregation)
    return add_param_container_split_axes!(
        container,
        param_key,
        attributes,
        Float64,
        param_axs,
        multiplier_axs,
        (),
        time_steps,
    )
end

"""
Fill the members of the row aggregation and create one parameter per row and time step.
Call it once, after the member values, the multipliers, and the component names are set.
No-op for a container without row aggregation.
"""
function build_row_aggregates!(container::OptimizationContainer, param_container)
    _build_row_aggregates!(container, param_container, get_attributes(param_container))
    return
end

_build_row_aggregates!(::OptimizationContainer, _, ::TimeSeriesAttributes) = nothing

function _build_row_aggregates!(
    container::OptimizationContainer,
    param_container,
    attributes::TimeSeriesAttributes{<:IS.TimeSeriesData, <:RowAggregation},
)
    aggregation = get_row_aggregation(attributes)
    param_index = get_parameter_array(param_container).lookup[1]
    multiplier_array = get_multiplier_array(param_container)
    parent_mult = multiplier_array.data
    mult_index = multiplier_array.lookup[1]
    row_index = aggregation.values.lookup[1]
    for (name, ts_uuid) in attributes.component_name_to_ts_uuid
        if !haskey(aggregation.row_of, name)
            error("Component $name of $(get_time_series_name(attributes)) has no row key.")
        end
        i = mult_index[name]
        multiplier = parent_mult[i, 1]
        if !all(==(multiplier), @view parent_mult[i, :])
            error(
                "Component $name of $(get_time_series_name(attributes)) has a " *
                "time-varying multiplier. Row aggregation needs a constant multiplier.",
            )
        end
        push!(
            aggregation.members[row_index[aggregation.row_of[name]]],
            (param_index[ts_uuid], multiplier),
        )
    end
    jump_model = get_jump_model(container)
    parent_values = aggregation.values.data
    for t in axes(parent_values, 2), r in axes(parent_values, 1)
        _set_parameter_at!(parent_values, jump_model, 0.0, r, t)
    end
    refresh_row_aggregates!(attributes, get_parameter_array_data(param_container))
    return
end

"""
Allocate a cost-function parameter container (`CostFunctionAttributes`).

Note that `data_type` sets both the attributes' type parameter and the parameter array's element
type, bypassing `get_param_eltype(container)`.
"""
function add_cost_function_parameter_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    variable_types::Tuple{Vararg{Type}},
    axs...;
    sos_variable::SOSStatusVariable.Value = SOSStatusVariable.NO_VARIABLE,
    uses_compact_power::Bool = false,
    data_type::DataType = Float64,
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: ObjectiveFunctionParameter, U <: IS.InfrastructureSystemsComponent}
    param_key = ParameterKey(T, U, meta)
    attributes =
        CostFunctionAttributes{data_type}(variable_types, sos_variable, uses_compact_power)
    return add_param_container_shared_axes!(
        container,
        param_key,
        attributes,
        data_type,
        axs...;
        sparse = sparse,
    )
end

"""
Allocate a parameter container fed by another container's values (`VariableValueAttributes`).
`source_key` is the variable/aux-variable key supplying those values.
"""
function add_variable_value_parameter_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    source_key::V,
    axs...;
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: VariableValueParameter,
    U <: IS.InfrastructureSystemsComponent,
    V <: OptimizationContainerKey,
}
    param_key = ParameterKey(T, U, meta)
    attributes = VariableValueAttributes(source_key)
    return add_param_container_shared_axes!(
        container, param_key, attributes, get_param_eltype(container), axs...;
        sparse = sparse)
end

"""
Allocate an event parameter container (`EventParametersAttributes`). `U` is the component type
the event affects; `V` is the supplemental-attribute type describing the event itself.
"""
function add_event_parameter_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{V},
    axs...;
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: EventParameter,
    U <: IS.InfrastructureSystemsComponent,
    V <: IS.SupplementalAttribute,
}
    param_key = ParameterKey(T, U, meta)
    attributes = EventParametersAttributes(V, T)
    return add_param_container_shared_axes!(
        container, param_key, attributes, get_param_eltype(container), axs...;
        sparse = sparse)
end
