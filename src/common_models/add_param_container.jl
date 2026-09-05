"""
Thin wrappers around `add_param_container_split_axes!` and `add_param_container_shared_axes!`.
Each one constructs one `ParameterAttributes` subtype and is named after it; the `T <: ...`
constraints are primarily sanity checks (each builder has a single method) rather than selecting
among multiple overloads of a single `add_param_container!` function.
Legacy `add_param_container!` shims live in `add_param_container_shims.jl`.
"""

"""
Allocate a time-series parameter container (`TimeSeriesAttributes`). Parameter and multiplier
arrays may have different first axes, so this is the only builder using the split-axes allocator.
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
        get_param_eltype(container),
        param_axs,
        multiplier_axs,
        additional_axs,
        time_steps;
        sparse = sparse,
    )
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
    sos_variable::SOSStatusVariable = SOSStatusVariable.NO_VARIABLE,
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
