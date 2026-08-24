# Compatibility shims for the old `add_param_container!` name (issue #147).
#
# These forwarders exist only so downstream (POM, PowerSystemsInvestments) keeps working while
# it migrates. Delete this file once all downstream code has been updated to use the new names.

add_param_container!(
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
} = add_time_series_parameter_container!(
    container,
    T,
    U,
    V,
    name,
    param_axs,
    multiplier_axs,
    additional_axs,
    time_steps;
    sparse = sparse,
    meta = meta,
)

add_param_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    variable_types::Tuple{Vararg{Type}},
    sos_variable::SOSStatusVariable = SOSStatusVariable.NO_VARIABLE,
    uses_compact_power::Bool = false,
    data_type::DataType = Float64,
    axs...;
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: ObjectiveFunctionParameter, U <: IS.InfrastructureSystemsComponent} =
    add_cost_function_parameter_container!(
        container,
        T,
        U,
        variable_types,
        axs...;
        sos_variable = sos_variable,
        uses_compact_power = uses_compact_power,
        data_type = data_type,
        sparse = sparse,
        meta = meta,
    )

# Also covers the old `T <: FixValueParameter` method, which was byte-for-byte identical to the
# `VariableValueParameter` one (`FixValueParameter <: VariableValueParameter`).
add_param_container!(
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
} = add_variable_value_parameter_container!(
    container, T, U, source_key, axs...; sparse = sparse, meta = meta)

add_param_container!(
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
    V <: IS.InfrastructureSystemsComponent,
} = add_event_parameter_container!(
    container, T, U, V, axs...; sparse = sparse, meta = meta)
