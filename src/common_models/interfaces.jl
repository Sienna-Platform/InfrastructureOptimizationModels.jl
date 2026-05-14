"""
Extension point: Add parameters to the optimization container.
Concrete implementations are in PowerOperationsModels.
"""
function add_parameters!(
    ::OptimizationContainer,
    ::Type{T},
    devices::U,
    model::DeviceModel{D, W},
) where {
    T <: ParameterType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    W <: AbstractDeviceFormulation,
} where {D <: IS.InfrastructureSystemsComponent}
    error(
        "add_parameters! not implemented for parameter type $T, device type $D with formulation $W. Implement this method in PowerOperationsModels.",
    )
end

###############################
###### get_foo functions ######
###############################

# stuff associated to a formulation: attributes, time series names
"""
Extension point: Get default attributes for a device formulation.
"""
function get_default_attributes(
    ::Type{U},
    ::Type{F},
) where {U <: IS.InfrastructureSystemsComponent, F <: AbstractDeviceFormulation}
    return Dict{String, Any}()
end

"""
Extension point: Get default time series names for a device formulation.
"""
function get_default_time_series_names(
    ::Type{U},
    ::Type{F},
) where {U <: IS.InfrastructureSystemsComponent, F <: AbstractDeviceFormulation}
    return Dict{Type{<:ParameterType}, String}()
end

# variable properties, for device or service formulation
"""
Extension point: Is the variable binary/integer?
"""
function get_variable_binary(
    ::Type{T},
    ::Type{U},
    ::Type{F},
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    error("`get_variable_binary` not implemented for $T and $U (with formulation $F).")
end

"""
Extension point: Get variable lower bound.
"""
get_variable_lower_bound(
    ::Type{<:VariableType},
    ::IS.InfrastructureSystemsComponent,
    ::Type{<:Union{AbstractDeviceFormulation, AbstractServiceFormulation}},
) = nothing

"""
Extension point: Get variable upper bound.
"""
get_variable_upper_bound(
    ::Type{<:VariableType},
    ::IS.InfrastructureSystemsComponent,
    ::Type{<:Union{AbstractDeviceFormulation, AbstractServiceFormulation}},
) = nothing

"""
Extension point: Get variable warm start value.
"""
get_variable_warm_start_value(
    ::Type{<:VariableType},
    ::IS.InfrastructureSystemsComponent,
    ::Type{<:Union{AbstractDeviceFormulation, AbstractServiceFormulation}},
) = nothing

###############################
###### Proportional Cost ######
###############################

"""
Extension point: Get proportional cost term from operation cost data.
Non-time-varying signature - returns a single cost value for all time steps.
"""
function proportional_cost(
    ::O,
    ::Type{V},
    ::C,
    ::Type{F},
) where {
    O <: IS.DeviceParameter,
    V <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "proportional cost not implemented for non-time-varying case for cost type $O, variable type $V, component type $C, formulation $F.",
    )
end

"""
Extension point: Get proportional cost term from operation cost data.
Time-varying signature - may return different values per time step.
"""
function proportional_cost(
    ::OptimizationContainer,
    ::O,
    ::Type{V},
    ::C,
    ::Type{F},
    ::Int,
) where {
    O <: IS.DeviceParameter,
    V <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "proportional cost not implemented for time-varying case for cost type $O, variable type $V, component type $C, formulation $F.",
    )
end

"""
Extension point: Check if proportional cost term is time-variant.
Returns true if the cost should be added to the variant objective expression.
"""
is_time_variant_proportional(::IS.DeviceParameter) = false

# corresponds to get_must_run for thermals, but avoiding device specific code here.
"""
Extension point: whether to skip adding proportional cost for a given device.

For thermals, equivalent to `get_must_run`, but that implementation belongs in POM.
"""
skip_proportional_cost(d::IS.InfrastructureSystemsComponent) = false

###############################
#### System query stubs #######
###############################
# Extension points for querying a system object. POM provides methods for
# PSY.System; tests provide methods for MockSystem. IOM itself never accesses
# sys.data.

"Extension point: time-series resolutions available on the system."
function get_time_series_resolutions end

"Extension point: counts summary of time series on the system."
function get_time_series_counts end

"Extension point: counts by component type of time series on the system."
function get_time_series_counts_by_type end

"Extension point: forecast interval configured on the system."
function get_forecast_interval end

"Extension point: forecast horizon configured on the system."
function get_forecast_horizon end

"Extension point: summary table of forecasts on the system."
function get_forecast_summary_table end

"Extension point: transform single time series into deterministic forecasts on the system."
function transform_single_time_series! end

"Extension point: stable UUID for the system (used as a filename identifier)."
function get_system_uuid end

"Extension point: get components of type `T` in a subsystem of the system."
function get_subsystem_components end

###############################
###### Start-up Cost ##########
###############################

"""
Extension point: Convert raw startup cost to a scalar value.
Device-specific implementations (e.g., for StartUpStages, MultiStartVariable) are in POM.
"""
function start_up_cost(
    cost::Any, # could be NamedTuple, StartUpStages, AffExpr, or Float.
    ::Type{T},
    ::Type{V},
    ::Type{F},
) where {
    T <: IS.InfrastructureSystemsComponent,
    V <: VariableType,
    F <: AbstractDeviceFormulation,
}
    error(
        "start_up_cost not implemented for cost type $(typeof(cost)), device type $T, " *
        "variable type $V, formulation $F.",
    )
end

"""
Extension point: Solve the power flow model and update aux-variable inputs.
Signature: `solve_power_flow!(pf_e_data, container, system)`. Concrete
implementations require PowerFlows integration; provided in PowerOperationsModels.
"""
function solve_power_flow! end

"""
Extension point: Get the underlying `PowerFlowData` from a `PowerFlowEvaluationData` wrapper.
Concrete implementation lives in the PowerFlows extension of POM.
"""
function get_power_flow_data end

"""
Extension point: Calculate auxiliary variable values.
Concrete implementations in PowerOperationsModels for specific aux variable types.
"""
function calculate_aux_variable_value! end

"""
Extension point: Check if an auxiliary variable type comes from power flow evaluation.
Default: false. Override in POM for PowerFlowAuxVariableType subtypes.
"""
is_from_power_flow(::Type{<:AuxVariableType}) = false

"""
Extension point: Get minimum and maximum limits for a given component, constraint type, and device formulation.
"""
get_min_max_limits(
    ::IS.InfrastructureSystemsComponent,
    ::Type{<:ConstraintType},
    ::Type{<:AbstractDeviceFormulation},
) = nothing

"""
Extension point: variable cost.

The one exception where it isn't just `get_variable(cost)`: storage devices, where we
need to map `ActivePower{In/Out}` to {charge/discharge} variable cost.
"""
function variable_cost(
    cost::IS.DeviceParameter,
    ::Type{<:VariableType},
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:AbstractDeviceFormulation},
)
    return get_variable_cost(cost)
end

"""
Extension point: read the primary variable cost from an operation-cost object.
POM provides methods (typically delegating to `PSY.get_variable`).
"""
function get_variable_cost end

variable_cost(
    ::Nothing,
    ::Type{<:VariableType},
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:AbstractDeviceFormulation},
) = 0.0

"""
Extension point: get the initial condition type for a given constraint, device, and formulation.
Concrete implementations in POM. Used for ramp constraints.
"""
_get_initial_condition_type(
    X::Type{<:ConstraintType},
    Y::Type{<:IS.InfrastructureSystemsComponent},
    Z::Type{<:AbstractDeviceFormulation},
) = error("`_get_initial_condition_type` not implemented for $X , $Y and $Z")

"""
    update_container_parameter_values!(optimization_container, model, key, input)

Update parameter values in the optimization container from the given input data.
This is an extension point — concrete implementations should be defined in
PowerOperationsModels (or PowerSimulations for simulation-specific variants).

Only called in `emulation_model.jl`: that file's contents and this function should
likely be moved to POM or PSI.
"""
function update_container_parameter_values! end

"""
Component type associated with network duals. Returns ACBus when passed in `nothing`.
Only called in a single spot in `add_constraint_dual!` for the network model.
"""
function component_for_network_dual end

"""
Component type associated with hvdc interpolation constraints. Returns DCBus when passed in
`nothing`. Only called in a single spot in `incremental.jl`.
"""
function component_for_hvdc_interpolation end
