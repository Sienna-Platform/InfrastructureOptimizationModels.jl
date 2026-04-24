"""
Container for the initial condition data
"""
mutable struct InitialCondition{
    T <: InitialConditionType,
    U <: Union{JuMP.VariableRef, Float64, Nothing},
}
    component::IS.InfrastructureSystemsComponent
    value::U
end

function InitialCondition(
    ::Type{T},
    component::IS.InfrastructureSystemsComponent,
    value::U,
) where {T <: InitialConditionType, U <: Union{JuMP.VariableRef, Float64}}
    return InitialCondition{T, U}(component, value)
end

function InitialCondition(
    ::InitialConditionKey{T, U},
    component::U,
    value::V,
) where {
    T <: InitialConditionType,
    U <: IS.InfrastructureSystemsComponent,
    V <: Union{JuMP.VariableRef, Float64},
}
    return InitialCondition{T, U}(component, value)
end

function get_condition(p::InitialCondition{T, Float64}) where {T <: InitialConditionType}
    return p.value
end

function get_condition(
    p::InitialCondition{T, JuMP.VariableRef},
) where {T <: InitialConditionType}
    return jump_value(p.value)
end

function get_condition(
    ::InitialCondition{T, Nothing},
) where {T <: InitialConditionType}
    return
end

get_component(ic::InitialCondition) = ic.component
get_value(ic::InitialCondition) = ic.value
get_component_name(ic::InitialCondition) = IS.get_name(ic.component)
get_component_type(ic::InitialCondition) = typeof(ic.component)
get_ic_type(
    ::Type{InitialCondition{T, U}},
) where {T <: InitialConditionType, U <: Union{JuMP.VariableRef, Float64, Nothing}} = T
get_ic_type(
    ::InitialCondition{T, U},
) where {T <: InitialConditionType, U <: Union{JuMP.VariableRef, Float64, Nothing}} = T

"""
Stores data to populate initial conditions before the build call
"""
mutable struct InitialConditionsData
    duals::Dict{ConstraintKey, JuMPArray}
    parameters::Dict{ParameterKey, JuMPArray}
    variables::Dict{VariableKey, JuMPArray}
    aux_variables::Dict{AuxVarKey, JuMPArray}
end

function InitialConditionsData()
    return InitialConditionsData(
        Dict{ConstraintKey, JuMPArray}(),
        Dict{ParameterKey, JuMPArray}(),
        Dict{VariableKey, JuMPArray}(),
        Dict{AuxVarKey, JuMPArray}(),
    )
end

@generated function get_initial_condition_value(
    ic_data::InitialConditionsData,
    ::Type{T},
    ::Type{U},
) where {
    T <: Union{VariableType, AuxVariableType, ConstraintType, ParameterType},
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    field = QuoteNode(store_field_for_type(T))
    K = key_for_type(T)
    return :(return getfield(ic_data, $field)[$K(T, U)])
end

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_initial_condition_value(
    ic_data::InitialConditionsData, ::T, ::Type{U},
) where {
    T <: Union{VariableType, AuxVariableType, ConstraintType, ParameterType},
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} =
    get_initial_condition_value(ic_data, T, U)

@generated function has_initial_condition_value(
    ic_data::InitialConditionsData,
    ::Type{T},
    ::Type{U},
) where {
    T <: Union{VariableType, AuxVariableType, ConstraintType, ParameterType},
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    field = QuoteNode(store_field_for_type(T))
    K = key_for_type(T)
    return :(return haskey(getfield(ic_data, $field), $K(T, U)))
end

# TODO: deprecate once POM is migrated to pass types (issue #18)
has_initial_condition_value(
    ic_data::InitialConditionsData, ::T, ::Type{U},
) where {
    T <: Union{VariableType, AuxVariableType, ConstraintType, ParameterType},
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} =
    has_initial_condition_value(ic_data, T, U)
