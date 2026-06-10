# Store const definitions
# Update src/simulation/simulation_store_common.jl with any changes.
const STORE_CONTAINER_DUALS = :duals
const STORE_CONTAINER_PARAMETERS = :parameters
const STORE_CONTAINER_VARIABLES = :variables
const STORE_CONTAINER_AUX_VARIABLES = :aux_variables
const STORE_CONTAINER_EXPRESSIONS = :expressions
const STORE_CONTAINERS = (
    STORE_CONTAINER_DUALS,
    STORE_CONTAINER_PARAMETERS,
    STORE_CONTAINER_VARIABLES,
    STORE_CONTAINER_AUX_VARIABLES,
    STORE_CONTAINER_EXPRESSIONS,
)
const STORE_CONTAINER_TYPES = (
    ConstraintType,
    ParameterType,
    VariableType,
    AuxVariableType,
    ExpressionType,
)

# Derives from store_field_for_type in optimization_container_utils.jl
get_store_container_type(
    ::OptimizationContainerKey{T, U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType} = store_field_for_type(T)

abstract type AbstractModelStore end

# Required fields for subtypes
# - :duals
# - :parameters
# - :variables
# - :aux_variables
# - :expressions

# Required methods for subtypes:
# - read_optimizer_stats
#
# Each subtype must have a field for each instance of STORE_CONTAINERS.

function Base.empty!(store::T) where {T <: AbstractModelStore}
    for (name, type) in zip(fieldnames(T), fieldtypes(T))
        val = get_data_field(store, name)
        try
            empty!(val)
        catch
            @error "Base.empty! must be customized for type $T or skipped"
            rethrow()
        end
    end
end

get_data_field(store::AbstractModelStore, ::Val{S}) where {S} = getfield(store, S)
@inline Base.@constprop :aggressive get_data_field(
    store::AbstractModelStore,
    type::Symbol,
) =
    get_data_field(store, Val(type))

function Base.isempty(store::T) where {T <: AbstractModelStore}
    for (name, type) in zip(fieldnames(T), fieldtypes(T))
        val = get_data_field(store, name)
        try
            !isempty(val) && return false
        catch
            @error "Base.isempty must be customized for type $T or skipped"
            rethrow()
        end
    end

    return true
end

# Route through `get_data_field` so subtypes with indirected containers (e.g.
# EmulationModelStore, whose fields sit inside `data_container`) can override field access.
@generated function list_fields(
    store::AbstractModelStore,
    ::Type{T},
) where {T <: OptimizationKeyType}
    field = QuoteNode(store_field_for_type(T))
    return :(return keys(get_data_field(store, Val($field))))
end

@generated function list_keys(
    store::AbstractModelStore,
    ::Type{T},
) where {T <: OptimizationKeyType}
    field = QuoteNode(store_field_for_type(T))
    return :(return collect(keys(get_data_field(store, Val($field)))))
end

@generated function get_value(
    store::AbstractModelStore,
    ::Type{T},
    ::Type{U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    K = key_for_type(T)
    field = QuoteNode(store_field_for_type(T))
    return :(return get_data_field(store, Val($field))[$K(T, U)])
end

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_value(
    store::AbstractModelStore, ::T, ::Type{U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType} =
    get_value(store, T, U)
