abstract type OptimizationContainerKey{
    T <: OptimizationKeyType,
    U <: InfrastructureSystemsType,
} end

# These functions define the column names of DataFrames in all read-output functions.
# The function get_second_dimension_output_column_name is only used if the output data
# has three or more dimensions.
# Parent packages can override these functions to provide their own column names.
# We could consider making the time dimension ("DateTime") customizable, but it's probably
# better not to.
get_first_dimension_output_column_name(::OptimizationContainerKey) = "name"
get_second_dimension_output_column_name(::OptimizationContainerKey) = "name2"

struct VariableKey{T <: VariableType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

struct ConstraintKey{T <: ConstraintType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

struct AuxVarKey{T <: AuxVariableType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

struct ParameterKey{T <: ParameterType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

struct InitialConditionKey{T <: InitialConditionType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

struct ExpressionKey{T <: ExpressionType, U <: InfrastructureSystemsType} <:
       OptimizationContainerKey{T, U}
    meta::String
end

get_entry_type(
    ::OptimizationContainerKey{T, <:InfrastructureSystemsType},
) where {T <: OptimizationKeyType} = T
get_component_type(
    ::OptimizationContainerKey{<:OptimizationKeyType, U},
) where {U <: InfrastructureSystemsType} = U

# okay to construct AuxVarKey with abstract component type, but not others.
maybe_throw_if_abstract(::Type{T}, ::Type{U}) where {T <: OptimizationKeyType, U} =
    isabstracttype(U) && throw(ArgumentError("Type $U can't be abstract"))

maybe_throw_if_abstract(::Type{<:ConstraintType}, ::Type{U}) where {U} = nothing

const CONTAINER_KEY_EMPTY_META = ""

# see https://discourse.julialang.org/t/parametric-constructor-where-type-being-constructed-is-parameter/129866/3
function (M::Type{S} where {S <: OptimizationContainerKey})(
    ::Type{T},
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    check_meta_chars(meta)
    maybe_throw_if_abstract(T, U)
    return M{T, U}(meta)
end

function (M::Type{S} where {S <: OptimizationContainerKey})(
    ::Type{T},
    ::Type{U},
    meta::String,
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    maybe_throw_if_abstract(T, U)
    return M{T, U}(meta)
end

function make_key(
    ::Type{S},
    ::Type{T},
    ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {
    S <: OptimizationContainerKey,
    T <: OptimizationKeyType,
    U <: InfrastructureSystemsType,
}
    return S{T, U}(meta)
end

### Encoding keys ###

# IS `strip_module_name` reads `T.parameters`, which only exists on `DataType`. PSY6
# parameterized some component families on a unit-system type (e.g.
# `ReserveDemandCurve{ReserveUp}` is now `ReserveDemandCurve{ReserveUp, U} where U`),
# making the partially-applied type a `UnionAll` with no `.parameters` -> FieldError.
# Handle that here by keeping the bound parameters (dropping the free `TypeVar`s) so the
# encoded key stays unique per direction (ReserveUp vs ReserveDown).
#
# CAVEAT: this is a local workaround. The proper fix is upstream in IS
# `strip_module_name(::Type{T})`, which should handle `T isa UnionAll` the same way it
# already handles `T isa Union`. We can't make that change from here because IS is pulled
# in as a pinned git-branch dependency, not an editable in-repo checkout. Once IS gains
# UnionAll support, this helper can be dropped in favor of calling `strip_module_name(U)`
# directly again.
function _strip_module_name_for_key(U::Type)
    U isa UnionAll || return strip_module_name(U)
    base = Base.unwrap_unionall(U)
    parts = String[]
    for p in base.parameters
        p isa TypeVar && continue
        push!(parts, p isa Type ? string(nameof(p)) : string(p))
    end
    return if isempty(parts)
        string(nameof(U))
    else
        string(nameof(U)) * "{" * join(parts, ", ") * "}"
    end
end

@generated function encode_symbol(
    ::Type{T},
    ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    meta_str = :meta
    U_str =
        replace(
            replace(_strip_module_name_for_key(U), "{" => COMPONENT_NAME_DELIMITER),
            "}" => "",
        )
    T_str = strip_module_name(T)

    :(Symbol(
        $T_str * COMPONENT_NAME_DELIMITER * $U_str *
        (isempty($meta_str) ? "" : COMPONENT_NAME_DELIMITER * $meta_str),
    ))
end

function encode_key(
    key::OptimizationContainerKey{T, U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    return encode_symbol(T, U, key.meta)
end

encode_key_as_string(key::OptimizationContainerKey) = string(encode_key(key))
encode_keys_as_strings(container_keys) = [encode_key_as_string(k) for k in container_keys]

function check_meta_chars(meta::String)
    # Underscores in this field will prevent us from being able to decode keys.
    if occursin(COMPONENT_NAME_DELIMITER, meta)
        throw(InvalidValue("'$COMPONENT_NAME_DELIMITER' is not allowed in meta"))
    end
end

function should_write_resulting_value(
    ::OptimizationContainerKey{T, <:InfrastructureSystemsType},
) where {T <: OptimizationKeyType}
    return should_write_resulting_value(T)
end

function convert_output_to_natural_units(
    ::OptimizationContainerKey{T, <:InfrastructureSystemsType},
) where {T <: OptimizationKeyType}
    return convert_output_to_natural_units(T)
end

Base.convert(::Type{ExpressionKey}, name::Symbol) = ExpressionKey(decode_symbol(name)...)
Base.convert(::Type{ConstraintKey}, name::Symbol) = ConstraintKey(decode_symbol(name)...)
