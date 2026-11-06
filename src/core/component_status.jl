#################################################################################
# Per-step component statuses
#################################################################################

"""
Statuses of the components of one type over the time steps, as returned by
[`get_component_statuses`](@ref) and read with [`get_component_status`](@ref).
"""
abstract type AbstractComponentStatuses end

"""
Every component is active at every step: [`get_component_status`](@ref) returns `1.0`. This
is what [`get_component_statuses`](@ref) returns by default.
"""
struct AllComponentsActive <: AbstractComponentStatuses end

"""
    ComponentStatusArray(values)

Statuses read from `values[name, t]`, a two-dimensional array whose first axis holds component
names and whose second axis holds the time steps. A component missing from the first axis is
active (`1.0`) at every step.

Construction hashes the names once; a lookup is then constant time. For statuses held in a
parameter container, use [`get_component_status_array`](@ref), which builds this once per
container.
"""
struct ComponentStatusArray{A <: AbstractArray{Float64, 2}} <: AbstractComponentStatuses
    values::A
    names::Set{String}
end

ComponentStatusArray(values::AbstractArray{Float64, 2}) =
    ComponentStatusArray(values, Set{String}(axes(values, 1)))

"""
    get_component_status_array(container, key::ParameterKey) -> ComponentStatusArray

The statuses held by the parameter container `key`, as a [`ComponentStatusArray`](@ref) that
reads its values in place. It is built once per container and reused by every later call, so
a [`get_component_statuses`](@ref) method can return it at no cost per lookup; the cache is
cleared by [`reset_optimization_model!`](@ref). The parameter array must hold `Float64`.
"""
function get_component_status_array(container::OptimizationContainer, key::ParameterKey)
    cache = container.component_status_cache
    cached = get(cache, key, nothing)
    isnothing(cached) || return cached::ComponentStatusArray
    statuses = ComponentStatusArray(get_parameter_array(container, key))
    cache[key] = statuses
    return statuses
end

"""
    get_component_statuses(container, ::Type{C}) -> AbstractComponentStatuses

Extension point: the per-step statuses of the components of type `C` in `container`. A status
is a `Float64`: `1.0` when the component is active at that step, `0.0` when it is inactive.
The default, [`AllComponentsActive`](@ref), keeps every component active.

The generic builders read it to scale what they assume of an active component:
range bounds on variables that are [`fixed_when_inactive`](@ref) and on range expressions,
the on status of the piecewise linear cost rows (with the commitment of an
always-committed component, see [`is_always_committed`](@ref)), the minimum-power offset of
block offers, the proportional cost booked for components without a commitment variable, and
the parameterized bounds. Ramp and minimum up/down rows do not read it: those limits still
apply while a component is inactive. [`fix_inactive_variables!`](@ref) fixes the variables a component holds while it is
inactive.

Downstream packages add a method for their component types, typically returning a
[`ComponentStatusArray`](@ref) built from a parameter container. A builder fetches the
statuses once per component type, so the method may allocate. Extend this function, not the
four-argument [`get_component_status`](@ref).
"""
get_component_statuses(::OptimizationContainer, ::Type) = AllComponentsActive()

"""
    get_component_status(statuses, name, t) -> Float64
    get_component_status(container, ::Type{C}, name, t) -> Float64

The status of component `name` at step `t`: `1.0` when active, `0.0` when inactive. The
second form reads [`get_component_statuses`](@ref)`(container, C)`; a builder that reads many
entries fetches the statuses once and uses the first form.
"""
get_component_status(::AllComponentsActive, ::AbstractString, ::Int) = 1.0

function get_component_status(statuses::ComponentStatusArray, name::AbstractString, t::Int)
    name in statuses.names || return 1.0
    return statuses.values[name, t]
end

get_component_status(
    container::OptimizationContainer,
    ::Type{C},
    name::AbstractString,
    t::Int,
) where {C} = get_component_status(get_component_statuses(container, C), name, t)

# An on status `value` (a number, a JuMP parameter or an expression) scaled by `status`.
# Status one keeps the value as it is and status zero drops it.
function _scale_by_status(value, status::Float64)
    isone(status) && return value
    iszero(status) && return 0.0
    return status * value
end

# The statuses that scale a bound on the entries of `U` for components of type `C`. A bound on
# a variable that stays free while its component is inactive (see `fixed_when_inactive`),
# such as a stored energy level, keeps its value. A bound on a variable fixed while inactive,
# or on a range expression, is scaled by the status.
_bound_statuses(
    container::OptimizationContainer,
    ::Type{U},
    ::Type{C},
) where {
    U <: VariableType,
    C,
} = fixed_when_inactive(U) ? get_component_statuses(container, C) : AllComponentsActive()

#################################################################################
# Component entries of a container
#################################################################################

"""
    component_axis(::Type{T}, array) -> Int

Extension point: the position, in the index of an entry of `array`, of the component name for
a container of key entry type `T` (a variable, constraint, expression, parameter or auxiliary
variable type). The time step is always the last index.

Defaults: `1` for a `DenseAxisArray`, whose entries are indexed `(name, ..., t)`; the
second-to-last position for an `N`-dimensional `SparseAxisArray`, whose entries are indexed
`(..., name, t)`. A type whose sparse entries hold the name elsewhere, such as
`(owner, name, scenario, t)`, adds a method.
"""
component_axis(::Type{<:OptimizationKeyType}, ::DenseAxisArray) = 1
component_axis(::Type{<:OptimizationKeyType}, ::SparseAxisArray{T, N}) where {T, N} = N - 1

"""
    component_axis_type(::Type{C}) -> Type

The component type whose names fill the component axis (see [`component_axis`](@ref)) of a
container keyed on `C`: `C` itself, or `A` for a [`ComponentPairKey`](@ref)`{A, B}`.
"""
component_axis_type(::Type{C}) where {C} = C
component_axis_type(::Type{ComponentPairKey{A, B}}) where {A, B} = A

"""
    each_component_entry(key, array)

Iterate the entries of the container `array` stored under `key` as `(name, t, entry)` tuples:
the component name (read at [`component_axis`](@ref)), the time step and the stored entry.
Every entry of a multi-dimensional container is visited, so a component appears once per
entry at a step. Unassigned entries of a dense container and `nothing` entries of a sparse
one are skipped.
"""
function each_component_entry(key::OptimizationContainerKey, array::DenseAxisArray)
    position = component_axis(get_entry_type(key), array)
    array_axes = axes(array)
    data = array.data
    return (
        (array_axes[position][I[position]], array_axes[end][I[end]], data[I]) for
        I in CartesianIndices(data) if isassigned(data, Tuple(I)...)
    )
end

function each_component_entry(key::OptimizationContainerKey, array::SparseAxisArray)
    position = component_axis(get_entry_type(key), array)
    return (
        (index[position], last(index), entry) for
        (index, entry) in array.data if !isnothing(entry)
    )
end

#################################################################################
# Inactive variables
#################################################################################

"""
    fixed_when_inactive(::Type{<:VariableType}) -> Bool

Extension point: whether [`fix_inactive_variables!`](@ref) fixes the variables of this type
to zero at the steps their component is inactive. Default `false`.

The range builders scale a bound on a variable by the component status exactly when its type
is fixed when inactive (see [`add_range_constraints!`](@ref)). The trait is global per type,
so declare it only for types that are never used as state, such as a stored energy level,
whose bounds must hold while the component is inactive.
"""
fixed_when_inactive(::Type{<:VariableType}) = false

"""
    fix_inactive_variables!(container)

Fix to zero every variable whose type is [`fixed_when_inactive`](@ref), at each step its
component's status (see [`get_component_statuses`](@ref)) is zero. The component of an
entry is read with [`each_component_entry`](@ref), and its type with
[`component_axis_type`](@ref) of the key's component type. Call it once the variables of
every component are built and before the model is solved; rows that scale by the status
stay feasible with the fixed variables.
"""
function fix_inactive_variables!(container::OptimizationContainer)
    for (key, array) in get_variables(container)
        fixed_when_inactive(get_entry_type(key)) || continue
        statuses =
            get_component_statuses(container, component_axis_type(get_component_type(key)))
        _fix_inactive_entries!(statuses, key, array)
    end
    return
end

_fix_inactive_entries!(::AllComponentsActive, ::VariableKey, _) = nothing

function _fix_inactive_entries!(
    statuses::AbstractComponentStatuses,
    key::VariableKey,
    array,
)
    for (name, t, entry) in each_component_entry(key, array)
        iszero(get_component_status(statuses, name, t)) || continue
        _fix_to_zero!(entry)
    end
    return
end

_fix_to_zero!(variable::JuMP.VariableRef) = JuMP.fix(variable, 0.0; force = true)
_fix_to_zero!(::Any) = nothing

#################################################################################
# Parameter values that follow the component status
#################################################################################

"""
    scaled_by_component_status(::Type{<:ParameterType}) -> Bool

Extension point: whether the effective value of a parameter (value × multiplier) is zero
while its component is inactive. The builders that read such a parameter in the model scale
it by the status; every path that reports or evaluates the parameter's value
([`lookup_value`](@ref), the stored outputs, [`read_parameters`](@ref)) applies the status
through [`calculate_parameter_values`](@ref)`(container, key)`. Default `false`.

The status scales a copy of the multiplier, so the parameter container is never modified;
dense and sparse containers are supported, with the component read at
[`component_axis`](@ref). `read_parameters` reports an
`ObjectiveFunctionParameter` without its multiplier, so without the status either.
"""
scaled_by_component_status(::Type{<:ParameterType}) = false

"""
    calculate_parameter_values(container, key::ParameterKey)

The effective values of the parameter container `key` (value × multiplier), with the
multiplier scaled by the component status (see [`get_component_statuses`](@ref)) when the
parameter type is [`scaled_by_component_status`](@ref).
"""
function calculate_parameter_values(container::OptimizationContainer, key::ParameterKey)
    param_container = get_parameter(container, key)
    return calculate_parameter_values(
        get_attributes(param_container),
        get_parameter_array(param_container),
        _status_scaled_multiplier(container, key, get_multiplier_array(param_container)),
    )
end

# The multiplier of `key` scaled by the component status, as a copy; the multiplier itself
# when the parameter type does not follow the status.
function _status_scaled_multiplier(
    container::OptimizationContainer,
    key::ParameterKey{P, C},
    multiplier,
) where {P <: ParameterType, C}
    scaled_by_component_status(P) || return multiplier
    statuses = get_component_statuses(container, component_axis_type(C))
    statuses isa AllComponentsActive && return multiplier
    return _scale_by_statuses(statuses, key, multiplier)
end

function _scale_by_statuses(
    statuses::AbstractComponentStatuses,
    key::ParameterKey,
    multiplier::DenseAxisArray,
)
    position = component_axis(get_entry_type(key), multiplier)
    array_axes = axes(multiplier)
    scaled = DenseAxisArray(copy(multiplier.data), array_axes...)
    for I in CartesianIndices(scaled.data)
        name = array_axes[position][I[position]]
        scaled.data[I] *= get_component_status(statuses, name, array_axes[end][I[end]])
    end
    return scaled
end

function _scale_by_statuses(
    statuses::AbstractComponentStatuses,
    key::ParameterKey,
    multiplier::SparseAxisArray,
)
    position = component_axis(get_entry_type(key), multiplier)
    return SparseAxisArray(
        Dict(
            index => value * get_component_status(statuses, index[position], last(index))
            for (index, value) in multiplier.data
        ),
    )
end
