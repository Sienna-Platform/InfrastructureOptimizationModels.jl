"""
Direction of flow on a two-terminal component, relative to its from and to terminals.
"""
abstract type FlowDirection end
struct FromTo <: FlowDirection end
struct ToFrom <: FlowDirection end

"""Meta tag for a directional constraint container ("ft" or "tf")."""
constraint_meta(::FromTo) = "ft"
constraint_meta(::ToFrom) = "tf"

"""
A `(min, max)` pair for each direction. Values are in the unit system of the model.
"""
const DirectionalMinMax = NamedTuple{(:from_to, :to_from), Tuple{MinMax, MinMax}}

"""Value of `limits` in direction `dir`. `limits` has the fields `from_to` and `to_from`."""
get_directional_value(limits, ::FromTo) = limits.from_to
get_directional_value(limits, ::ToFrom) = limits.to_from

"""`limits` seen from the other terminal: the two directions swap."""
reverse_directions(limits) = (from_to = limits.to_from, to_from = limits.from_to)

"""
Upper limit in direction `dir`: the smaller of `rating` and the `max` of that direction.
"""
effective_limit(limits::DirectionalMinMax, rating::Float64, dir::FlowDirection) =
    min(rating, get_directional_value(limits, dir).max)

"""
Add `flow(name, t) <= limit(name, t)` for each name and time step, in a container with
meta `constraint_meta(dir)`. The caller supplies `flow` with the sign of direction `dir`.
"""
function add_directional_limit_constraints!(
    container::OptimizationContainer,
    ::Type{C},
    ::Type{D},
    dir::FlowDirection,
    names::Vector{String},
    flow::F,
    limit::G,
) where {C <: ConstraintType, D, F, G}
    time_steps = get_time_steps(container)
    con = add_constraints_container!(
        container, C, D, names, time_steps; meta = constraint_meta(dir),
    )
    jump_model = get_jump_model(container)
    for name in names, t in time_steps
        con[name, t] = JuMP.@constraint(jump_model, flow(name, t) <= limit(name, t))
    end
    return
end

_direction_label(::FromTo) = "from_to"
_direction_label(::ToFrom) = "to_from"

_is_invalid_limit(value::Float64) = !isfinite(value) || value < 0.0

function _check_limit_value(::Type{D}, name::String, dir::FlowDirection, value) where {D}
    if _is_invalid_limit(value)
        throw(
            IS.InvalidValue(
                "$(nameof(D)) $(name) has $(_direction_label(dir)) limit $(value). \
                 A directional limit must be finite and non-negative.",
            ),
        )
    end
    return
end

function _warn_limit_names(::Type{D}, names::Vector{String}, message::String) where {D}
    isempty(names) && return
    shown = join(first(names, 5), ", ")
    @warn "$(length(names)) $(nameof(D)) components $(message): $(shown)" _group =
        LOG_GROUP_MODELS_VALIDATION
    return
end

"""
Check the directional limits of the components of type `D`. Each entry is
`(name, limits, rating)`. Error on a `max` that is negative, `NaN`, or `Inf`. Warn once
for limits that never bind and once for a `min` that is not zero; a `min` has no effect.
"""
function validate_directional_limits(
    ::Type{D},
    entries::Vector{Tuple{String, DirectionalMinMax, Float64}},
) where {D}
    non_binding = String[]
    nonzero_min = String[]
    for (name, lims, rating) in entries
        for dir in (FromTo(), ToFrom())
            value = get_directional_value(lims, dir)
            _check_limit_value(D, name, dir, value.max)
            if value.max >= rating
                push!(non_binding, name)
            end
            if !iszero(value.min)
                push!(nonzero_min, name)
            end
        end
    end
    _warn_limit_names(
        D, unique(non_binding), "have a directional limit at or above the rating",
    )
    _warn_limit_names(
        D, unique(nonzero_min), "have a directional min that is not zero; min is ignored",
    )
    return
end

"""
Check the time series values of a directional limit. Error at the first value that is
negative, `NaN`, or `Inf`, and name its time step.
"""
function validate_directional_limit_values(
    ::Type{D},
    name::String,
    dir::FlowDirection,
    values::AbstractVector{Float64},
) where {D}
    for (t, value) in enumerate(values)
        if _is_invalid_limit(value)
            throw(
                IS.InvalidValue(
                    "$(nameof(D)) $(name) has $(_direction_label(dir)) time series \
                     value $(value) at time step $(t). A directional limit must be \
                     finite and non-negative.",
                ),
            )
        end
    end
    return
end
