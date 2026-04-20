# Generic helpers for building constraints
#
# Design principles:
# - Single timestep only - device looping stays in caller (POM)
# - Pass limits directly, not devices
# - Use binary_var argument to unify simple and semicontinuous cases
# - Caller manages container creation (dense for bulk operations)

#######################################
######## Bound Direction (internal) ###
#######################################

"""
Abstract type for bound direction. Used to unify upper/lower bound logic.
"""
abstract type BoundDirection end
struct LowerBound <: BoundDirection end
struct UpperBound <: BoundDirection end

"""Meta tag for constraint container ("lb" or "ub")."""
constraint_meta(::LowerBound) = "lb"
constraint_meta(::UpperBound) = "ub"

"""Extract the relevant bound from a (min=, max=) NamedTuple."""
get_bound(::LowerBound, limits) = limits.min
get_bound(::UpperBound, limits) = limits.max

"""Create a bound constraint with the appropriate direction."""
_make_bound_constraint(::LowerBound, model, lhs, rhs) = JuMP.@constraint(model, lhs >= rhs)
_make_bound_constraint(::UpperBound, model, lhs, rhs) = JuMP.@constraint(model, lhs <= rhs)

#######################################
######## Range Constraint Helpers #####
#######################################

"""
Add a range bound constraint.

    LowerBound: variable >= bound * binary_var
    UpperBound: variable <= bound * binary_var

When `binary_var = 1.0` (default), this is a simple bound constraint.
When `binary_var` is a JuMP variable, this is a semicontinuous constraint.

# Arguments
- `dir`: `LowerBound()` or `UpperBound()`
- `jump_model`: the JuMP model
- `con_container`: constraint container (DenseAxisArray or SparseAxisArray)
- `name`: component name
- `t`: time period
- `variable`: the variable or expression to constrain
- `bound`: the bound value
- `binary_var`: multiplier for semicontinuous constraints (default: 1.0)
"""
function add_range_bound_constraint!(
    dir::BoundDirection,
    jump_model::JuMP.Model,
    con_container,
    name::String,
    t::Int,
    variable::JuMPOrFloat,
    bound::Float64,
    binary_var::JuMPOrFloat = 1.0,
)
    con_container[name, t] =
        _make_bound_constraint(dir, jump_model, variable, bound * binary_var)
    return
end

add_range_bound_constraint!(
    dir::BoundDirection,
    jump_model::JuMP.Model,
    con_container,
    name::String,
    t::Int,
    variable::JuMPOrFloat,
    bound::Int,
    binary_var::JuMPOrFloat = 1.0,
) = add_range_bound_constraint!(
    dir,
    jump_model,
    con_container,
    name,
    t,
    variable,
    Float64(bound),
    binary_var,
)

"""
Add an equality constraint.

    variable == value

Use this for the fixed-value case when min ≈ max.
"""
function add_range_equality_constraint!(
    jump_model::JuMP.Model,
    con_container,
    name::String,
    t::Int,
    variable::JuMPOrFloat,
    value::Float64,
)
    con_container[name, t] = JuMP.@constraint(jump_model, variable == value)
    return
end

#######################################
######## Ramp Constraint Helpers ######
#######################################

"""Create paired up/down constraint containers."""
function add_updown_constraints_containers!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{V},
    names,
    time_steps,
) where {T <: ConstraintType, V <: IS.InfrastructureSystemsComponent}
    return (
        up = add_constraints_container!(container, T, V, names, time_steps; meta = "up"),
        down = add_constraints_container!(
            container,
            T,
            V,
            names,
            time_steps;
            meta = "dn",
        ),
    )
end

"""
Add a pair of ramp-up and ramp-down constraints for a single (name, t).

    Ramp up:   current.up - previous <= limits.up * dt + slack.up
    Ramp down (bound_decrease=true, default):
               previous - current.down <= limits.down * dt + slack.down
    Ramp down (bound_decrease=false):
                previous - current.down + slack.down >= - limits.down * dt
        equivalent to: current.down - previous <= limits.down * dt + slack.down
            i.e. we're actually bounding the increase not the decrease.
            (the lower bound represents something different in the Expression case)

When `bound_decrease = false` (default), the down constraint mirrors the up constraint
direction, bounding how much power can increase from the previous value (using the down
ramp rate). When `bound_decrease = true`, the down constraint flips current and previous,
bounding how much power can decrease.

# Arguments
- `jump_model`: the JuMP model
- `cons`: UpDownPair of constraint containers (up=, down=)
- `name`: component name
- `t`: time period
- `current`: UpDownPair of current values (up=, down=); use same value for both if not using expressions
- `previous`: previous timestep value (ic_power for t=1, variable[t-1] for t>1)
- `limits`: UpDown (Float64) ramp limits
- `dt`: minutes per period
- `slack`: UpDownPair of slack variables
- `bound_decrease`: if false, flip current/previous in down constraint (default: true)
"""
@inline function add_ramp_constraint_pair!(
    jump_model::JuMP.Model,
    cons::UpDownPair{C}, # constraint containers
    name::String,
    t::Int,
    current::UpDownPair{V},
    previous::JuMPOrFloat,
    limits::UpDown,
    dt::Number,
    slack::UpDownPair{S},
    bound_decrease::Bool = true,
) where {C, V <: JuMPOrFloat, S <: JuMPOrFloat}
    cons.up[name, t] = JuMP.@constraint(
        jump_model,
        current.up - previous <= limits.up * dt + slack.up
    )
    if bound_decrease
        # must decrease by AT MOST ramp limit
        cons.down[name, t] = JuMP.@constraint(
            jump_model,
            previous - current.down <= limits.down * dt + slack.down
        )
    else
        # This reads as "must decrease by AT LEAST ramp limit," but it's correct.
        # The lower bound represents something different in the Expression case.
        cons.down[name, t] = JuMP.@constraint(
            jump_model,
            previous - current.down + slack.down >= -limits.down * dt
        )
    end
    return
end

"""
Add a pair of ramp-up and ramp-down constraints with start/stop condition terms.

Like [`add_ramp_constraint_pair!`](@ref), but includes start/stop condition terms that
relax the ramp limits when a unit is starting up or shutting down.

    Ramp up:   current.up - previous <= limits.up * dt + slack.up + startstop.up
    Ramp down: previous - current.down <= limits.down * dt + slack.down + startstop.down

# Arguments
- `jump_model`: the JuMP model
- `cons`: UpDownPair of constraint containers (up=, down=)
- `name`: component name
- `t`: time period
- `current`: UpDownPair of current values (up=, down=); use same value for both if not using expressions
- `previous`: previous timestep value (ic_power for t=1, variable[t-1] for t>1)
- `limits`: UpDown (Float64) ramp limits
- `dt`: minutes per period
- `startstop`: UpDownPair of start/stop condition terms
- `slack`: UpDownPair of slack variables
"""
@inline function add_ramp_constraint_startstop_pair!(
    jump_model::JuMP.Model,
    cons::UpDownPair{C}, # constraint containers
    name::String,
    t::Int,
    current::UpDownPair{V},
    previous::JuMPOrFloat,
    limits::UpDown,
    dt::Number,
    startstop::UpDownPair{R},
    slack::UpDownPair{S},
) where {C, V <: JuMPOrFloat, S <: JuMPOrFloat, R <: JuMPOrFloat}
    cons.up[name, t] = JuMP.@constraint(
        jump_model,
        current.up - previous <= limits.up * dt + slack.up + startstop.up
    )
    # must decrease by AT MOST ramp limit
    cons.down[name, t] = JuMP.@constraint(
        jump_model,
        previous - current.down <= limits.down * dt + startstop.down + slack.down
    )
    return
end
