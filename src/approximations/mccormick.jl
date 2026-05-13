# McCormick envelope for bilinear products z = x·y.
# Adds 4 linear inequalities that bound z given variable bounds on x and y.

"Standard McCormick envelope constraints bounding the bilinear product z = x·y."
struct McCormickConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables."
struct ReformulatedMcCormickConstraint <: ConstraintType end

# --- Pure-JuMP single-element helpers ---

"""
    build_mccormick_envelope(model, x, y, z, x_min, x_max, y_min, y_max; lower_bounds = true)

Add the four McCormick inequalities bounding `z ≈ x·y` to `model` and return
them as a `(c1, c2, c3, c4)` tuple. If `lower_bounds == false`, the first
two constraints (`z ≥ …` lower envelopes) are omitted; the returned tuple
slots are `nothing` in their place.

Inputs may be `JuMP.AbstractJuMPScalar` (variable or affine expression).
"""
function build_mccormick_envelope(
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    z::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    lower_bounds::Bool = true,
)
    c1 = if lower_bounds
        JuMP.@constraint(model, z >= x_min * y + x * y_min - x_min * y_min)
    else
        nothing
    end
    c2 = if lower_bounds
        JuMP.@constraint(model, z >= x_max * y + x * y_max - x_max * y_max)
    else
        nothing
    end
    c3 = JuMP.@constraint(model, z <= x_max * y + x * y_min - x_max * y_min)
    c4 = JuMP.@constraint(model, z <= x_min * y + x * y_max - x_min * y_max)
    return (c1, c2, c3, c4)
end

"""
    build_mccormick_envelope(model, x, y, z, x_bounds, y_bounds; lower_bounds = true)

Vectorized McCormick envelope over a (name, t) grid: for each (name, t)
adds the four inequalities bounding `z[name, t] ≈ x[name, t] · y[name, t]`.
Returns a `DenseAxisArray` indexed by (name, k, t) where k ∈ 1:4 holds
the four constraints (or a `Union{Missing, ConstraintRef}` array entry
for the omitted lower bounds when `lower_bounds == false`).
"""
function build_mccormick_envelope(
    model::JuMP.Model,
    x,
    y,
    z,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax};
    lower_bounds::Bool = true,
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)

    cons = JuMP.Containers.DenseAxisArray{Any}(undef, name_axis, 1:4, time_axis)
    for (i, name) in enumerate(name_axis), t in time_axis
        xb = x_bounds[i]
        yb = y_bounds[i]
        IS.@assert_op xb.max > xb.min
        IS.@assert_op yb.max > yb.min
        c1, c2, c3, c4 = build_mccormick_envelope(
            model,
            x[name, t],
            y[name, t],
            z[name, t],
            xb.min,
            xb.max,
            yb.min,
            yb.max;
            lower_bounds,
        )
        cons[name, 1, t] = c1
        cons[name, 2, t] = c2
        cons[name, 3, t] = c3
        cons[name, 4, t] = c4
    end
    return cons
end

# --- Bin2 reformulated-McCormick helpers (used inside build_bilinear_approx(::Bin2Config, ...)) ---

"""
    build_reformulated_mccormick(model, x, y, zp1, zx, zy, x_min, x_max, y_min, y_max)

Build the four reformulated-McCormick inequalities for the Bin2 separable
identity, in terms of the quadratic approximations zp1 ≈ (x+y)², zx ≈ x²,
zy ≈ y². Returns the four constraints as a tuple.
"""
function build_reformulated_mccormick(
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    zp1::JuMP.AbstractJuMPScalar,
    zx::JuMP.AbstractJuMPScalar,
    zy::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    c1 = JuMP.@constraint(
        model,
        zp1 - zx - zy >= 2.0 * (x_min * y + x * y_min - x_min * y_min),
    )
    c2 = JuMP.@constraint(
        model,
        zp1 - zx - zy >= 2.0 * (x_max * y + x * y_max - x_max * y_max),
    )
    c3 = JuMP.@constraint(
        model,
        zp1 - zx - zy <= 2.0 * (x_max * y + x * y_min - x_max * y_min),
    )
    c4 = JuMP.@constraint(
        model,
        zp1 - zx - zy <= 2.0 * (x_min * y + x * y_max - x_min * y_max),
    )
    return (c1, c2, c3, c4)
end

"""
    build_reformulated_mccormick(model, x, y, zp1, zx, zy, x_bounds, y_bounds)

Vectorized reformulated McCormick over the (name, t) grid.
"""
function build_reformulated_mccormick(
    model::JuMP.Model,
    x,
    y,
    zp1,
    zx,
    zy,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    cons = JuMP.Containers.DenseAxisArray{Any}(undef, name_axis, 1:4, time_axis)
    for (i, name) in enumerate(name_axis), t in time_axis
        xb = x_bounds[i]
        yb = y_bounds[i]
        IS.@assert_op xb.max > xb.min
        IS.@assert_op yb.max > yb.min
        c1, c2, c3, c4 = build_reformulated_mccormick(
            model,
            x[name, t],
            y[name, t],
            zp1[name, t],
            zx[name, t],
            zy[name, t],
            xb.min,
            xb.max,
            yb.min,
            yb.max,
        )
        cons[name, 1, t] = c1
        cons[name, 2, t] = c2
        cons[name, 3, t] = c3
        cons[name, 4, t] = c4
    end
    return cons
end

# --- IOM-side McCormick container registration ---

"""
    register_mccormick_envelope!(container, ::Type{C}, cons, meta)

Register a McCormick constraint array (as returned by `build_mccormick_envelope`)
into the optimization container under `McCormickConstraint` with the given `meta`.
"""
function register_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    cons,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(cons, 1)
    time_axis = axes(cons, 3)
    target = add_constraints_container!(
        container,
        McCormickConstraint,
        C,
        collect(name_axis),
        1:4,
        time_axis;
        sparse = true,
        meta,
    )
    for name in name_axis, k in 1:4, t in time_axis
        c = cons[name, k, t]
        c === nothing && continue
        target[(name, k, t)] = c
    end
    return
end

"""
    register_reformulated_mccormick!(container, ::Type{C}, cons, meta)

Register a reformulated McCormick constraint array (as returned by
`build_reformulated_mccormick`) into the optimization container under
`ReformulatedMcCormickConstraint`.
"""
function register_reformulated_mccormick!(
    container::OptimizationContainer,
    ::Type{C},
    cons,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(cons, 1)
    time_axis = axes(cons, 3)
    target = add_constraints_container!(
        container,
        ReformulatedMcCormickConstraint,
        C,
        collect(name_axis),
        1:4,
        time_axis;
        sparse = true,
        meta,
    )
    for name in name_axis, k in 1:4, t in time_axis
        target[(name, k, t)] = cons[name, k, t]
    end
    return
end
