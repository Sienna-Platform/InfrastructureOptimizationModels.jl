# McCormick envelope for bilinear products z = x·y.
# Adds up to 4 linear inequalities that bound z given variable bounds on x and y.
# The lower envelopes (z ≥ …) can be omitted when a tighter lower bound is
# supplied elsewhere; the upper envelopes (z ≤ …) are always present.

"McCormick envelope lower-bound constraints: z ≥ …  (entries c1, c2)."
struct McCormickLowerConstraint <: ConstraintType end

"McCormick envelope upper-bound constraints: z ≤ …  (entries c3, c4)."
struct McCormickUpperConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables."
struct ReformulatedMcCormickConstraint <: ConstraintType end

# --- Pure-JuMP single-element helpers ---

"""
    build_mccormick_envelope(model, x, y, z, x_min, x_max, y_min, y_max; lower_bounds = true)

Build the McCormick inequalities bounding `z ≈ x·y` on `model`. Returns a
NamedTuple `(lower, upper)` of `(c, c)` tuples (`lower === nothing` when
`lower_bounds == false`). Inputs may be any `JuMP.AbstractJuMPScalar`.
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
    c3 = JuMP.@constraint(model, z <= x_max * y + x * y_min - x_max * y_min)
    c4 = JuMP.@constraint(model, z <= x_min * y + x * y_max - x_min * y_max)
    lower = if lower_bounds
        c1 = JuMP.@constraint(model, z >= x_min * y + x * y_min - x_min * y_min)
        c2 = JuMP.@constraint(model, z >= x_max * y + x * y_max - x_max * y_max)
        (c1, c2)
    else
        nothing
    end
    return (lower = lower, upper = (c3, c4))
end

"""
    build_mccormick_envelope(model, x, y, z, x_bounds, y_bounds; lower_bounds = true)

Vectorized McCormick envelope over a `(name, t)` grid. Returns a NamedTuple
`(lower, upper)` where each side is a pair `(c, c)` of 2D `DenseAxisArray`s
indexed by `(name, t)`. `lower === nothing` when `lower_bounds == false`.
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
    for i in eachindex(x_bounds)
        IS.@assert_op x_bounds[i].max > x_bounds[i].min
        IS.@assert_op y_bounds[i].max > y_bounds[i].min
    end

    xmin = JuMP.Containers.DenseAxisArray([b.min for b in x_bounds], name_axis)
    xmax = JuMP.Containers.DenseAxisArray([b.max for b in x_bounds], name_axis)
    ymin = JuMP.Containers.DenseAxisArray([b.min for b in y_bounds], name_axis)
    ymax = JuMP.Containers.DenseAxisArray([b.max for b in y_bounds], name_axis)

    upper_1 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        z[name, t] <=
        xmax[name] * y[name, t] + x[name, t] * ymin[name] -
        xmax[name] * ymin[name],
    )
    upper_2 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        z[name, t] <=
        xmin[name] * y[name, t] + x[name, t] * ymax[name] -
        xmin[name] * ymax[name],
    )
    lower = if lower_bounds
        lower_1 = JuMP.@constraint(
            model,
            [name = name_axis, t = time_axis],
            z[name, t] >=
            xmin[name] * y[name, t] + x[name, t] * ymin[name] -
            xmin[name] * ymin[name],
        )
        lower_2 = JuMP.@constraint(
            model,
            [name = name_axis, t = time_axis],
            z[name, t] >=
            xmax[name] * y[name, t] + x[name, t] * ymax[name] -
            xmax[name] * ymax[name],
        )
        (lower_1, lower_2)
    else
        nothing
    end
    return (lower = lower, upper = (upper_1, upper_2))
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

Vectorized reformulated McCormick over the `(name, t)` grid. Returns a
4-tuple of 2D `DenseAxisArray`s, one per cut.
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
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        IS.@assert_op x_bounds[i].max > x_bounds[i].min
        IS.@assert_op y_bounds[i].max > y_bounds[i].min
    end

    xmin = JuMP.Containers.DenseAxisArray([b.min for b in x_bounds], name_axis)
    xmax = JuMP.Containers.DenseAxisArray([b.max for b in x_bounds], name_axis)
    ymin = JuMP.Containers.DenseAxisArray([b.min for b in y_bounds], name_axis)
    ymax = JuMP.Containers.DenseAxisArray([b.max for b in y_bounds], name_axis)

    c1 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        zp1[name, t] - zx[name, t] - zy[name, t] >=
        2.0 * (xmin[name] * y[name, t] + x[name, t] * ymin[name] -
         xmin[name] * ymin[name]),
    )
    c2 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        zp1[name, t] - zx[name, t] - zy[name, t] >=
        2.0 * (xmax[name] * y[name, t] + x[name, t] * ymax[name] -
         xmax[name] * ymax[name]),
    )
    c3 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        zp1[name, t] - zx[name, t] - zy[name, t] <=
        2.0 * (xmax[name] * y[name, t] + x[name, t] * ymin[name] -
         xmax[name] * ymin[name]),
    )
    c4 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        zp1[name, t] - zx[name, t] - zy[name, t] <=
        2.0 * (xmin[name] * y[name, t] + x[name, t] * ymax[name] -
         xmin[name] * ymax[name]),
    )
    return (c1, c2, c3, c4)
end

# --- IOM-side McCormick container registration ---

"""
    register_mccormick_envelope!(container, ::Type{C}, mc, meta)

Register a McCormick envelope (NamedTuple `(lower, upper)` as returned by
the vectorized `build_mccormick_envelope`) into the optimization container.
`mc.upper` is written under `McCormickUpperConstraint`; `mc.lower`, when
non-`nothing`, under `McCormickLowerConstraint`.
"""
function register_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    mc::NamedTuple,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    _register_mccormick_side!(container, C, McCormickUpperConstraint, mc.upper, meta)
    _register_mccormick_side!(container, C, McCormickLowerConstraint, mc.lower, meta)
    return
end

# No-op when this McCormick envelope was disabled at the call site.
register_mccormick_envelope!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing

function _register_mccormick_side!(
    container::OptimizationContainer,
    ::Type{C},
    ::Type{K},
    cons::Tuple{<:JuMP.Containers.DenseAxisArray, <:JuMP.Containers.DenseAxisArray},
    meta::String,
) where {
    C <: IS.InfrastructureSystemsComponent,
    K <: ConstraintType,
}
    c1, c2 = cons
    name_axis = axes(c1, 1)
    time_axis = axes(c1, 2)
    target = add_constraints_container!(
        container, K, C, name_axis, 1:2, time_axis; meta,
    )
    @views target.data[:, 1, :] .= c1.data
    @views target.data[:, 2, :] .= c2.data
    return
end

# No-op when the lower-bound side wasn't built.
_register_mccormick_side!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:ConstraintType},
    ::Nothing,
    ::String,
) = nothing

"""
    register_reformulated_mccormick!(container, ::Type{C}, cons, meta)

Register a reformulated McCormick constraint set (the 4-tuple of 2D
constraint containers returned by `build_reformulated_mccormick`) into the
optimization container under `ReformulatedMcCormickConstraint`.
"""
function register_reformulated_mccormick!(
    container::OptimizationContainer,
    ::Type{C},
    cons::Tuple{
        <:JuMP.Containers.DenseAxisArray,
        <:JuMP.Containers.DenseAxisArray,
        <:JuMP.Containers.DenseAxisArray,
        <:JuMP.Containers.DenseAxisArray,
    },
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    c1, c2, c3, c4 = cons
    name_axis = axes(c1, 1)
    time_axis = axes(c1, 2)
    target = add_constraints_container!(
        container,
        ReformulatedMcCormickConstraint,
        C,
        name_axis,
        1:4,
        time_axis;
        meta,
    )
    @views target.data[:, 1, :] .= c1.data
    @views target.data[:, 2, :] .= c2.data
    @views target.data[:, 3, :] .= c3.data
    @views target.data[:, 4, :] .= c4.data
    return
end

# No-op when this McCormick envelope was disabled at the call site.
register_reformulated_mccormick!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing
