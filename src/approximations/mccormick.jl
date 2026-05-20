# McCormick envelope for bilinear products z = x·y.
#
# Three scalar build_* functions form the pure-JuMP math layer (no IOM deps):
#   build_mccormick_envelope  — 4 constraints (upper_1, upper_2, lower_1, lower_2)
#   build_mccormick_upper     — 2 constraints (upper_1, upper_2 only)
#   build_reformulated_mccormick — 4 constraints for the Bin2 separable identity
#
# Each has a matching IOM adapter (add_*_approx!) that allocates a container
# with a 1:N dimension matching the math's return shape, loops over (name, t),
# and writes the scalar refs into the slots.

# --- Container key types ---

"McCormick envelope upper-bound constraints: z ≤ … (legacy vectorized split)."
struct McCormickUpperConstraint <: ConstraintType end

"McCormick envelope lower-bound constraints: z ≥ … (legacy vectorized split)."
struct McCormickLowerConstraint <: ConstraintType end

"Combined McCormick envelope constraints for the full 4-constraint envelope."
struct McCormickConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables."
struct ReformulatedMcCormickConstraint <: ConstraintType end

# --- Scalar build_* (pure JuMP, primary API) ---

"""
    build_mccormick_envelope(model, x, y, z, x_min, x_max, y_min, y_max)

Build the four McCormick inequalities for z ≈ x·y at a single cell:
* upper_1: z ≤ x_max · y + x · y_min − x_max · y_min
* upper_2: z ≤ x_min · y + x · y_max − x_min · y_max
* lower_1: z ≥ x_min · y + x · y_min − x_min · y_min
* lower_2: z ≥ x_max · y + x · y_max − x_max · y_max

Returns a flat NamedTuple `(upper_1, upper_2, lower_1, lower_2)` of scalar
constraint refs. Inputs are JuMP scalars and plain Float64 bounds.
"""
function build_mccormick_envelope(
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    z::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    upper_1 = JuMP.@constraint(model, z <= x_max * y + x * y_min - x_max * y_min)
    upper_2 = JuMP.@constraint(model, z <= x_min * y + x * y_max - x_min * y_max)
    lower_1 = JuMP.@constraint(model, z >= x_min * y + x * y_min - x_min * y_min)
    lower_2 = JuMP.@constraint(model, z >= x_max * y + x * y_max - x_max * y_max)
    return (; upper_1, upper_2, lower_1, lower_2)
end

"""
    build_mccormick_upper(model, x, y, z, x_min, x_max, y_min, y_max)

Build only the upper-envelope McCormick inequalities (z ≤ …) at a single cell.
Used when a tighter lower bound on z is supplied elsewhere (NMDT residual
product under `tighten`, manual_sos2 inside its SOS2 envelope, etc.).

Returns a flat NamedTuple `(upper_1, upper_2)` of scalar constraint refs.
"""
function build_mccormick_upper(
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    z::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    upper_1 = JuMP.@constraint(model, z <= x_max * y + x * y_min - x_max * y_min)
    upper_2 = JuMP.@constraint(model, z <= x_min * y + x * y_max - x_min * y_max)
    return (; upper_1, upper_2)
end

"""
    build_reformulated_mccormick(model, x, y, zp1, zx, zy, x_min, x_max, y_min, y_max)

Build the four reformulated-McCormick inequalities for the Bin2 separable
identity (zp1 ≈ (x+y)², zx ≈ x², zy ≈ y²) at a single cell.

Returns a flat NamedTuple `(c1, c2, c3, c4)` of scalar constraint refs:
* c1, c2 are lower envelopes (zp1 − zx − zy ≥ 2 · …)
* c3, c4 are upper envelopes (zp1 − zx − zy ≤ 2 · …)
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
    return (; c1, c2, c3, c4)
end

# --- IOM adapters (allocate, loop, write) ---

"""
    add_mccormick_approx!(container, ::Type{C}, x_var, y_var, z_var, x_bounds, y_bounds, meta)

Allocate a `McCormickConstraint` container with axes `(name, 1:4, time)`,
loop over `(name, t)`, call `build_mccormick_envelope` per cell, and write
the four returned constraint refs into slots `1..4` of the container.

Returns the registered container.
"""
function add_mccormick_approx!(
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    z_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        IS.@assert_op x_bounds[i].max > x_bounds[i].min
        IS.@assert_op y_bounds[i].max > y_bounds[i].min
    end

    model = get_jump_model(container)
    target = add_constraints_container!(
        container, McCormickConstraint, C, name_axis, 1:4, time_axis; meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_mccormick_envelope(
                model,
                x_var[name, t], y_var[name, t], z_var[name, t],
                xmn, xmx, ymn, ymx,
            )
            for (k, ref) in enumerate(r)
                target[name, k, t] = ref
            end
        end
    end
    return target
end

"""
    add_mccormick_upper_approx!(container, ::Type{C}, x_var, y_var, z_var, x_bounds, y_bounds, meta)

Allocate a `McCormickUpperConstraint` container with axes `(name, 1:2, time)`,
loop over `(name, t)`, call `build_mccormick_upper` per cell, and write the
two returned constraint refs into slots `1..2`.

Returns the registered container.
"""
function add_mccormick_upper_approx!(
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    z_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        IS.@assert_op x_bounds[i].max > x_bounds[i].min
        IS.@assert_op y_bounds[i].max > y_bounds[i].min
    end

    model = get_jump_model(container)
    target = add_constraints_container!(
        container, McCormickUpperConstraint, C, name_axis, 1:2, time_axis; meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_mccormick_upper(
                model,
                x_var[name, t], y_var[name, t], z_var[name, t],
                xmn, xmx, ymn, ymx,
            )
            for (k, ref) in enumerate(r)
                target[name, k, t] = ref
            end
        end
    end
    return target
end

"""
    add_reformulated_mccormick_approx!(container, ::Type{C}, x_var, y_var, zp1, zx, zy, x_bounds, y_bounds, meta)

Allocate a `ReformulatedMcCormickConstraint` container with axes
`(name, 1:4, time)`, loop over `(name, t)`, call `build_reformulated_mccormick`
per cell, and write the four returned constraint refs into slots `1..4`.
"""
function add_reformulated_mccormick_approx!(
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    zp1_expr,
    zx_expr,
    zy_expr,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        IS.@assert_op x_bounds[i].max > x_bounds[i].min
        IS.@assert_op y_bounds[i].max > y_bounds[i].min
    end

    model = get_jump_model(container)
    target = add_constraints_container!(
        container, ReformulatedMcCormickConstraint, C, name_axis, 1:4, time_axis; meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_reformulated_mccormick(
                model,
                x_var[name, t], y_var[name, t],
                zp1_expr[name, t], zx_expr[name, t], zy_expr[name, t],
                xmn, xmx, ymn, ymx,
            )
            for (k, ref) in enumerate(r)
                target[name, k, t] = ref
            end
        end
    end
    return target
end

# --- Legacy vectorized build_* and register_* helpers ---
#
# Kept for callers in nmdt_discretization.jl, bin2.jl, and hybs.jl that have
# not yet migrated to the scalar+adapter pattern above. These will be removed
# in the sweep task once all callers are refactored.

"""
    build_mccormick_envelope(model, x, y, z, x_bounds, y_bounds; lower_bounds = true)

Legacy vectorized McCormick envelope over a `(name, t)` grid. Returns a
NamedTuple `(lower, upper)` where each side is a pair `(c, c)` of 2D
`DenseAxisArray`s indexed by `(name, t)`. `lower === nothing` when
`lower_bounds == false`.
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

"""
    build_reformulated_mccormick(model, x, y, zp1, zx, zy, x_bounds, y_bounds)

Legacy vectorized reformulated McCormick over the `(name, t)` grid. Returns
a 4-tuple of 2D `DenseAxisArray`s, one per cut.
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

"""
    register_mccormick_envelope!(container, ::Type{C}, mc, meta)

Legacy registration helper for the vectorized McCormick envelope. Splits
`mc.upper` into `McCormickUpperConstraint` and `mc.lower` (when non-nothing)
into `McCormickLowerConstraint`.
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

_register_mccormick_side!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:ConstraintType},
    ::Nothing,
    ::String,
) = nothing

"""
    register_reformulated_mccormick!(container, ::Type{C}, cons, meta)

Legacy registration helper for the vectorized reformulated McCormick.
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

register_reformulated_mccormick!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing
