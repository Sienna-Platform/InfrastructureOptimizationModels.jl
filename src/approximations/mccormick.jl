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

"McCormick envelope upper-bound constraints: z ≤ ... (axis 1:2 in container)."
struct McCormickUpperConstraint <: ConstraintType end

"Combined McCormick envelope constraints for the full 4-constraint envelope (axis 1:4)."
struct McCormickConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables (axis 1:4)."
struct ReformulatedMcCormickConstraint <: ConstraintType end

# --- Scalar build_* (pure JuMP) ---

"""
    build_mccormick_envelope(model, x, y, z, x_min, x_max, y_min, y_max)

Build the four McCormick inequalities for z ≈ x·y at a single cell.
Returns a flat NamedTuple `(upper_1, upper_2, lower_1, lower_2)` of scalar
constraint refs.
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

Build only the upper-envelope McCormick inequalities (z ≤ ...) at a single
cell. Used when a tighter lower bound on z is supplied elsewhere.

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

Returns a flat NamedTuple `(c1, c2, c3, c4)` of scalar constraint refs.
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

# --- IOM adapters ---

"""
    add_mccormick_approx!(container, ::Type{C}, x_var, y_var, z_var, x_bounds, y_bounds, meta)

Allocate a `McCormickConstraint` container with axes `(name, 1:4, time)`,
loop `(name, t)`, call `build_mccormick_envelope` per cell, and slot the
four refs.
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
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        @assert x_bounds[i].max > x_bounds[i].min
        @assert y_bounds[i].max > y_bounds[i].min
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

Allocate `McCormickUpperConstraint` `(name, 1:2, time)`; loop, call
`build_mccormick_upper` per cell, slot two refs.
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
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        @assert x_bounds[i].max > x_bounds[i].min
        @assert y_bounds[i].max > y_bounds[i].min
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

Allocate `ReformulatedMcCormickConstraint` `(name, 1:4, time)`; loop, call
`build_reformulated_mccormick` per cell, slot four refs.
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
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        @assert x_bounds[i].max > x_bounds[i].min
        @assert y_bounds[i].max > y_bounds[i].min
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
