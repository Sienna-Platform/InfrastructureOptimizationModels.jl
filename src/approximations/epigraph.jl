# Epigraph (Q^{L1}) LP-only lower bound for x² using tangent-line cuts.
# Pure LP — zero binary variables. Creates a variable z that lower-bounds
# x² (approximately) bounded from below by supporting hyperplanes of the
# parabola.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024), Q^{L1} relaxation.

"Expression container for epigraph quadratic approximation results."
struct EpigraphExpression <: ExpressionType end

"Auxiliary continuous variables (g₀, …, g_L) for tooth-based PWL approximations."
struct SawtoothAuxVariable <: VariableType end

"LP relaxation constraints (g_j ≤ 2g_{j-1}, g_j ≤ 2(1−g_{j-1}))."
struct SawtoothLPConstraint <: ConstraintType end

"Links g₀ to the normalized x value."
struct SawtoothLinkingConstraint <: ConstraintType end

"Variable representing a lower-bounded approximation of x² in epigraph relaxation."
struct EpigraphVariable <: VariableType end

"Tangent-line lower-bound constraints in epigraph relaxation."
struct EpigraphTangentConstraint <: ConstraintType end

"Tangent-line lower-bound expression fL used in the epigraph formulation."
struct EpigraphTangentExpression <: ExpressionType end

"""
    scale_back_g_basis(x_min, delta, g_var, name, t, levels)

Build the affine "scale back to actual dimensions" expression
    x² ≈ x_min² + (2·x_min·δ + δ²)·g₀ − Σ_{j ∈ levels} δ²·2^{−2j}·g_j
where `g_var` is the SawtoothAux g-basis variable container, `x_min` and
`delta = x_max − x_min` are per-name scalars, and `levels` selects which
g-basis levels participate in the residual sum.

Shared by sawtooth (PWL approximation) and epigraph (tangent cuts) — both
express the parabola anchor + residual decomposition in this form.
"""
@inline function scale_back_g_basis(x_min, delta, g_var, name, t, levels)
    return x_min^2 +
           (2.0 * x_min * delta + delta^2) * g_var[name, 0, t] -
           sum(delta^2 * 2.0^(-2j) * g_var[name, j, t] for j in levels)
end

"""
    scale_back_g_basis_scalar(x_min, delta, g_var, levels)

Scalar form of `scale_back_g_basis`: `g_var` is a 1D `DenseAxisArray`
indexed by j (the levels axis) for a single (name, t) cell.
"""
@inline function scale_back_g_basis_scalar(x_min, delta, g_var, levels)
    return x_min^2 +
           (2.0 * x_min * delta + delta^2) * g_var[0] -
           sum(delta^2 * 2.0^(-2j) * g_var[j] for j in levels)
end

"""
Config for epigraph (Q^{L1}) LP-only lower-bound quadratic approximation.

# Fields
- `depth::Int`: number of tangent-line breakpoints (2^depth + 1 tangent lines);
  pure LP, zero binary variables.
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int
end

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_quadratic_approx(config::EpigraphQuadConfig, model, x, x_min, x_max)

Scalar form: build the epigraph relaxation for a single JuMP scalar `x`
with bounds `[x_min, x_max]`. Creates the per-cell g basis (j ∈ 0:depth),
LP relaxation constraints, z variable, tangent expression `fL`, and the
`depth + 2` tangent cuts.

Returns a NamedTuple:
- `approximation`     :: JuMP.AffExpr   (1.0 · z)
- `z_var`             :: JuMP.VariableRef
- `g_var`             :: DenseAxisArray{VariableRef, 1} over `0:depth`
- `link_constraint`   :: scalar constraint linking g₀ to (x − x_min)/δ
- `lp_constraints`    :: DenseAxisArray{Constraint, 2} over `(1:depth, 1:2)`
- `tangent_expression`:: JuMP.AffExpr   (full-depth residual sum)
- `tangent_constraints`:: DenseAxisArray{Constraint, 1} over `1:(depth+2)`
"""
function build_quadratic_approx(
    config::EpigraphQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    IS.@assert_op config.depth >= 1
    IS.@assert_op x_max > x_min

    depth = config.depth
    delta = x_max - x_min
    z_ub = max(x_min^2, x_max^2)

    g_var = JuMP.@variable(
        model, [j = 0:depth],
        lower_bound = 0.0, upper_bound = 1.0,
        base_name = "SawtoothAux",
    )

    link_con = JuMP.@constraint(model, g_var[0] == (x - x_min) / delta)

    # T^L constraints: g_j ≤ 2 g_{j-1} and g_j ≤ 2(1 − g_{j-1}) for j = 1..L.
    lp_a = JuMP.@constraint(model, [j = 1:depth], g_var[j] <= 2.0 * g_var[j - 1])
    lp_b = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] <= 2.0 * (1.0 - g_var[j - 1]),
    )
    lp_cons = JuMP.Containers.DenseAxisArray{eltype(lp_a.data)}(
        undef, 1:depth, 1:2,
    )
    @views lp_cons.data[:, 1] .= lp_a.data
    @views lp_cons.data[:, 2] .= lp_b.data

    z_var = JuMP.@variable(
        model, lower_bound = 0.0, upper_bound = z_ub, base_name = "EpigraphVar",
    )

    fL_expr = JuMP.@expression(
        model,
        sum(delta^2 * 2.0^(-2j) * g_var[j] for j in 1:depth),
    )

    # Tangent cuts:
    #   k=1: z ≥ 0
    #   k=2: z ≥ 2·x_min + 2·δ·g₀ − 1
    #   k=j+2 for j=1..L: z ≥ scale_back_g_basis_scalar(1:j) − δ²·2^{−2j−2}
    tangent_zero = JuMP.@constraint(model, z_var >= 0.0)
    tangent_anchor = JuMP.@constraint(
        model, z_var >= 2.0 * x_min - 1.0 + 2.0 * delta * g_var[0],
    )
    tangent_levels = JuMP.@constraint(
        model, [j = 1:depth],
        z_var >=
            scale_back_g_basis_scalar(x_min, delta, g_var, 1:j) -
            delta^2 * 2.0^(-2j - 2),
    )

    tangent_cons = JuMP.Containers.DenseAxisArray{typeof(tangent_zero)}(
        undef, 1:(depth + 2),
    )
    tangent_cons[1] = tangent_zero
    tangent_cons[2] = tangent_anchor
    @views tangent_cons.data[3:end] .= tangent_levels.data

    approximation = JuMP.@expression(model, 1.0 * z_var)

    return (;
        approximation,
        z_var,
        g_var,
        link_constraint = link_con,
        lp_constraints = lp_cons,
        tangent_expression = fL_expr,
        tangent_constraints = tangent_cons,
    )
end

# --- IOM adapter (allocate, loop, write) ---

"""
    add_quadratic_approx!(config::EpigraphQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate all output containers (z, g, link/lp/tangent constraints, fL and
approximation expressions) with axes drawn from `x_var`'s `(name, t)` plus
the internal `(depth)` axes, then loop `(name, t)` calling the scalar
`build_quadratic_approx(::EpigraphQuadConfig, ...)` per cell. Writes the
scalar refs and small inner-axis arrays into the container slots.

Returns the registered `EpigraphExpression` container (the approximation).
"""
function add_quadratic_approx!(
    config::EpigraphQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    IS.@assert_op depth >= 1
    IS.@assert_op length(name_axis) == length(x_bounds)
    for b in x_bounds
        IS.@assert_op b.max > b.min
    end

    model = get_jump_model(container)

    z_target = add_variable_container!(
        container, EpigraphVariable, C, name_axis, time_axis; meta,
    )
    g_target = add_variable_container!(
        container, SawtoothAuxVariable, C, name_axis, 0:depth, time_axis; meta,
    )
    link_target = add_constraints_container!(
        container, SawtoothLinkingConstraint, C, name_axis, time_axis; meta,
    )
    fL_target = add_expression_container!(
        container, EpigraphTangentExpression, C, name_axis, time_axis; meta,
    )
    approx_target = add_expression_container!(
        container, EpigraphExpression, C, name_axis, time_axis; meta,
    )
    lp_target = add_constraints_container!(
        container, SawtoothLPConstraint, C, name_axis, 1:depth, 1:2, time_axis; meta,
    )
    tangent_target = add_constraints_container!(
        container, EpigraphTangentConstraint, C, name_axis, 1:(depth + 2), time_axis;
        meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            z_target[name, t] = r.z_var
            for j in 0:depth
                g_target[name, j, t] = r.g_var[j]
            end
            link_target[name, t] = r.link_constraint
            fL_target[name, t] = r.tangent_expression
            approx_target[name, t] = r.approximation
            for j in 1:depth, k in 1:2
                lp_target[name, j, k, t] = r.lp_constraints[j, k]
            end
            for j in 1:(depth + 2)
                tangent_target[name, j, t] = r.tangent_constraints[j]
            end
        end
    end
    return approx_target
end

# --- Legacy result struct + vectorized build + register
# (kept for the generic add_quadratic_approx! wrapper in common.jl until
# callers migrate; removed in sweep) ---

"""
Pure-JuMP result of legacy vectorized `build_quadratic_approx(::EpigraphQuadConfig, ...)`.
"""
struct EpigraphQuadResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    Z <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 2},
    G <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    LP <: JuMP.Containers.DenseAxisArray,
    LC <: JuMP.Containers.DenseAxisArray,
    FL <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    TC <: JuMP.Containers.DenseAxisArray,
} <: QuadraticApproxResult
    approximation::A
    z_var::Z
    g_var::G
    lp_constraints::LP
    link_constraints::LC
    tangent_expressions::FL
    tangent_constraints::TC
end

"""
    build_quadratic_approx(config::EpigraphQuadConfig, model, x, bounds)

Legacy vectorized form. LP-only lower bound for x² via 2^depth + 1
tangent-line cuts on the parabola at uniformly spaced breakpoints in
[x_min, x_max].
"""
function build_quadratic_approx(
    config::EpigraphQuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    IS.@assert_op config.depth >= 1
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.max > b.min
    end

    g_levels = 0:(config.depth)
    delta = JuMP.Containers.DenseAxisArray([b.max - b.min for b in bounds], name_axis)
    x_min_arr = JuMP.Containers.DenseAxisArray([b.min for b in bounds], name_axis)
    z_ub_arr = JuMP.Containers.DenseAxisArray(
        [max(b.min^2, b.max^2) for b in bounds],
        name_axis,
    )

    g_var = JuMP.@variable(
        model,
        [name = name_axis, j = g_levels, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "SawtoothAux",
    )

    link_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        g_var[name, 0, t] == (x[name, t] - x_min_arr[name]) / delta[name]
    )

    lp_a = JuMP.@constraint(
        model,
        [name = name_axis, j = 1:(config.depth), t = time_axis],
        g_var[name, j, t] <= 2.0 * g_var[name, j - 1, t],
    )
    lp_b = JuMP.@constraint(
        model,
        [name = name_axis, j = 1:(config.depth), t = time_axis],
        g_var[name, j, t] <= 2.0 * (1.0 - g_var[name, j - 1, t]),
    )
    lp_cons = JuMP.Containers.DenseAxisArray{eltype(lp_a.data)}(
        undef, name_axis, 1:(config.depth), 1:2, time_axis,
    )
    @views lp_cons.data[:, :, 1, :] .= lp_a.data
    @views lp_cons.data[:, :, 2, :] .= lp_b.data

    z_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        lower_bound = 0.0,
        upper_bound = z_ub_arr[name],
        base_name = "EpigraphVar",
    )

    fL_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(delta[name]^2 * 2.0^(-2j) * g_var[name, j, t] for j in 1:(config.depth))
    )

    tangent_zero = JuMP.@constraint(
        model, [name = name_axis, t = time_axis], z_var[name, t] >= 0.0,
    )
    tangent_anchor = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        z_var[name, t] >=
        2.0 * x_min_arr[name] - 1.0 + 2.0 * delta[name] * g_var[name, 0, t],
    )
    tangent_levels = JuMP.@constraint(
        model,
        [name = name_axis, j = 1:(config.depth), t = time_axis],
        z_var[name, t] >=
        scale_back_g_basis(
            x_min_arr[name], delta[name], g_var, name, t, 1:j,
        ) - delta[name]^2 * 2.0^(-2j - 2),
    )

    tangent_cons = JuMP.Containers.DenseAxisArray{eltype(tangent_zero.data)}(
        undef, name_axis, 1:(config.depth + 2), time_axis,
    )
    @views tangent_cons.data[:, 1, :] .= tangent_zero.data
    @views tangent_cons.data[:, 2, :] .= tangent_anchor.data
    @views tangent_cons.data[:, 3:end, :] .= tangent_levels.data

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        1.0 * z_var[name, t]
    )

    return EpigraphQuadResult(
        approximation,
        z_var,
        g_var,
        lp_cons,
        link_cons,
        fL_expr,
        tangent_cons,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::EpigraphQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    g_levels = axes(result.g_var, 2)
    lp_lvl_axis = axes(result.lp_constraints, 2)
    tangent_axis = axes(result.tangent_constraints, 2)

    z_target = add_variable_container!(
        container, EpigraphVariable, C, name_axis, time_axis; meta,
    )
    z_target.data .= result.z_var.data

    g_target = add_variable_container!(
        container, SawtoothAuxVariable, C, name_axis, g_levels, time_axis; meta,
    )
    g_target.data .= result.g_var.data

    link_target = add_constraints_container!(
        container, SawtoothLinkingConstraint, C, name_axis, time_axis; meta,
    )
    link_target.data .= result.link_constraints.data

    fL_target = add_expression_container!(
        container, EpigraphTangentExpression, C, name_axis, time_axis; meta,
    )
    fL_target.data .= result.tangent_expressions.data

    result_target = add_expression_container!(
        container, EpigraphExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    lp_target = add_constraints_container!(
        container, SawtoothLPConstraint, C, name_axis, lp_lvl_axis, 1:2, time_axis; meta,
    )
    lp_target.data .= result.lp_constraints.data

    tangent_target = add_constraints_container!(
        container, EpigraphTangentConstraint, C, name_axis, tangent_axis, time_axis;
        meta,
    )
    tangent_target.data .= result.tangent_constraints.data
    return
end
