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
    scale_back_g_basis(x_min, delta, g_var, levels)

Build the affine "scale back to actual dimensions" expression at one cell:
    x² ≈ x_min² + (2·x_min·δ + δ²)·g₀ − Σ_{j ∈ levels} δ²·2^{−2j}·g_j
where `g_var` is a 1D `DenseAxisArray` over the levels axis for a single
(name, t). Shared by sawtooth (PWL approximation) and epigraph (tangent cuts).
"""
@inline function scale_back_g_basis(x_min, delta, g_var, levels)
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

"""
    build_quadratic_approx(config::EpigraphQuadConfig, model, x, x_min, x_max)

Scalar form: build the epigraph relaxation for a single JuMP scalar `x`
with bounds `[x_min, x_max]`. Creates the per-cell g basis (j ∈ 0:depth),
LP relaxation constraints, z variable, tangent expression `fL`, and the
`depth + 2` tangent cuts.

Returns a NamedTuple with `(approximation, z_var, g_var, link_constraint,
lp_constraints, tangent_expression, tangent_constraints)`.
"""
function build_quadratic_approx(
    config::EpigraphQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    @assert config.depth >= 1
    @assert x_max > x_min

    depth = config.depth
    delta = x_max - x_min
    z_ub = max(x_min^2, x_max^2)

    g_var = JuMP.@variable(
        model, [j = 0:depth],
        lower_bound = 0.0, upper_bound = 1.0,
        base_name = "SawtoothAux",
    )

    link_con = JuMP.@constraint(model, g_var[0] == (x - x_min) / delta)

    lp_a = JuMP.@constraint(model, [j = 1:depth], g_var[j] <= 2.0 * g_var[j - 1])
    lp_b = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] <= 2.0 * (1.0 - g_var[j - 1]),
    )
    lp_cons = JuMP.Containers.DenseAxisArray{eltype(lp_a)}(
        undef, 1:depth, 1:2,
    )
    @views lp_cons.data[:, 1] .= lp_a
    @views lp_cons.data[:, 2] .= lp_b

    z_var = JuMP.@variable(
        model, lower_bound = 0.0, upper_bound = z_ub, base_name = "EpigraphVar",
    )

    fL_expr = JuMP.@expression(
        model,
        sum(delta^2 * 2.0^(-2j) * g_var[j] for j in 1:depth),
    )

    tangent_zero = JuMP.@constraint(model, z_var >= 0.0)
    tangent_anchor = JuMP.@constraint(
        model, z_var >= 2.0 * x_min - 1.0 + 2.0 * delta * g_var[0],
    )
    tangent_levels = JuMP.@constraint(
        model, [j = 1:depth],
        z_var >=
            scale_back_g_basis(x_min, delta, g_var, 1:j) -
            delta^2 * 2.0^(-2j - 2),
    )

    tangent_cons = JuMP.Containers.DenseAxisArray{typeof(tangent_zero)}(
        undef, 1:(depth + 2),
    )
    tangent_cons[1] = tangent_zero
    tangent_cons[2] = tangent_anchor
    @views tangent_cons.data[3:end] .= tangent_levels

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

# --- IOM allocation + per-cell write helpers (shared with sawtooth tightening
# and NMDT tightening) ---

function _alloc_epigraph_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, depth::Int, meta,
) where {C <: IS.InfrastructureSystemsComponent}
    return (
        z = add_variable_container!(
            container, EpigraphVariable, C, name_axis, time_axis; meta,
        ),
        g = add_variable_container!(
            container, SawtoothAuxVariable, C, name_axis, 0:depth, time_axis; meta,
        ),
        link = add_constraints_container!(
            container, SawtoothLinkingConstraint, C, name_axis, time_axis; meta,
        ),
        fL = add_expression_container!(
            container, EpigraphTangentExpression, C, name_axis, time_axis; meta,
        ),
        approx = add_expression_container!(
            container, EpigraphExpression, C, name_axis, time_axis; meta,
        ),
        lp = add_constraints_container!(
            container, SawtoothLPConstraint, C, name_axis, 1:depth, 1:2, time_axis;
            meta,
        ),
        tangent = add_constraints_container!(
            container, EpigraphTangentConstraint, C, name_axis, 1:(depth + 2), time_axis;
            meta,
        ),
    )
end

function _write_epigraph_cell!(targets, name, t, r, depth::Int)
    targets.z[name, t] = r.z_var
    for j in 0:depth
        targets.g[name, j, t] = r.g_var[j]
    end
    targets.link[name, t] = r.link_constraint
    targets.fL[name, t] = r.tangent_expression
    targets.approx[name, t] = r.approximation
    for j in 1:depth, k in 1:2
        targets.lp[name, j, k, t] = r.lp_constraints[j, k]
    end
    for j in 1:(depth + 2)
        targets.tangent[name, j, t] = r.tangent_constraints[j]
    end
    return
end

"""
    add_quadratic_approx!(config::EpigraphQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate all output containers (z, g, link/lp/tangent constraints, fL and
approximation expressions) with axes drawn from `x_var`'s `(name, t)` plus
the internal `(depth)` axes, then loop `(name, t)` calling the scalar
build per cell.
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
    @assert depth >= 1
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
    end

    model = get_jump_model(container)
    targets = _alloc_epigraph_targets!(container, C, name_axis, time_axis, depth, meta)

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            _write_epigraph_cell!(targets, name, t, r, depth)
        end
    end
    return targets.approx
end
