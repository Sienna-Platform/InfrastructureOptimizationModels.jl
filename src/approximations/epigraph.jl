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
Config for epigraph (Q^{L1}) LP-only lower-bound quadratic approximation.

# Fields
- `depth::Int`: number of tangent-line breakpoints (2^depth + 1 tangent lines);
  pure LP, zero binary variables.
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int
end

"""
Pure-JuMP result of `build_quadratic_approx(::EpigraphQuadConfig, ...)`.
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

LP-only lower bound for x² via 2^depth + 1 tangent-line cuts on the
parabola at uniformly spaced breakpoints in [x_min, x_max].
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

    # T^L constraints: g_j ≤ 2 g_{j-1} and g_j ≤ 2(1 − g_{j-1}) for j = 1..L.
    # Stack two depth × N × T families into a (name, j, k, t) container.
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

    # fL = Σ_{j=1..L} δ²·2^{−2j}·g_j  (full-depth residual sum used downstream
    # by the optional sawtooth tightening; the per-j partial sums for the
    # tangent cuts are formed inline below).
    fL_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(delta[name]^2 * 2.0^(-2j) * g_var[name, j, t] for j in 1:(config.depth))
    )

    # Tangent-line cuts:
    #   k=1: z ≥ 0
    #   k=2: z ≥ 2·x_min + 2·δ·g₀ − 1               (anchor at xh = 1/2)
    #   k=j+2 for j=1..L: z ≥ scale_back_g_basis(1:j) − δ²·2^{−2j−2}
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
