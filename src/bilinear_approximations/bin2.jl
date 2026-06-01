# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity: x·y = (1/2)*((x+y)² − x² - y²).
# Calls existing quadratic approximation functions for p²=(x+y)²

"Expression container for bilinear product (x·y) approximation results."
struct BilinearProductExpression <: ExpressionType end
"Variable container for bilinear product (x ̇y) approximation results."
struct BilinearProductVariable <: VariableType end
"Expression container for adding variables."
struct VariableSumExpression <: ExpressionType end
"Expression container for subtracting variables."
struct VariableDifferenceExpression <: ExpressionType end
"Constraint container for linking product expressions and variables."
struct BilinearProductLinkingConstraint <: ConstraintType end

# --- Bilinear approximation config hierarchy ---

"Abstract supertype for bilinear approximation method configurations."
abstract type BilinearApproxConfig end

"""
Config for Bin2 bilinear approximation using z = ½((x+y)² − x² − y²).

# Fields
- `quad_config::Q`: quadratic method used for x², y², and (x+y)²
- `add_mccormick::Bool`: whether to add reformulated McCormick cuts through separable variables (default true)

The Q type parameter lets tolerance helpers dispatch on the inner quad method;
see `tolerance_depth(::Type{Bin2Config{Q}}; …)`.
"""
struct Bin2Config{Q <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::Q
    add_mccormick::Bool

    Bin2Config(
        quad_config::Q;
        add_mccormick::Bool = true,
    ) where {Q <: QuadraticApproxConfig} =
        new{Q}(quad_config, add_mccormick)
end

# --- Tolerance helpers ---
#
# Notation: Δx, Δy are domain lengths (Δx = x_max − x_min); ε denotes errors.
#
# Bilinear identity:  xy = ½((x+y)² − x² − y²).
# Approximation:      z  = ½(z_p − z_x − z_y), where z_• is the inner-quad
# approximation of •² for • ∈ {x, y, x+y}.
#
# Let ε_x = z_x − x², ε_y = z_y − y², ε_p = z_p − (x+y)² be the per-term
# inner-quad errors. The inner quad's worst-case error magnitude scales as
# Δ²·c at depth L (c is the per-unit error coefficient — see
# `tolerance_depth(::Type{<:QuadraticApproxConfig})` for why Δ² appears), so
#   |ε_x| ≤ ε_x^max = Δx²·c,
#   |ε_y| ≤ ε_y^max = Δy²·c,
#   |ε_p| ≤ ε_p^max = (Δx+Δy)²·c.
#
# Substitute z_x = x² + ε_x, z_y = y² + ε_y, z_p = (x+y)² + ε_p into
# z = ½(z_p − z_x − z_y):
#   z = ½((x+y)² + ε_p − x² − ε_x − y² − ε_y)
#     = ½(2xy + ε_p − ε_x − ε_y)
#     = xy + ½(ε_p − ε_x − ε_y).
# So |z − xy| = ½|ε_p − ε_x − ε_y|. The worst-case depends on which side each
# ε can take:
#
# **One-sided-over inner quads** (Sawtooth, SolverSOS2, ManualSOS2): each
# ε_• ∈ [0, ε_•^max], so ε_p − ε_x − ε_y ∈ [−(ε_x^max + ε_y^max), ε_p^max].
# Therefore |z − xy| ≤ max(½ε_p^max, ½(ε_x^max + ε_y^max)). Since
# (Δx+Δy)² ≥ Δx² + Δy², ε_p^max ≥ ε_x^max + ε_y^max and the max collapses
# to ½ε_p^max. To hit τ, ask the inner Q for ε_p^max ≤ 2τ on Δx+Δy.
#
# **Two-sided inner quads** (NMDT, DNMDT — the McCormick on the δ·δ or δ·xh
# residual product has slack even at integer β in MIP, so the inner result
# straddles x²): each ε_• ∈ [−ε_•^max, ε_•^max], and the triangle inequality
# gives |z − xy| ≤ ½(ε_p^max + ε_x^max + ε_y^max) = c·(Δx² + Δy² + Δx·Δy)
# (the last equality uses ε_p^max = (Δx+Δy)²·c and expands the square).
# To hit τ, ask the inner Q for c ≤ τ/(Δx² + Δy² + Δx·Δy), which is
# equivalent to forwarding tolerance = (Δx+Δy)²·τ/(Δx² + Δy² + Δx·Δy) at
# max_delta = Δx+Δy.

"""
    tolerance_depth(::Type{Bin2Config{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Inner-quad depth such that Bin2's worst-case gap `|z − xy|` is ≤ `tolerance`.
Derivation: see the comment block above.

For **one-sided-over** inner quads (`SawtoothQuadConfig`, `SolverSOS2QuadConfig`,
`ManualSOS2QuadConfig`), forwards to
`tolerance_depth(Q; tolerance = 2·τ, max_delta = Δx + Δy)`.

For **two-sided** inner quads (`NMDTQuadConfig`, `DNMDTQuadConfig`), forwards
to `tolerance_depth(Q; tolerance = (Δx+Δy)²·τ/(Δx² + Δy² + Δx·Δy),
max_delta = Δx + Δy)`. For balanced `Δx = Δy = Δ` this is simply
`tolerance_depth(Q; tolerance = (4/3)·τ, max_delta = 2Δ)`.

`EpigraphQuadConfig` is excluded — it is one-sided-under, so its auxiliary
`z_x` has no upper bound in the LP relaxation and can drive `z` arbitrarily
far from `xy`.
"""
function tolerance_depth(
    ::Type{Bin2Config{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {
    Q <: Union{SawtoothQuadConfig, SolverSOS2QuadConfig, ManualSOS2QuadConfig},
}
    return tolerance_depth(Q;
        tolerance = 2 * tolerance,
        max_delta = max_delta_x + max_delta_y,
    )
end

function tolerance_depth(
    ::Type{Bin2Config{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: Union{NMDTQuadConfig, DNMDTQuadConfig}}
    sum_sq = max_delta_x^2 + max_delta_y^2 + max_delta_x * max_delta_y
    max_delta = max_delta_x + max_delta_y
    return tolerance_depth(Q;
        tolerance = max_delta^2 * tolerance / sum_sq,
        max_delta = max_delta,
    )
end

# --- Unified bilinear approximation dispatch ---

"""
    _add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Standard form: compute x² and y² quadratic approximations, then delegate to precomputed form.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _add_bilinear_approx!(
    config::Bin2Config,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    xsq = _add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        x_var, x_bounds, meta * "_x",
    )
    ysq = _add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        y_var, y_bounds, meta * "_y",
    )
    return _add_bilinear_approx!(
        config, container, C, names, time_steps,
        xsq, ysq, x_var, y_var,
        x_bounds, y_bounds, meta,
    )
end

"""
    _add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed form: Bin2 identity z = ½((x+y)² − x² − y²) with optional PWMCC concave cuts.
Accepts pre-computed quadratic approximations `xsq` ≈ x² and `ysq` ≈ y².

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _add_bilinear_approx!(
    config::Bin2Config,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xsq,
    ysq,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    # --- Bin2 identity: z = ½((x+y)² − x² − y²) ---

    # Bounds for p = x + y (per-name)
    p_bounds = [
        MinMax((
            min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max,
        )) for i in eachindex(x_bounds)
    ]

    meta_plus = meta * "_plus"

    p_expr = add_expression_container!(
        container,
        VariableSumExpression,
        C,
        names,
        time_steps;
        meta = meta_plus,
    )
    for name in names, t in time_steps
        p = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p, x_var[name, t], 1.0)
        add_proportional_to_jump_expression!(p, y_var[name, t], 1.0)
        p_expr[name, t] = p
    end

    # Approximate p² = (x+y)² using the provided quadratic config
    psq = _add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        p_expr, p_bounds, meta_plus,
    )

    result_expr = add_expression_container!(
        container,
        BilinearProductExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        # z = (1/2) * (p² − x² − y²)
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(result, psq[name, t], 0.5)
        add_proportional_to_jump_expression!(result, xsq[name, t], -0.5)
        add_proportional_to_jump_expression!(result, ysq[name, t], -0.5)
    end

    # --- Reformulated McCormick cuts (optional) ---
    if config.add_mccormick
        _add_reformulated_mccormick!(
            container, C, names, time_steps,
            x_var, y_var, psq, xsq, ysq,
            x_bounds, y_bounds, meta,
        )
    end

    return result_expr
end
