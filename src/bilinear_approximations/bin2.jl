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
# approximation of •² and (•)² for • ∈ {x, y, x+y}.
#
# Let ε_x = x² − z_x, ε_y = y² − z_y, ε_p = (x+y)² − z_p be the per-term
# inner-quad errors. For one-sided-over inner quads (Sawtooth, SolverSOS2,
# ManualSOS2), each ε_• ∈ [0, ε_•^max] where the bound scales as Δ²·c at
# depth L (c is the inner quad's per-unit error coefficient):
#   ε_x^max = Δx²·c,  ε_y^max = Δy²·c,  ε_p^max = (Δx+Δy)²·c.
#
# Substituting into z − xy yields  z − xy = ½(ε_x + ε_y − ε_p).
# With each ε_• ∈ [0, ε_•^max], the range of z − xy is
#   ½(0 + 0 − ε_p^max)  ≤  z − xy  ≤  ½(ε_x^max + ε_y^max − 0),
# so |z − xy| ≤ max(½ε_p^max, ½(ε_x^max + ε_y^max)).
#
# Now (Δx+Δy)² = Δx² + 2ΔxΔy + Δy² ≥ Δx² + Δy² (since Δx, Δy ≥ 0), so
# ε_p^max ≥ ε_x^max + ε_y^max, and the max collapses to ½ε_p^max.
# To hit user-target τ, ask the inner quad for ε_p^max ≤ 2τ on Δx+Δy.

"""
    tolerance_depth(::Type{Bin2Config{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Inner-quad depth such that Bin2's worst-case overestimation gap is ≤ `tolerance`.
Derivation: see the comment block above. Forwards to
`tolerance_depth(Q; tolerance = 2·τ, max_delta = Δx + Δy)`.

Defined for one-sided-over inner quads: `SawtoothQuadConfig`, `SolverSOS2QuadConfig`,
`ManualSOS2QuadConfig`, `NMDTQuadConfig`, `DNMDTQuadConfig`. `EpigraphQuadConfig`
is excluded — it is one-sided-under, so the sign of `ε_p` flips and the bound
above no longer applies; an Epigraph inner quad can drive `z` arbitrarily far
from `xy` under MIN/MAX objectives.

**Caveat for NMDT/DNMDT inner Q**: these are only one-sided-over when their
`epigraph_depth = 0`. With `epigraph_depth > 0`, the inner result becomes free
in `[epigraph(x), nmdt(x)]`, which crosses `x²` and breaks the derivation. Pass
NMDT/DNMDT inner Qs with `epigraph_depth = 0` only.
"""
function tolerance_depth(
    ::Type{Bin2Config{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {
    Q <: Union{
        SawtoothQuadConfig,
        SolverSOS2QuadConfig,
        ManualSOS2QuadConfig,
        NMDTQuadConfig,
        DNMDTQuadConfig,
    },
}
    return tolerance_depth(Q;
        tolerance = 2 * tolerance,
        max_delta = max_delta_x + max_delta_y,
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
