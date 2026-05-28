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
see `tol_depth(::Type{Bin2Config{Q}}; …)`.
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
# Bilinear identity:  xy = ½((x+y)² − x² − y²).
# Approximation:      z  = ½(z_p − z_x − z_y),  with each z_• an inner-quad approx.
# Error decomposition: z − xy = ½(Δ_p − Δ_x − Δ_y),  where
#   ε_x = Δx²·ε_unit,  ε_y = Δy²·ε_unit,  ε_p = (Δx+Δy)²·ε_unit
# all sharing the inner quad's per-unit error coefficient at the same depth L.

"""
    tol_depth(::Type{Bin2Config{Q}}; tolerance, max_delta_x, max_delta_y)::Int

For one-sided-over inner quads (`Sawtooth`, `SolverSOS2`, `ManualSOS2`):

each Δ ∈ [0, ε]. Worst case at a corner of the Δ-box gives
`|z − xy| ≤ max(½ε_p, ½(ε_x + ε_y))`. Since `(Δx+Δy)² ≥ Δx² + Δy²` always,
`½ε_p` dominates, so `|z − xy| ≤ ½ε_p`. To hit user-target `τ`, request the
inner quad with `tolerance = 2τ` at `max_delta = Δx + Δy`.
"""
function tol_depth(
    ::Type{Bin2Config{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: Union{SawtoothQuadConfig, SolverSOS2QuadConfig, ManualSOS2QuadConfig}}
    return tol_depth(Q;
        tolerance = 2 * tolerance,
        max_delta = max_delta_x + max_delta_y,
    )
end

# Other inner quads are not supported as Bin2 inner:
#
# - EpigraphQuadConfig: the inner result is a free variable z ∈ [epigraph(x), ub],
#   not pinned. Bin2's affine combination ½(z_p − z_x − z_y) then has each z_•
#   slack inside its interval; under min/max objectives the LP drives the result
#   arbitrarily far from xy.
# - NMDTQuadConfig / DNMDTQuadConfig: Bin2 hands `_add_quadratic_approx!` an
#   AffExpr (x+y), but NMDT's `_normed_variable!` accepts only VariableRef.
#
# Both cases raise MethodError on `tol_depth` here (and at simulation time)
# until the underlying limitation is lifted.

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
