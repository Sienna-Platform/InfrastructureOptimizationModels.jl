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
- `quad_config::QuadraticApproxConfig`: quadratic method used for x², y², and (x+y)²
- `add_mccormick::Bool`: whether to add reformulated McCormick cuts through separable variables (default true)
"""
struct Bin2Config <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    add_mccormick::Bool
end
Bin2Config(quad_config::QuadraticApproxConfig) = Bin2Config(quad_config, true)

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
