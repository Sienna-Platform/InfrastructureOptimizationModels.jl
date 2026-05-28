# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS (Hybrid Separable) bilinear approximation.

Combines Bin2 lower bound and Bin3 upper bound with shared quadratic for x², y²
and LP-only epigraph for (x+y)², (x−y)².

# Fields
- `quad_config::Q`: quadratic method used for the shared x² and y² terms
- `epigraph_depth::Int`: depth for the epigraph Q^{L1} LP-only approximation of cross-terms (x±y)²
- `add_mccormick::Bool`: whether to add standard McCormick envelope cuts on the product variable (default false)

The Q type parameter lets tolerance helpers dispatch on the inner quad method.
HybS sandwich validity requires Q to over-estimate (so `lower ≤ xy ≤ upper`):
only `Sawtooth`, `SolverSOS2`, `ManualSOS2` are supported by the tolerance
helpers; see `tol_depth(::Type{HybSConfig{Q}}; …)` and
`tol_epigraph_depth(::Type{HybSConfig{Q}}; …)`.
"""
struct HybSConfig{Q <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::Q
    epigraph_depth::Int
    add_mccormick::Bool

    HybSConfig(
        quad_config::Q;
        epigraph_depth::Int,
        add_mccormick::Bool = false,
    ) where {Q <: QuadraticApproxConfig} =
        new{Q}(quad_config, epigraph_depth, add_mccormick)
end

# --- Tolerance helpers ---
#
# HybS adds:
#   z ≥ ½(z_p1 − z_x − z_y)    z_p1 ≈ (x+y)² via Epigraph (one-sided under)
#   z ≤ ½(z_x + z_y − z_p2)    z_p2 ≈ (x−y)² via Epigraph (one-sided under)
#                              z_x ≈ x², z_y ≈ y² via inner Q (one-sided over)
# So z is free in [lower, upper] with
#   lower = xy + ½(Δ_p1 − Δ_x − Δ_y)    Δ_p ∈ [−ε_e, 0],  Δ_x,Δ_y ∈ [0, ε_q]
#   upper = xy + ½(Δ_x + Δ_y − Δ_p2)
# giving |z − xy| ≤ ε_q + ½ε_e.
#
# Helpers below use a half-and-half budget split:
#   ε_q ≤ τ/2  ⇒ inner Q at tolerance τ/2 over max_delta = max(Δx, Δy)
#   ε_e ≤ τ    ⇒ epigraph at tolerance τ   over max_delta = Δx + Δy
# Total worst-case: ε_q + ½ε_e ≤ τ/2 + ½·τ = τ.
#
# Restricted to one-sided-over Q. Two-sided Q (NMDT, DNMDT) or one-sided-under
# (Epigraph) make `lower > xy` reachable, breaking sandwich validity, so those
# combinations raise MethodError.

"""
    tol_depth(::Type{HybSConfig{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Inner-quad depth for HybS at target tolerance `τ`. Returns the smallest depth
whose inner-quad error on `[ax, ax+Δx]` (and `[ay, ay+Δy]`) is `≤ τ/2`, which
satisfies the `ε_q ≤ τ/2` half of the HybS budget.

Only defined for `Q ∈ {SawtoothQuadConfig, SolverSOS2QuadConfig,
ManualSOS2QuadConfig}` — the one-sided-over inner quads for which the HybS
sandwich is valid. Other Q raise `MethodError`.

See also `tol_epigraph_depth(::Type{HybSConfig{Q}}; …)` for the second knob.
"""
function tol_depth(
    ::Type{HybSConfig{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: Union{SawtoothQuadConfig, SolverSOS2QuadConfig, ManualSOS2QuadConfig}}
    return tol_depth(Q;
        tolerance = tolerance / 2,
        max_delta = max(max_delta_x, max_delta_y),
    )
end

"""
    tol_epigraph_depth(::Type{HybSConfig{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Epigraph depth for HybS at target tolerance `τ`. Returns the smallest depth
whose epigraph error on the cross-term range `Δx + Δy` is `≤ τ`, which
satisfies the `½ε_e ≤ τ/2` half of the HybS budget.

Only defined for `Q ∈ {SawtoothQuadConfig, SolverSOS2QuadConfig,
ManualSOS2QuadConfig}`. Other Q raise `MethodError`.
"""
function tol_epigraph_depth(
    ::Type{HybSConfig{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: Union{SawtoothQuadConfig, SolverSOS2QuadConfig, ManualSOS2QuadConfig}}
    return tol_depth(EpigraphQuadConfig;
        tolerance = tolerance,
        max_delta = max_delta_x + max_delta_y,
    )
end

# --- Unified HybS dispatch methods ---

"""
    _add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Approximate x·y using HybS (Hybrid Separable) relaxation with config-selected quadratic method.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _add_bilinear_approx!(
    config::HybSConfig,
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
    _add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

HybS bilinear approximation with pre-computed quadratic approximations for x² and y².

Combines Bin2 and Bin3 separable identities:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

The cross-terms (x+y)² and (x−y)² always use epigraph Q^{L1} (pure LP).

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _add_bilinear_approx!(
    config::HybSConfig,
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
    # Bounds for auxiliary variables (per-name)
    p1_bounds = [
        MinMax((
            min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max,
        )) for i in eachindex(x_bounds)
    ]
    p2_bounds = [
        MinMax((
            min = x_bounds[i].min - y_bounds[i].max,
            max = x_bounds[i].max - y_bounds[i].min,
        )) for i in eachindex(x_bounds)
    ]

    jump_model = get_jump_model(container)

    # Meta suffixes for cross-term expressions
    meta_p1 = meta * "_plus"
    meta_p2 = meta * "_diff"

    p1_expr = add_expression_container!(
        container,
        VariableSumExpression,
        C,
        names,
        time_steps;
        meta = meta_p1,
    )
    p2_expr = add_expression_container!(
        container,
        VariableDifferenceExpression,
        C,
        names,
        time_steps;
        meta = meta_p2,
    )

    for name in names, t in time_steps
        x = x_var[name, t]
        y = y_var[name, t]

        # p1 = x + y
        p1 = p1_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p1, x, 1.0)
        add_proportional_to_jump_expression!(p1, y, 1.0)

        # p2 = x − y
        p2 = p2_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p2, x, 1.0)
        add_proportional_to_jump_expression!(p2, y, -1.0)
    end

    # --- Epigraph Q^{L1} lower bound for (x+y)² and (x−y)² (no binaries) ---
    epi_cfg = EpigraphQuadConfig(; depth = config.epigraph_depth)
    zp1_expr = _add_quadratic_approx!(
        epi_cfg,
        container, C, names, time_steps,
        p1_expr, p1_bounds, meta_p1,
    )
    zp2_expr = _add_quadratic_approx!(
        epi_cfg,
        container, C, names, time_steps,
        p2_expr, p2_bounds, meta_p2,
    )

    # --- Create z variable and two-sided HybS bounds ---
    z_var = add_variable_container!(
        container,
        BilinearProductVariable,
        C,
        names,
        time_steps;
        meta,
    )
    hybrid_cons = add_constraints_container!(
        container,
        HybSBoundConstraint,
        C,
        names,
        1:2,
        time_steps;
        sparse = true,
        meta,
    )
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        xb = x_bounds[i]
        yb = y_bounds[i]
        IS.@assert_op xb.max > xb.min
        IS.@assert_op yb.max > yb.min

        # Compute valid bounds for z ≈ x·y from variable bounds
        z_lo = min(xb.min * yb.min, xb.min * yb.max, xb.max * yb.min, xb.max * yb.max)
        z_hi = max(xb.min * yb.min, xb.min * yb.max, xb.max * yb.min, xb.max * yb.max)

        z =
            z_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "HybSProduct_$(C)_{$(name), $(t)}",
                lower_bound = z_lo,
                upper_bound = z_hi,
            )

        zx = xsq[name, t]
        zy = ysq[name, t]
        zp1 = zp1_expr[name, t]
        zp2 = zp2_expr[name, t]

        # Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y)
        hybrid_cons[(name, 1, t)] = JuMP.@constraint(
            jump_model,
            z >= 0.5 * (zp1 - zx - zy),
        )
        # Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2)
        hybrid_cons[(name, 2, t)] = JuMP.@constraint(
            jump_model,
            z <= 0.5 * (zx + zy - zp2),
        )

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    # --- Standard McCormick envelope cuts on the product variable ---
    if config.add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, z_var,
            x_bounds, y_bounds, meta,
        )
    end

    return result_expr
end
