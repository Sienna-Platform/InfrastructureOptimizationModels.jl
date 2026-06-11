# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS (Hybrid Separable) bilinear approximation.

Adds two inequalities that sandwich `z ≈ x·y`:
- lower: `z ≥ ½(z_p1 − z_x − z_y)` where `z_p1 ≤ (x+y)²` is an epigraph (LP-only)
  lower bound and `z_x ≥ x²`, `z_y ≥ y²` come from the inner quadratic `Q`.
- upper: `z ≤ ½(z_x + z_y − z_p2)` where `z_p2 ≤ (x−y)²` is an epigraph (LP-only)
  lower bound on the cross-difference.

# Fields
- `quad_config::Q`: quadratic method for the shared x² and y² terms
- `cross_term_depth::Int`: depth for the epigraph approximation of the cross-terms (x±y)².
  This is a **structural** part of HybS (the LP-only lower bounds that form the sandwich),
  not an optional tightener.
- `tightener::Tightener`: optional strengthener (default `NoTightener()`). The supported
  tightener is `McCormickTightener` (standard McCormick envelope cuts on the product
  variable; `partitions`/`backend` ignored here).

`Q` must be a **strictly one-sided-over** quadratic approximation (so
`z_x ≥ x²` and `z_y ≥ y²` hold at every MIP-feasible point). With a two-sided
or one-sided-under inner Q the sandwich can become infeasible. Sketch: let
`ε_Q ≥ 0` bound `|z_• − •²|` and `ε_E ≥ 0` bound `(•)² − z_p•`. With z_x, z_y
two-sided (each can be ≤ x² by up to ε_Q) and z_p1, z_p2 at their lowest:

```
Lower − Upper = ½(z_p1 + z_p2) − z_x − z_y
              = ½((x+y)² − ε_E + (x−y)² − ε_E) − (x² − ε_Q) − (y² − ε_Q)
              = x² + y² − ε_E − x² − y² + 2ε_Q
              = 2ε_Q − ε_E
```

So `Lower > Upper` whenever `2ε_Q > ε_E` — model infeasible. This rules out
`EpigraphQuadConfig` (one-sided under, so `z_x` has no MIP-feasible upper
bound) and `NMDTQuadConfig` / `DNMDTQuadConfig` (two-sided in both regimes:
the McCormick on the δ·δ or δ·xh residual product has slack even at integer β, and with an
`EpigraphTightener` the result floats in `[epigraph(x), nmdt(x)]`). Only `SawtoothQuadConfig`
and `SOS2QuadConfig` (`sidedness(Q) == OneSidedOver()`) are supported — the constructor
enforces this. See `tolerance_depth(::Type{HybSConfig{Q}}; …)` and
`tolerance_epigraph_depth(::Type{HybSConfig{Q}}; …)`.
"""
struct HybSConfig{Q <: QuadraticApproxConfig} <: SeparableConfig
    quad_config::Q
    cross_term_depth::Int
    tightener::Tightener

    function HybSConfig(
        quad_config::Q;
        cross_term_depth::Int,
        tightener::Tightener = NoTightener(),
    ) where {Q <: QuadraticApproxConfig}
        sidedness(Q) isa OneSidedOver || throw(
            ArgumentError(
                "HybSConfig requires a one-sided-over inner Q; got $(Q) with sidedness " *
                "$(sidedness(Q)). Only SawtoothQuadConfig and SOS2QuadConfig qualify.",
            ),
        )
        supports_tightener(HybSConfig, tightener) || throw(
            ArgumentError("HybSConfig does not support tightener $(typeof(tightener))"),
        )
        return new{Q}(quad_config, cross_term_depth, tightener)
    end
end

"HybS supports standard McCormick envelope cuts (`McCormickTightener`) on the product."
supports_tightener(::Type{<:HybSConfig}, ::McCormickTightener) = true

# --- Tolerance helpers ---
#
# Notation: Δx, Δy are domain lengths; all ε's are ≥ 0. Let
#   ε_x = z_x − x²,  ε_y = z_y − y²       inner-quad over-errors
#   ε_p1 = (x+y)² − z_p1,  ε_p2 = (x−y)² − z_p2   epigraph under-errors
# and let ε_Q, ε_E be the worst-case bounds:
#   ε_x, ε_y ≤ ε_Q   (inner Q at depth L on its domain)
#   ε_p1, ε_p2 ≤ ε_E (epigraph at depth L_e on its domain)
#
# Substitute z_x = x² + ε_x, z_y = y² + ε_y, z_p1 = (x+y)² − ε_p1 into the
# Bin2 lower-bound expression:
#   ½(z_p1 − z_x − z_y)
#     = ½((x+y)² − ε_p1 − x² − ε_x − y² − ε_y)
#     = ½(2xy − ε_p1 − ε_x − ε_y)
#     = xy − ½(ε_p1 + ε_x + ε_y)
# Since ε_p1 ≤ ε_E and ε_x, ε_y ≤ ε_Q, the gap is at most ½ε_E + ε_Q below xy.
# Similarly the Bin3 upper-bound expression ½(z_x + z_y − z_p2) simplifies to
# xy + ½(ε_p2 + ε_x + ε_y), at most ½ε_E + ε_Q above xy. Combining,
#   |z − xy| ≤ ½ε_E + ε_Q.
#
# To meet user tolerance τ, the helpers below allocate ε_E ≤ τ and ε_Q ≤ τ/2,
# giving ½τ + τ/2 = τ. This allocation is arbitrary — any pair satisfying
# ½ε_E + ε_Q ≤ τ would work; the chosen split is just one valid point.
#
# Restricted to one-sided-over Q (see struct docstring for the asymmetry
# argument). Other Q raise MethodError.

"""
    tolerance_depth(::Type{HybSConfig{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Inner-quad depth for HybS at target tolerance `τ`. Returns the smallest depth
whose inner-quad error on `[ax, ax+Δx]` (and `[ay, ay+Δy]`) is `≤ τ/2`, which
satisfies the `ε_Q ≤ τ/2` half of the HybS budget.

Defined only for strictly one-sided-over inner quads:
`SawtoothQuadConfig`, `SOS2QuadConfig`. Other Q
raise `MethodError`. See the `HybSConfig` docstring for why two-sided or
one-sided-under inner Qs (`EpigraphQuadConfig`, `NMDTQuadConfig`,
`DNMDTQuadConfig`) make the sandwich infeasible.

See also `tolerance_epigraph_depth(::Type{HybSConfig{Q}}; …)` for the second knob.
"""
function tolerance_depth(
    ::Type{HybSConfig{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: QuadraticApproxConfig}
    sidedness(Q) isa OneSidedOver || throw(
        ArgumentError(
            "HybSConfig requires a one-sided-over inner Q; got $(Q) with sidedness " *
            "$(sidedness(Q)).",
        ),
    )
    return tolerance_depth(Q;
        tolerance = tolerance / 2,
        max_delta = max(max_delta_x, max_delta_y),
    )
end

"""
    tolerance_epigraph_depth(::Type{HybSConfig{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Epigraph depth for HybS at target tolerance `τ`. Returns the smallest depth
whose epigraph error on the cross-term range `Δx + Δy` is `≤ τ`, which
satisfies the `½ε_E ≤ τ/2` half of the HybS budget.

Defined only for strictly one-sided-over inner quads:
`SawtoothQuadConfig`, `SOS2QuadConfig`. Other Q
raise `MethodError`.
"""
function tolerance_epigraph_depth(
    ::Type{HybSConfig{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: QuadraticApproxConfig}
    sidedness(Q) isa OneSidedOver || throw(
        ArgumentError(
            "HybSConfig requires a one-sided-over inner Q; got $(Q) with sidedness " *
            "$(sidedness(Q)).",
        ),
    )
    return tolerance_depth(EpigraphQuadConfig;
        tolerance = tolerance,
        max_delta = max_delta_x + max_delta_y,
    )
end

# --- Unified HybS dispatch methods ---

"""
    add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Approximate x·y using HybS (Hybrid Separable) relaxation with config-selected quadratic method.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function add_bilinear_approx!(
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
    xsq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        x_var, x_bounds, meta * "_x",
    )
    ysq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        y_var, y_bounds, meta * "_y",
    )
    return _assemble_hybs!(
        config, container, C, names, time_steps,
        xsq, ysq, x_var, y_var,
        x_bounds, y_bounds, meta,
    )
end

"""
    _assemble_hybs!(config::HybSConfig, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

HybS bilinear approximation with pre-computed quadratic approximations for x² and y².

Combines Bin2 and Bin3 separable identities:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

The cross-terms (x+y)² and (x−y)² always use epigraph Q^{L1} (pure LP).

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _assemble_hybs!(
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
    epi_cfg = EpigraphQuadConfig(; depth = config.cross_term_depth)
    zp1_expr = add_quadratic_approx!(
        epi_cfg,
        container, C, names, time_steps,
        p1_expr, p1_bounds, meta_p1,
    )
    zp2_expr = add_quadratic_approx!(
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
    if config.tightener isa McCormickTightener
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, z_var,
            x_bounds, y_bounds, meta,
        )
    end

    return result_expr
end
