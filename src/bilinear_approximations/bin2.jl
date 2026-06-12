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
    add_bilinear_approx!(config, container, ::Type{C}, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Approximate the bilinear product `x·y` for each `(name, t)`, storing the result in an
expression container and returning it. This is the uniform-arity interface every
`BilinearApproxConfig` implements; methods differ only by the config type. Internal staged
builders that take precomputed inputs (e.g. precomputed `xsq`/`ysq` or an `NMDTDiscretization`)
use distinct names and are not part of this interface.
"""
function add_bilinear_approx! end

# Fallback: a config without a concrete `add_bilinear_approx!` method lands here.
add_bilinear_approx!(config::BilinearApproxConfig, args...) = error(
    "add_bilinear_approx! is not implemented for $(typeof(config)); required signature: " *
    "(config, container, ::Type{C}, names::Vector{String}, time_steps::UnitRange{Int}, " *
    "x_var, y_var, x_bounds::Vector{MinMax}, y_bounds::Vector{MinMax}, meta::String)",
)

# Bilinear tightener support (mirrors the quadratic fallbacks in common.jl).
supports_tightener(::Type{<:BilinearApproxConfig}, ::Tightener) = false
supports_tightener(::Type{<:BilinearApproxConfig}, ::NoTightener) = true

"""
Abstract supertype for separable bilinear methods that reduce `x·y` to a combination of
squares evaluated with an inner quadratic config `Q` — `Bin2Config` and `HybSConfig`.
"""
abstract type SeparableConfig <: BilinearApproxConfig end

"""
Config for Bin2 bilinear approximation using z = ½((x+y)² − x² − y²).

# Fields
- `quad_config::Q`: quadratic method used for x², y², and (x+y)²
- `tightener::Tightener`: optional strengthener (default `McCormickTightener()`). The supported
  tightener is `McCormickTightener` (standard reformulated McCormick cuts through the separable
  variables; `partitions`/`backend` are ignored here). Use `NoTightener()` to disable.

The `Q` type parameter lets `tolerance_depth` dispatch on the inner quad's `sidedness`;
see `tolerance_depth(::Type{Bin2Config{Q}}; …)`.
"""
struct Bin2Config{Q <: QuadraticApproxConfig} <: SeparableConfig
    quad_config::Q
    tightener::Tightener

    function Bin2Config(
        quad_config::Q;
        tightener::Tightener = McCormickTightener(),
    ) where {Q <: QuadraticApproxConfig}
        supports_tightener(Bin2Config, tightener) || throw(
            ArgumentError("Bin2Config does not support tightener $(typeof(tightener))"),
        )
        return new{Q}(quad_config, tightener)
    end
end

"Bin2 supports standard reformulated McCormick cuts (`McCormickTightener`)."
supports_tightener(::Type{<:Bin2Config}, ::McCormickTightener) = true

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
# Δ²·ε_q at depth L (ε_q is the per-unit-domain error magnitude — see
# `tolerance_depth(::Type{<:QuadraticApproxConfig})` for why Δ² appears), so
#   |ε_x| ≤ ε_x^max = Δx²·ε_q,
#   |ε_y| ≤ ε_y^max = Δy²·ε_q,
#   |ε_p| ≤ ε_p^max = (Δx+Δy)²·ε_q.
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
# straddles x²): each ε_• ∈ [−ε_•^max, ε_•^max]. Step-by-step:
#
#   |z − xy| = ½|ε_p − ε_x − ε_y|
#           ≤ ½(|ε_p| + |ε_x| + |ε_y|)              (triangle inequality)
#           ≤ ½(ε_p^max + ε_x^max + ε_y^max)        (worst-case per term)
#           = ½((Δx+Δy)² + Δx² + Δy²)·ε_q           (substitute ε_•^max definitions)
#           = ½(Δx² + 2·Δx·Δy + Δy² + Δx² + Δy²)·ε_q (expand (Δx+Δy)²)
#           = (Δx² + Δx·Δy + Δy²)·ε_q.              (collect terms, divide by 2)
#
# To hit user tolerance τ, require (Δx² + Δx·Δy + Δy²)·ε_q ≤ τ, i.e.
# ε_q ≤ τ / (Δx² + Δx·Δy + Δy²). The inner-Q `tolerance_depth` call takes a
# `tolerance` argument τ_inner and `max_delta` argument Δ_inner, and the
# returned depth guarantees ε_q ≤ τ_inner / Δ_inner². The inner Q here is
# called with Δ_inner = Δx+Δy (the domain of x+y), so picking
#
#   τ_inner = (Δx+Δy)² · τ / (Δx² + Δx·Δy + Δy²)
#
# yields ε_q ≤ τ_inner / (Δx+Δy)² = τ / (Δx² + Δx·Δy + Δy²), as required.

"""
    tolerance_depth(::Type{Bin2Config{Q}}; tolerance, max_delta_x, max_delta_y)::Int

Inner-quad depth such that Bin2's worst-case gap `|z − xy|` is ≤ `tolerance`.
Derivation: see the comment block above.

Dispatches on `sidedness(Q)`:
- **one-sided-over** (`SawtoothQuadConfig`, `SOS2QuadConfig`): forwards to
  `tolerance_depth(Q; tolerance = 2·τ, max_delta = Δx + Δy)`.
- **two-sided** (`NMDTQuadConfig`): forwards to
  `tolerance_depth(Q; tolerance = (Δx+Δy)²·τ/(Δx² + Δy² + Δx·Δy), max_delta = Δx + Δy)`.
- **one-sided-under** (`EpigraphQuadConfig`): throws — no finite bound exists (see below).

`EpigraphQuadConfig` is excluded. The epigraph is one-sided-under
(z_• ≤ •²), so with ε_x = x² − z_x ≥ 0 and ε_y = y² − z_y ≥ 0 (unbounded
above in the LP relaxation, since the LP imposes no lower-side gap on z_x,
z_y) and ε_p = (x+y)² − z_p ≥ 0:

```
z − xy = ½(z_p − z_x − z_y) − xy
      = ½((x+y)² − ε_p − (x² − ε_x) − (y² − ε_y)) − xy
      = ½(2xy − ε_p + ε_x + ε_y) − xy
      = ½(ε_x + ε_y − ε_p).
```

In the LP relaxation ε_x and ε_y have no finite upper bound (an LP solver
can drive z_x and z_y as low as the variable bounds allow), so `z − xy`
can be made arbitrarily large. No finite `tolerance_depth` recovers a
bound. This rules out epigraph as an inner Q for Bin2.
"""
function tolerance_depth(
    ::Type{Bin2Config{Q}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: QuadraticApproxConfig}
    return _bin2_tolerance_depth(sidedness(Q), Q; tolerance, max_delta_x, max_delta_y)
end

# Bin2 tolerance depth, dispatched on the inner quad's sidedness trait (see the derivation block
# above). One-sided-over and two-sided have finite bounds; other sidednesses do not.
function _bin2_tolerance_depth(
    ::OneSidedOver,
    ::Type{Q};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: QuadraticApproxConfig}
    return tolerance_depth(
        Q;
        tolerance = 2 * tolerance,
        max_delta = max_delta_x + max_delta_y,
    )
end

function _bin2_tolerance_depth(
    ::TwoSided,
    ::Type{Q};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
) where {Q <: QuadraticApproxConfig}
    max_delta = max_delta_x + max_delta_y
    sum_sq = max_delta_x^2 + max_delta_y^2 + max_delta_x * max_delta_y
    return tolerance_depth(Q;
        tolerance = max_delta^2 * tolerance / sum_sq,
        max_delta = max_delta,
    )
end

_bin2_tolerance_depth(
    s::ApproxSidedness,
    ::Type{Q};
    kwargs...,
) where {Q <: QuadraticApproxConfig} = throw(
    ArgumentError(
        "Bin2Config requires a one-sided-over or two-sided inner Q; got $(Q) " *
        "with sidedness $(s), which has no finite Bin2 tolerance bound.",
    ),
)

# --- Unified bilinear approximation dispatch ---

"""
    add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Standard form: compute x² and y² quadratic approximations, then delegate to `_assemble_separable!`.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function add_bilinear_approx!(
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
    xsq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        x_var, x_bounds, meta * "_x",
    )
    ysq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        y_var, y_bounds, meta * "_y",
    )
    return _assemble_separable!(
        config, container, C, names, time_steps,
        xsq, ysq, x_var, y_var,
        x_bounds, y_bounds, meta,
    )
end

"""
    _assemble_separable!(config::Bin2Config, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed form: Bin2 identity z = ½((x+y)² − x² − y²) with optional reformulated McCormick cuts.
Accepts pre-computed quadratic approximations `xsq` ≈ x² and `ysq` ≈ y².

`_assemble_separable!` is the shared staged-assembly interface for `SeparableConfig` methods (Bin2
and HybS); callers dispatch on the config type instead of branching on it.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _assemble_separable!(
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
    jump_model = get_jump_model(container)
    for name in names, t in time_steps
        p_expr[name, t] = JuMP.@expression(jump_model, x_var[name, t] + y_var[name, t])
    end

    # Approximate p² = (x+y)² using the provided quadratic config
    psq = add_quadratic_approx!(
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
        result_expr[name, t] = JuMP.@expression(
            jump_model,
            0.5 * psq[name, t] - 0.5 * xsq[name, t] - 0.5 * ysq[name, t],
        )
    end

    # --- Reformulated McCormick cuts (optional, via tightener dispatch) ---
    apply_tightener!(
        config.tightener, config, container, C, names, time_steps,
        x_var, y_var, psq, xsq, ysq, x_bounds, y_bounds, meta,
    )

    return result_expr
end

"Apply reformulated McCormick cuts on the Bin2 separable variables (valid inequality)."
function apply_tightener!(
    ::McCormickTightener,
    ::Bin2Config,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    psq,
    xsq,
    ysq,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    _add_reformulated_mccormick!(
        container, C, names, time_steps,
        x_var, y_var, psq, xsq, ysq,
        x_bounds, y_bounds, meta,
    )
    return
end
