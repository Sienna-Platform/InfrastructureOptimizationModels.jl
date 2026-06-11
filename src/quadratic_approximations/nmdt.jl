# NMDT (Normalized Multiparametric Disaggregation Technique) quadratic approximation of x².
# Normalizes x to [0,1], discretizes using L binary variables β₁,…,β_L plus a
# residual δ ∈ [0, 2^{−L}], then replaces each binary-continuous product β_i·xh
# with a McCormick-linearized auxiliary variable. Assembles the result via the
# separable identity x² = (lx·xh + x_min)². Optionally tightens with an epigraph
# lower bound on xh².
#
# A single parametric config `NMDTQuadConfig{V}` covers both variants:
#   - `SingleNMDT`: discretizes one factor (cross term δ·xh), error `Δ²·2^{-L-2}`.
#   - `DoubleNMDT`: discretizes both factors (DNMDT), error `Δ²·2^{-2L-2}`.
# NMDT Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation
# technique for global optimization of polynomial programming problems.

"""
Config for NMDT quadratic approximation, parameterized by the discretization variant
`V <: NMDTVariant` (`SingleNMDT` or `DoubleNMDT`).

# Fields
- `depth::Int`: number of binary discretization levels L
- `tightener::Tightener`: optional strengthener (default `EpigraphTightener(3·depth)`). The
  supported tightener is `EpigraphTightener(L_e)`, adding an epigraph Q^{L1} lower bound on
  xh²; use `NoTightener()` to disable.

`DoubleNMDT` gives worst-case `|result_expr − x²| ≤ Δ²·2^{-2L-2}`; `SingleNMDT` gives
`Δ²·2^{-L-2}` (single `L`, since only one factor is discretized). Both are two-sided
magnitude bounds: the binary discretization `x = x_grid + δ` is exact in MIP, but the
residual product (continuous×continuous McCormick) has slack at interior points, so
`result_expr` floats around `x²`.

With an `EpigraphTightener(L_e)`, the McCormick lower bounds on the binary–continuous
products are dropped and a lower bound `result_expr ≥ epigraph(x)` is added in their place;
the worst case becomes `max(<variant bound>, Δ²·2^{-2L_e-4})`. Contrast with a
`McCormickTightener` on the SOS2 config, which adds genuine LP cuts and never changes the
MIP-feasible set.

See `tolerance_depth(::Type{NMDTQuadConfig{V}}; …)` to derive `depth` from a target
tolerance, and `tolerance_epigraph_depth(::Type{<:NMDTQuadConfig}; …)` for the matching
`EpigraphTightener` depth.
"""
struct NMDTQuadConfig{V <: NMDTVariant} <: QuadraticApproxConfig
    depth::Int
    tightener::Tightener

    function NMDTQuadConfig{V}(;
        depth::Int,
        tightener::Tightener = EpigraphTightener(3 * depth),
    ) where {V <: NMDTVariant}
        supports_tightener(NMDTQuadConfig{V}, tightener) || throw(
            ArgumentError("NMDTQuadConfig does not support tightener $(typeof(tightener))"),
        )
        return new{V}(depth, tightener)
    end
end

"NMDT is two-sided around x² (continuous×continuous McCormick slack on the residual)."
sidedness(::Type{<:NMDTQuadConfig}) = TwoSided()

"NMDT supports an `EpigraphTightener` lower bound."
supports_tightener(::Type{<:NMDTQuadConfig}, ::EpigraphTightener) = true

"""
    tolerance_depth(::Type{NMDTQuadConfig{DoubleNMDT}}; tolerance, max_delta)::Int

Smallest DNMDT depth `L` whose worst-case gap on `[a, a+Δ]` falls within `tolerance`.
Inverts `Δ²·2^{-2L-2} ≤ τ`: `L = ⌈(log₂(Δ²/τ) − 2) / 2⌉`, clamped to `L ≥ 1`. Sizes only
the DNMDT side; pair with `tolerance_epigraph_depth` to size the `EpigraphTightener`.
"""
function tolerance_depth(
    ::Type{NMDTQuadConfig{DoubleNMDT}};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive((log2(max_delta^2 / tolerance) - 2) / 2)
end

"""
    tolerance_depth(::Type{NMDTQuadConfig{SingleNMDT}}; tolerance, max_delta)::Int

Smallest single-NMDT depth `L` whose worst-case gap on `[a, a+Δ]` falls within `tolerance`.
Inverts `Δ²·2^{-L-2} ≤ τ`: `L = ⌈log₂(Δ²/τ) − 2⌉`, clamped to `L ≥ 1`. Sizes only the NMDT
side; pair with `tolerance_epigraph_depth` to size the `EpigraphTightener`.
"""
function tolerance_depth(
    ::Type{NMDTQuadConfig{SingleNMDT}};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive(log2(max_delta^2 / tolerance) - 2)
end

"""
    tolerance_epigraph_depth(::Type{<:NMDTQuadConfig}; tolerance, max_delta)::Int

Smallest `EpigraphTightener` depth whose epigraph error on `[a, a+Δ]` is `≤ tolerance`.
Pass `tightener = EpigraphTightener(tolerance_epigraph_depth(...))` alongside
`tolerance_depth(NMDTQuadConfig{V}; …)` to honor the tolerance contract.
"""
function tolerance_epigraph_depth(
    ::Type{<:NMDTQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    return tolerance_depth(EpigraphQuadConfig; tolerance, max_delta)
end

# --- DoubleNMDT (DNMDT) ---

"""
    _quadratic_from_discretization!(config::NMDTQuadConfig{DoubleNMDT}, container, C, names, time_steps, x_disc, bounds, meta)

Approximate x² using the Double NMDT (DNMDT) method from a pre-built discretization.

Constructs two binary-continuous products (β·xh and β·δ) and delegates to the core
DNMDT assembler, storing results in a `QuadraticExpression` container. With an
`EpigraphTightener`, tightens lower bounds via `_tighten_lower_bounds!`.
"""
function _quadratic_from_discretization!(
    config::NMDTQuadConfig{DoubleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    tighten = config.tightener isa EpigraphTightener
    bx_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 0.0, 1.0,
        config.depth, meta * "_bx_xh"; tighten,
    )
    bx_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.delta_var, 0.0, 2.0^(-config.depth),
        config.depth, meta * "_bx_dx"; tighten,
    )

    result_expr = _assemble_dnmdt!(
        container, C, names, time_steps,
        bx_xh_expr, bx_dx_expr, bx_xh_expr, bx_dx_expr,
        x_disc, x_disc, bounds, bounds,
        config.depth, meta; tighten,
        result_type = QuadraticExpression,
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, bounds, config.tightener.depth, meta,
        )
    end

    return result_expr
end

"""
    add_quadratic_approx!(config::NMDTQuadConfig{DoubleNMDT}, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using the Double NMDT (DNMDT) method from raw variable inputs.
Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
"""
function add_quadratic_approx!(
    config::NMDTQuadConfig{DoubleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, bounds, config.depth, meta,
    )

    return _quadratic_from_discretization!(
        config, container, C, names, time_steps,
        x_disc, bounds, meta,
    )
end

# --- SingleNMDT ---

"""
    _quadratic_from_discretization!(config::NMDTQuadConfig{SingleNMDT}, container, C, names, time_steps, x_disc, bounds, meta)

Approximate x² using the single-NMDT method from a pre-built discretization.

Computes the binary-continuous product β·xh and residual product δ·xh, then assembles x²
via `_assemble_product!`. With an `EpigraphTightener`, tightens lower bounds.
"""
function _quadratic_from_discretization!(
    config::NMDTQuadConfig{SingleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    tighten = config.tightener isa EpigraphTightener
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 0.0, 1.0,
        config.depth, meta; tighten,
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 1.0,
        config.depth, meta; tighten,
    )

    result_expr = _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, x_disc, bounds, bounds,
        meta; result_type = QuadraticExpression,
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, bounds, config.tightener.depth, meta,
        )
    end

    return result_expr
end

"""
    add_quadratic_approx!(config::NMDTQuadConfig{SingleNMDT}, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using the single-NMDT method from raw variable inputs.
Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
"""
function add_quadratic_approx!(
    config::NMDTQuadConfig{SingleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, bounds, config.depth, meta,
    )

    return _quadratic_from_discretization!(
        config, container, C, names, time_steps,
        x_disc, bounds, meta,
    )
end
