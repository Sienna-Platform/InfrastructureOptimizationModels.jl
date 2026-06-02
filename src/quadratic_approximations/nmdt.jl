# NMDT (Normalized Multiparametric Disaggregation Technique) quadratic approximation of x².
# Normalizes x to [0,1], discretizes using L binary variables β₁,…,β_L plus a
# residual δ ∈ [0, 2^{−L}], then replaces each binary-continuous product β_i·xh
# with a McCormick-linearized auxiliary variable. Assembles the result via the
# separable identity x² = (lx·xh + x_min)². Optionally tightens with an epigraph
# lower bound on xh².
# NMDT Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation
# technique for global optimization of polynomial programming problems.

"""
Config for double-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L
- `epigraph_depth::Int`: depth of an additional epigraph Q^{L1} lower bound;
  0 disables (default 3×depth)

The DNMDT side gives worst-case `|result_expr − x²| ≤ Δ²·2^{-2L-2}` (two-sided
magnitude bound). The binary discretization `x = x_grid + δ` is exact in MIP
(binary×binary and binary×continuous McCormick are tight at integer β), but
the residual `δ²` term is approximated by a McCormick envelope on (δ, δ) over
`[0, 2^{-L}]²` — and continuous×continuous McCormick has slack at interior δ
even at integer β. So `result_expr` floats in an interval around `x²` whose
half-width is at most `Δ²·2^{-2L-2}` (max gap at `δ = 2^{-L-1}`).

When `epigraph_depth = L_e > 0`, the McCormick lower bounds on the
binary–continuous products are dropped and a lower bound `result_expr ≥
epigraph(x)` is added in their place. The result remains two-sided around
`x²`, but the lower envelope is now the global epigraph instead of the
per-product McCormick LBs; worst case becomes
`max(Δ²·2^{-2L-2}, Δ²·2^{-2L_e-4})`. Contrast with `pwmcc_segments` on the
SOS2 variants, which adds genuine LP cuts and never changes the MIP-feasible set.

See `tolerance_depth(::Type{DNMDTQuadConfig}; …)` to derive `depth` from a
target tolerance, and `tolerance_epigraph_depth(::Type{DNMDTQuadConfig}; …)`
for the matching `epigraph_depth`.
"""
struct DNMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int

    DNMDTQuadConfig(; depth::Int, epigraph_depth::Int = 3 * depth) =
        new(depth, epigraph_depth)
end

"""
    tolerance_depth(::Type{DNMDTQuadConfig}; tolerance, max_delta)::Int

Smallest DNMDT depth `L` whose worst-case overestimation gap on `[a, a+Δ]`
falls within `tolerance`. Inverts `Δ²·2^{-2L-2} ≤ τ`:
```
L = ⌈(log₂(Δ²/τ) − 2) / 2⌉
```
clamped to `L ≥ 1`. Sizes only the DNMDT side.

**Contract on `epigraph_depth`**: the returned depth meets the tolerance iff
the user picks `epigraph_depth = 0` (tightening disabled) or
`epigraph_depth ≥ depth − 1`. The DNMDT side error is `Δ²·2^{-2L-2}` and
the epigraph side error is `Δ²·2^{-2L_e-4}`; epigraph ≤ DNMDT iff
`L_e ≥ L − 1`. When `0 < epigraph_depth < depth − 1`, the epigraph side
has a larger error than the DNMDT side and the realized error can exceed
`tolerance`. Use `tolerance_epigraph_depth` to size both knobs consistently.
"""
function tolerance_depth(
    ::Type{DNMDTQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive((log2(max_delta^2 / tolerance) - 2) / 2)
end

"""
    tolerance_epigraph_depth(::Type{DNMDTQuadConfig}; tolerance, max_delta)::Int

Smallest `epigraph_depth` whose epigraph error on `[a, a+Δ]` is `≤ tolerance`.
Pass alongside `tolerance_depth(DNMDTQuadConfig; …)` to honor the contract.
"""
function tolerance_epigraph_depth(
    ::Type{DNMDTQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    return tolerance_depth(EpigraphQuadConfig; tolerance, max_delta)
end

"""
Config for single-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L
- `epigraph_depth::Int`: depth of an additional epigraph Q^{L1} lower bound;
  0 disables (default 3×depth)

The NMDT side gives worst-case `|result_expr − x²| ≤ Δ²·2^{-L-2}` two-sided
magnitude bound (note the single `L`, not `2L` — single NMDT discretizes only
one factor). The binary discretization `x = x_grid + δ` is exact in MIP, but
the cross term `δ·xh` (where `xh` is the full normalized x) is approximated by
McCormick on (δ, xh) — both continuous — and that has slack at interior values
even at integer β. So `result_expr` floats in a two-sided interval around `x²`.

When `epigraph_depth = L_e > 0`, the McCormick lower bounds on the
binary–continuous products are dropped and a lower bound `result_expr ≥
epigraph(x)` is added in their place. The result remains two-sided around
`x²`, but the lower envelope is now the global epigraph instead of the
per-product McCormick LBs; worst case becomes
`max(Δ²·2^{-L-2}, Δ²·2^{-2L_e-4})`. Contrast with `pwmcc_segments` on the
SOS2 variants, which adds genuine LP cuts and never changes the MIP-feasible set.

See `tolerance_depth(::Type{NMDTQuadConfig}; …)` to derive `depth` from a
target tolerance, and `tolerance_epigraph_depth(::Type{NMDTQuadConfig}; …)`
for the matching `epigraph_depth`.
"""
struct NMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int

    NMDTQuadConfig(; depth::Int, epigraph_depth::Int = 3 * depth) =
        new(depth, epigraph_depth)
end

"""
    tolerance_depth(::Type{NMDTQuadConfig}; tolerance, max_delta)::Int

Smallest NMDT depth `L` whose worst-case overestimation gap on `[a, a+Δ]`
falls within `tolerance`. Inverts `Δ²·2^{-L-2} ≤ τ`:
```
L = ⌈log₂(Δ²/τ) − 2⌉
```
clamped to `L ≥ 1`. Sizes only the NMDT side.

**Contract on `epigraph_depth`**: the returned depth meets the tolerance iff
the user picks `epigraph_depth = 0` (tightening disabled) or
`epigraph_depth ≥ ⌈(depth − 2) / 2⌉`. The NMDT side error is `Δ²·2^{-L-2}`
and the epigraph side error is `Δ²·2^{-2L_e-4}`; epigraph ≤ NMDT iff
`2L_e + 4 ≥ L + 2`, i.e. `L_e ≥ (L − 2)/2`. When the user picks a smaller
`epigraph_depth`, the epigraph side has a larger error than the NMDT side
and the realized error can exceed `tolerance`. Use
`tolerance_epigraph_depth` to size both knobs consistently.
"""
function tolerance_depth(
    ::Type{NMDTQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive(log2(max_delta^2 / tolerance) - 2)
end

"""
    tolerance_epigraph_depth(::Type{NMDTQuadConfig}; tolerance, max_delta)::Int

Smallest `epigraph_depth` whose epigraph error on `[a, a+Δ]` is `≤ tolerance`.
Pass alongside `tolerance_depth(NMDTQuadConfig; …)` to honor the contract.
"""
function tolerance_epigraph_depth(
    ::Type{NMDTQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    return tolerance_depth(EpigraphQuadConfig; tolerance, max_delta)
end

"""
    _add_quadratic_approx!(config::DNMDTQuadConfig, container, C, names, time_steps, x_disc, bounds, meta)

Approximate x² using the Double NMDT (DNMDT) method from a pre-built discretization.

Constructs two binary-continuous products (β·xh and β·δ) and delegates to the core
DNMDT assembler, storing results in a `QuadraticExpression` container. Optionally
tightens lower bounds with an epigraph relaxation via `_tighten_lower_bounds!`.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` (binary discretization levels) and `epigraph_depth` (LP tightening depth; 0 to disable, default 3×depth)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_quadratic_approx!(
    config::DNMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    tighten = config.epigraph_depth > 0
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

    if config.epigraph_depth > 0
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, config.epigraph_depth, meta,
        )
    end

    return result_expr
end

"""
    _add_quadratic_approx!(config::DNMDTQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using the Double NMDT (DNMDT) method from raw variable inputs.

Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
Stores results in a `QuadraticExpression` container.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` (binary discretization levels) and `epigraph_depth` (LP tightening depth; 0 to disable, default 3×depth)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_quadratic_approx!(
    config::DNMDTQuadConfig,
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

    return _add_quadratic_approx!(
        config, container, C, names, time_steps,
        x_disc, bounds, meta,
    )
end

"""
    _add_quadratic_approx!(config::NMDTQuadConfig, container, C, names, time_steps, x_disc, bounds, meta)

Approximate x² using the NMDT method from a pre-built discretization.

Computes the binary-continuous product β·xh and residual product δ·xh, then
assembles x² via `_assemble_product!`. Stores results in a `QuadraticExpression`
container. Optionally tightens lower bounds with an epigraph relaxation.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` (binary discretization levels) and `epigraph_depth` (LP tightening depth; 0 to disable, default 3×depth)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_quadratic_approx!(
    config::NMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    tighten = config.epigraph_depth > 0
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

    if config.epigraph_depth > 0
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, config.epigraph_depth, meta,
        )
    end

    return result_expr
end

"""
    _add_quadratic_approx!(config::NMDTQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using the NMDT method from raw variable inputs.

Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
Stores results in a `QuadraticExpression` container.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` (binary discretization levels) and `epigraph_depth` (LP tightening depth; 0 to disable, default 3×depth)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_quadratic_approx!(
    config::NMDTQuadConfig,
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

    return _add_quadratic_approx!(
        config, container, C, names, time_steps,
        x_disc, bounds, meta,
    )
end
