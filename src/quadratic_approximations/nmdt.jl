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
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 3×depth)
"""
struct DNMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
DNMDTQuadConfig(depth::Int) = DNMDTQuadConfig(depth, 3 * depth)

"""
Config for single-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 3×depth)
"""
struct NMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
NMDTQuadConfig(depth::Int) = NMDTQuadConfig(depth, 3 * depth)

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
