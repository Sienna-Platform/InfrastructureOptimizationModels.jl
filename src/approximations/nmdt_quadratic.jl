# NMDT (Normalized Multiparametric Disaggregation Technique) approximations
# for x². Two variants:
#   NMDTQuadConfig  — single discretization on x.
#   DNMDTQuadConfig — double NMDT: convex combination of (x discretized) and
#                     (x discretized) again, with shared β·δ residual.
#
# Both normalize x to xh ∈ [0,1], discretize via L binary digits + residual,
# linearize the binary-continuous products with McCormick envelopes, and
# reassemble x² from the normalized components. Optional epigraph Q^{L1}
# lower-bound tightening on xh².
# Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation
# technique for global optimization of polynomial programming problems.

"""
Config for single-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L.
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound;
  0 to disable (default 3·depth).
"""
struct NMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
function NMDTQuadConfig(depth::Int)
    return NMDTQuadConfig(depth, 3 * depth)
end

"""
Config for double-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L.
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound;
  0 to disable (default 3·depth).
"""
struct DNMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
function DNMDTQuadConfig(depth::Int)
    return DNMDTQuadConfig(depth, 3 * depth)
end

# --- Scalar build (pure JuMP) ---

"""
    build_quadratic_approx(config::NMDTQuadConfig, model, x, x_min, x_max)

Scalar form: approximate x² via single NMDT for one cell. Discretize xh,
build the binary-continuous product β·xh, the residual product δ·xh, and
reassemble x². When `epigraph_depth > 0`, also build an epigraph lower
bound on xh² and tighten with `x²_approx ≥ epi`.

Returns `(; approximation, discretization, bx_xh_product, residual_product,
tightening)`.
"""
function build_quadratic_approx(
    config::NMDTQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    tighten = config.epigraph_depth > 0
    disc = build_discretization(model, x, x_min, x_max, config.depth)
    bx_xh = build_binary_continuous_product(
        model, disc.beta_var, disc.norm_expr, 0.0, 1.0, config.depth; tighten,
    )
    dz = build_residual_product(
        model, disc.delta_var, disc.norm_expr, 1.0, config.depth; tighten,
    )
    approximation = build_assembled_product(
        model, [bx_xh.result_expression], dz.z_var,
        disc.norm_expr, disc.norm_expr, x_min, x_max, x_min, x_max,
    )
    tightening = if tighten
        epi = build_quadratic_approx(
            EpigraphQuadConfig(config.epigraph_depth), model, disc.norm_expr, 0.0,
            1.0,
        )
        tcon = JuMP.@constraint(model, approximation >= epi.approximation)
        (; epigraph = epi, constraint = tcon)
    else
        nothing
    end
    return (;
        approximation,
        discretization = disc,
        bx_xh_product = bx_xh,
        residual_product = dz,
        tightening,
    )
end

"""
    build_quadratic_approx(config::DNMDTQuadConfig, model, x, x_min, x_max)

Scalar form: approximate x² via double NMDT for one cell — convex
combination of two NMDT estimates with shared discretization on x.

Returns the same NamedTuple shape as `NMDTQuadConfig` plus an additional
`bx_dx_product` field for the second binary-continuous product step.
"""
function build_quadratic_approx(
    config::DNMDTQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    tighten = config.epigraph_depth > 0
    disc = build_discretization(model, x, x_min, x_max, config.depth)
    bx_xh = build_binary_continuous_product(
        model, disc.beta_var, disc.norm_expr, 0.0, 1.0, config.depth; tighten,
    )
    bx_dx = build_binary_continuous_product(
        model, disc.beta_var, disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth; tighten,
    )
    dz = build_residual_product(
        model, disc.delta_var, disc.delta_var,
        2.0^(-config.depth), config.depth; tighten,
    )
    approximation = build_assembled_dnmdt(
        model,
        bx_xh.result_expression,
        bx_dx.result_expression,
        bx_xh.result_expression,
        bx_dx.result_expression,
        dz.z_var,
        disc.norm_expr, disc.norm_expr,
        x_min, x_max, x_min, x_max,
    )
    tightening = if tighten
        epi = build_quadratic_approx(
            EpigraphQuadConfig(config.epigraph_depth), model, disc.norm_expr, 0.0,
            1.0,
        )
        tcon = JuMP.@constraint(model, approximation >= epi.approximation)
        (; epigraph = epi, constraint = tcon)
    else
        nothing
    end
    return (;
        approximation,
        discretization = disc,
        bx_xh_product = bx_xh,
        bx_dx_product = bx_dx,
        residual_product = dz,
        tightening,
    )
end

# --- IOM allocation + per-cell write helpers ---

function _alloc_nmdt_tightening_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, epi_depth::Int, meta,
) where {C <: IS.InfrastructureSystemsComponent}
    epi_targets = _alloc_epigraph_targets!(
        container, C, name_axis, time_axis, epi_depth, meta * "_epi",
    )
    tighten_cons = add_constraints_container!(
        container, NMDTTightenConstraint, C, name_axis, time_axis; meta,
    )
    return (epi = epi_targets, tighten = tighten_cons)
end

function _write_nmdt_tightening_cell!(targets, name, t, tightening, epi_depth::Int)
    _write_epigraph_cell!(targets.epi, name, t, tightening.epigraph, epi_depth)
    targets.tighten[name, t] = tightening.constraint
    return
end

"""
    add_quadratic_approx!(config::NMDTQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate discretization + binary-continuous-product + residual-product
containers, plus when tightening is enabled the epigraph + tightening
constraint containers. Loop `(name, t)`.
"""
function add_quadratic_approx!(
    config::NMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    tighten = config.epigraph_depth > 0
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
    end

    model = get_jump_model(container)

    disc_targets = _alloc_discretization_targets!(
        container, C, name_axis, time_axis, depth, meta,
    )
    bx_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta; tighten,
    )
    res_targets = _alloc_residual_product_targets!(
        container, C, name_axis, time_axis, meta; tighten,
    )
    approx_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    tighten_targets = if tighten
        _alloc_nmdt_tightening_targets!(
        container, C, name_axis, time_axis, config.epigraph_depth, meta,
    )
    else
        nothing
    end

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            _write_discretization_cell!(disc_targets, name, t, r.discretization, depth)
            _write_binary_continuous_product_cell!(
                bx_targets,
                name,
                t,
                r.bx_xh_product,
                depth,
            )
            _write_residual_product_cell!(res_targets, name, t, r.residual_product)
            approx_target[name, t] = r.approximation
            if tighten
                _write_nmdt_tightening_cell!(
                    tighten_targets, name, t, r.tightening, config.epigraph_depth,
                )
            end
        end
    end
    return approx_target
end

"""
    add_quadratic_approx!(config::DNMDTQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate two binary-continuous-product container sets
(`meta * "_bx_xh"` and `meta * "_bx_dx"`) plus the rest of the NMDT pieces.
"""
function add_quadratic_approx!(
    config::DNMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    tighten = config.epigraph_depth > 0
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
    end

    model = get_jump_model(container)

    disc_targets = _alloc_discretization_targets!(
        container, C, name_axis, time_axis, depth, meta,
    )
    bx_xh_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_bx_xh"; tighten,
    )
    bx_dx_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_bx_dx"; tighten,
    )
    res_targets = _alloc_residual_product_targets!(
        container, C, name_axis, time_axis, meta; tighten,
    )
    approx_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    tighten_targets = if tighten
        _alloc_nmdt_tightening_targets!(
        container, C, name_axis, time_axis, config.epigraph_depth, meta,
    )
    else
        nothing
    end

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            _write_discretization_cell!(disc_targets, name, t, r.discretization, depth)
            _write_binary_continuous_product_cell!(
                bx_xh_targets, name, t, r.bx_xh_product, depth,
            )
            _write_binary_continuous_product_cell!(
                bx_dx_targets, name, t, r.bx_dx_product, depth,
            )
            _write_residual_product_cell!(res_targets, name, t, r.residual_product)
            approx_target[name, t] = r.approximation
            if tighten
                _write_nmdt_tightening_cell!(
                    tighten_targets, name, t, r.tightening, config.epigraph_depth,
                )
            end
        end
    end
    return approx_target
end
