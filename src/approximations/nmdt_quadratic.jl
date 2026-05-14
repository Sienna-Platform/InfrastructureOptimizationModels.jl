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

# --- Shared epigraph tightening helper ---

"""
Result of an epigraph tightening step on an NMDT quadratic approximation.
"""
struct NMDTEpigraphTightening{
    EPI <: EpigraphQuadResult,
    CONS <: JuMP.Containers.DenseAxisArray,
}
    epigraph::EPI
    constraints::CONS
end

function _build_nmdt_tightening(
    model::JuMP.Model,
    approximation,
    x_disc::NMDTDiscretization,
    epigraph_depth::Int,
)
    name_axis = axes(approximation, 1)
    time_axis = axes(approximation, 2)
    fake_bounds = fill((min = 0.0, max = 1.0), length(name_axis))
    epi = build_quadratic_approx(
        EpigraphQuadConfig(epigraph_depth), model, x_disc.norm_expr, fake_bounds,
    )
    cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        approximation[name, t] >= epi.approximation[name, t]
    )
    return NMDTEpigraphTightening(epi, cons)
end

function _register_tightening!(
    container::OptimizationContainer,
    ::Type{C},
    t::NMDTEpigraphTightening,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_in_container!(container, C, t.epigraph, meta * "_epi")
    name_axis = axes(t.constraints, 1)
    time_axis = axes(t.constraints, 2)
    target = add_constraints_container!(
        container, NMDTTightenConstraint, C, name_axis, time_axis; meta,
    )
    target.data .= t.constraints.data
    return
end

# No-op when tightening is disabled (config.epigraph_depth = 0).
_register_tightening!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing

# --- NMDT (single) ---

"""
Pure-JuMP result of `build_quadratic_approx(::NMDTQuadConfig, ...)`.
"""
struct NMDTQuadResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    D <: NMDTDiscretization,
    BX <: NMDTBinaryContinuousProduct,
    DZ <: NMDTResidualProduct,
    T <: Union{Nothing, NMDTEpigraphTightening},
} <: QuadraticApproxResult
    approximation::A
    discretization::D
    bx_xh_product::BX
    residual_product::DZ
    tightening::T
end

"""
    build_quadratic_approx(config::NMDTQuadConfig, model, x, bounds)

Approximate x² using single NMDT: discretize xh, build the binary-continuous
product Σ 2^{−i}·u_i ≈ β·xh and the residual product z ≈ δ·xh, then
reassemble x² from these normalized components.
"""
function build_quadratic_approx(
    config::NMDTQuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    tighten = config.epigraph_depth > 0
    x_disc = build_discretization(model, x, bounds, config.depth)
    bx_xh = build_binary_continuous_product(
        model, x_disc.beta_var, x_disc.norm_expr, 0.0, 1.0, config.depth; tighten,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, x_disc.norm_expr, 1.0, config.depth; tighten,
    )
    approximation = build_assembled_product(
        model,
        [bx_xh.result_expression],
        dz.z_var,
        x_disc.norm_expr,
        x_disc.norm_expr,
        bounds,
        bounds,
    )
    tightening = if tighten
        _build_nmdt_tightening(model, approximation, x_disc, config.epigraph_depth)
    else
        nothing
    end
    return NMDTQuadResult(approximation, x_disc, bx_xh, dz, tightening)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::NMDTQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_discretization!(container, C, result.discretization, meta)
    register_binary_continuous_product!(container, C, result.bx_xh_product, meta)
    register_residual_product!(container, C, result.residual_product, meta)

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    _register_tightening!(container, C, result.tightening, meta)
    return
end

# --- DNMDT ---

"""
Pure-JuMP result of `build_quadratic_approx(::DNMDTQuadConfig, ...)`.
"""
struct DNMDTQuadResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    D <: NMDTDiscretization,
    BX_XH <: NMDTBinaryContinuousProduct,
    BX_DX <: NMDTBinaryContinuousProduct,
    DZ <: NMDTResidualProduct,
    T <: Union{Nothing, NMDTEpigraphTightening},
} <: QuadraticApproxResult
    approximation::A
    discretization::D
    bx_xh_product::BX_XH
    bx_dx_product::BX_DX
    residual_product::DZ
    tightening::T
end

"""
    build_quadratic_approx(config::DNMDTQuadConfig, model, x, bounds)

Approximate x² using double NMDT: combine two NMDT estimates with shared
discretization on x, plus the residual product δ_x·δ_x.
"""
function build_quadratic_approx(
    config::DNMDTQuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    tighten = config.epigraph_depth > 0
    x_disc = build_discretization(model, x, bounds, config.depth)
    bx_xh = build_binary_continuous_product(
        model, x_disc.beta_var, x_disc.norm_expr, 0.0, 1.0, config.depth; tighten,
    )
    bx_dx = build_binary_continuous_product(
        model,
        x_disc.beta_var,
        x_disc.delta_var,
        0.0,
        2.0^(-config.depth),
        config.depth;
        tighten,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, x_disc.delta_var,
        2.0^(-config.depth), config.depth; tighten,
    )
    approximation = build_assembled_dnmdt(
        model,
        bx_xh.result_expression,
        bx_dx.result_expression,
        bx_xh.result_expression,
        bx_dx.result_expression,
        dz.z_var,
        x_disc,
        x_disc,
        bounds,
        bounds,
    )
    tightening = if tighten
        _build_nmdt_tightening(model, approximation, x_disc, config.epigraph_depth)
    else
        nothing
    end
    return DNMDTQuadResult(approximation, x_disc, bx_xh, bx_dx, dz, tightening)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::DNMDTQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_discretization!(container, C, result.discretization, meta)
    register_binary_continuous_product!(
        container, C, result.bx_xh_product, meta * "_bx_xh",
    )
    register_binary_continuous_product!(
        container, C, result.bx_dx_product, meta * "_bx_dx",
    )
    register_residual_product!(container, C, result.residual_product, meta)

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    _register_tightening!(container, C, result.tightening, meta)
    return
end
