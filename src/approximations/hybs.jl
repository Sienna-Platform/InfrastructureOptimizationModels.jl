# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines a Bin2 lower bound and a Bin3 upper bound with shared quadratic
# approximations of x², y² and pure-LP epigraph relaxations of (x+y)², (x−y)².
# Uses 2L binaries instead of Bin2's 3L.
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS bilinear approximation.

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for x² and y².
- `epigraph_depth::Int`: depth of the epigraph Q^{L1} approximation of (x±y)².
- `add_mccormick::Bool`: whether to add a standard McCormick envelope on z (default false).
"""
struct HybSConfig{QC <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::QC
    epigraph_depth::Int
    add_mccormick::Bool
end
function HybSConfig(quad_config::QuadraticApproxConfig, epigraph_depth::Int)
    return HybSConfig(quad_config, epigraph_depth, false)
end

"""
Pure-JuMP result of `build_bilinear_approx(::HybSConfig, ...)`.
"""
struct HybSBilinearResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    XSQ <: QuadraticApproxResult,
    YSQ <: QuadraticApproxResult,
    P1 <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    P2 <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    ZP1 <: EpigraphQuadResult,
    ZP2 <: EpigraphQuadResult,
    ZV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 2},
    BC <: JuMP.Containers.DenseAxisArray,
    MC <: Union{Nothing, NamedTuple{(:lower, :upper)}},
} <: BilinearApproxResult
    approximation::A
    xsq_result::XSQ
    ysq_result::YSQ
    sum_expression::P1
    diff_expression::P2
    sum_epigraph::ZP1
    diff_epigraph::ZP2
    z_var::ZV
    bound_constraints::BC
    mccormick_constraints::MC
end

"""
    build_bilinear_approx(config::HybSConfig, model, x, y, x_bounds, y_bounds)

HybS bilinear approximation. Builds x² and y² via the chosen quadratic
method, builds (x+y)² and (x−y)² via the epigraph Q^{L1} relaxation, and
constrains a fresh product variable z with two-sided bounds derived from
the Bin2 lower / Bin3 upper identities.
"""
function build_bilinear_approx(
    config::HybSConfig,
    model::JuMP.Model,
    x,
    y,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    xsq = build_quadratic_approx(config.quad_config, model, x, x_bounds)
    ysq = build_quadratic_approx(config.quad_config, model, y, y_bounds)
    return _build_hybs_with_precomputed(
        config, model, x, y, xsq, ysq, x_bounds, y_bounds,
    )
end

# Shared math between the standard and precomputed-form entrypoints. Wraps
# pre-existing x² / y² approximations behind a `QuadraticApproxResult`-shaped
# adapter so the call site can come from either flow.
function _build_hybs_with_precomputed(
    config::HybSConfig,
    model::JuMP.Model,
    x,
    y,
    xsq::QuadraticApproxResult,
    ysq::QuadraticApproxResult,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)

    p1_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] + y[name, t]
    )
    p2_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] - y[name, t]
    )
    p1_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]
    p2_bounds = [
        (min = x_bounds[i].min - y_bounds[i].max,
            max = x_bounds[i].max - y_bounds[i].min)
        for i in eachindex(x_bounds)
    ]

    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1 = build_quadratic_approx(epi_cfg, model, p1_expr, p1_bounds)
    zp2 = build_quadratic_approx(epi_cfg, model, p2_expr, p2_bounds)

    z_lo = JuMP.Containers.DenseAxisArray(
        [
            min(
                x_bounds[i].min * y_bounds[i].min,
                x_bounds[i].min * y_bounds[i].max,
                x_bounds[i].max * y_bounds[i].min,
                x_bounds[i].max * y_bounds[i].max,
            ) for i in eachindex(x_bounds)
        ],
        name_axis,
    )
    z_hi = JuMP.Containers.DenseAxisArray(
        [
            max(
                x_bounds[i].min * y_bounds[i].min,
                x_bounds[i].min * y_bounds[i].max,
                x_bounds[i].max * y_bounds[i].min,
                x_bounds[i].max * y_bounds[i].max,
            ) for i in eachindex(x_bounds)
        ],
        name_axis,
    )

    z_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        lower_bound = z_lo[name],
        upper_bound = z_hi[name],
        base_name = "HybSProduct",
    )

    # Bin2 lower bound: z ≥ ½·(zp1 − zx − zy)
    bound_1 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        z_var[name, t] >=
        0.5 * (
            zp1.approximation[name, t] - xsq.approximation[name, t] -
            ysq.approximation[name, t]
        ),
    )
    # Bin3 upper bound: z ≤ ½·(zx + zy − zp2)
    bound_2 = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        z_var[name, t] <=
        0.5 * (
            xsq.approximation[name, t] + ysq.approximation[name, t] -
            zp2.approximation[name, t]
        ),
    )
    # bound_1 is `z >= …` (GreaterThan), bound_2 is `z <= …` (LessThan) — use the
    # abstract ConstraintRef so both kinds fit in the same container.
    bound_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
        undef, name_axis, 1:2, time_axis,
    )
    @views bound_cons.data[:, 1, :] .= bound_1.data
    @views bound_cons.data[:, 2, :] .= bound_2.data

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        1.0 * z_var[name, t]
    )

    mc = if config.add_mccormick
        build_mccormick_envelope(model, x, y, z_var, x_bounds, y_bounds)
    else
        nothing
    end

    return HybSBilinearResult(
        approximation, xsq, ysq, p1_expr, p2_expr, zp1, zp2,
        z_var, bound_cons, mc,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::HybSBilinearResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_in_container!(container, C, result.xsq_result, meta * "_x")
    register_in_container!(container, C, result.ysq_result, meta * "_y")
    register_in_container!(container, C, result.sum_epigraph, meta * "_plus")
    register_in_container!(container, C, result.diff_epigraph, meta * "_diff")

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)

    p1_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    p1_target.data .= result.sum_expression.data

    p2_target = add_expression_container!(
        container, VariableDifferenceExpression, C, name_axis, time_axis;
        meta = meta * "_diff",
    )
    p2_target.data .= result.diff_expression.data

    z_target = add_variable_container!(
        container, BilinearProductVariable, C, name_axis, time_axis; meta,
    )
    z_target.data .= result.z_var.data

    bound_target = add_constraints_container!(
        container, HybSBoundConstraint, C, name_axis, 1:2, time_axis; meta,
    )
    bound_target.data .= result.bound_constraints.data

    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    register_mccormick_envelope!(container, C, result.mccormick_constraints, meta)
    return
end

"""
    add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps,
                         xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: accepts already-built quadratic approximation
expression containers `xsq` ≈ x² and `ysq` ≈ y², and builds the HybS bilinear
approximation on top without re-computing them.
"""
function add_bilinear_approx!(
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
    xsq_wrapped = _PrebuiltQuadApprox(xsq)
    ysq_wrapped = _PrebuiltQuadApprox(ysq)
    result = _build_hybs_with_precomputed(
        config, get_jump_model(container), x_var, y_var,
        xsq_wrapped, ysq_wrapped, x_bounds, y_bounds,
    )
    # Register only the new objects (epi, p_expr, z_var, bound_cons, approx, mc).
    register_in_container!(container, C, result.sum_epigraph, meta * "_plus")
    register_in_container!(container, C, result.diff_epigraph, meta * "_diff")

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)

    p1_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    p1_target.data .= result.sum_expression.data
    p2_target = add_expression_container!(
        container, VariableDifferenceExpression, C, name_axis, time_axis;
        meta = meta * "_diff",
    )
    p2_target.data .= result.diff_expression.data

    z_target = add_variable_container!(
        container, BilinearProductVariable, C, name_axis, time_axis; meta,
    )
    z_target.data .= result.z_var.data

    bound_target = add_constraints_container!(
        container, HybSBoundConstraint, C, name_axis, 1:2, time_axis; meta,
    )
    bound_target.data .= result.bound_constraints.data

    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    register_mccormick_envelope!(container, C, result.mccormick_constraints, meta)
    return result_target
end
