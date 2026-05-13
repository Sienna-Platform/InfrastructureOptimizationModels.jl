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
struct HybSConfig <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    epigraph_depth::Int
    add_mccormick::Bool
end
function HybSConfig(quad_config::QuadraticApproxConfig, epigraph_depth::Int)
    return HybSConfig(quad_config, epigraph_depth, false)
end

"""
Pure-JuMP result of `build_bilinear_approx(::HybSConfig, ...)`.
"""
struct HybSBilinearResult{A, XSQ, YSQ, P1, P2, ZP1, ZP2, ZV, BC, MC} <: BilinearApproxResult
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
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)

    xsq = build_quadratic_approx(config.quad_config, model, x, x_bounds)
    ysq = build_quadratic_approx(config.quad_config, model, y, y_bounds)

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

    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1 = build_quadratic_approx(epi_cfg, model, p1_expr, p1_bounds)
    zp2 = build_quadratic_approx(epi_cfg, model, p2_expr, p2_bounds)

    z_lo = [
        min(
            x_bounds[i].min * y_bounds[i].min,
            x_bounds[i].min * y_bounds[i].max,
            x_bounds[i].max * y_bounds[i].min,
            x_bounds[i].max * y_bounds[i].max,
        ) for i in eachindex(x_bounds)
    ]
    z_hi = [
        max(
            x_bounds[i].min * y_bounds[i].min,
            x_bounds[i].min * y_bounds[i].max,
            x_bounds[i].max * y_bounds[i].min,
            x_bounds[i].max * y_bounds[i].max,
        ) for i in eachindex(x_bounds)
    ]
    z_lo_arr = JuMP.Containers.DenseAxisArray(z_lo, name_axis)
    z_hi_arr = JuMP.Containers.DenseAxisArray(z_hi, name_axis)

    z_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        base_name = "HybSProduct",
    )
    for name in name_axis, t in time_axis
        JuMP.set_lower_bound(z_var[name, t], z_lo_arr[name])
        JuMP.set_upper_bound(z_var[name, t], z_hi_arr[name])
    end

    bound_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, 1:2, time_axis,
    )
    for name in name_axis, t in time_axis
        # Bin2 lower bound: z ≥ ½·(zp1 − zx − zy)
        bound_cons[name, 1, t] = JuMP.@constraint(
            model,
            z_var[name, t] >=
            0.5 *
            (
                zp1.approximation[name, t] - xsq.approximation[name, t] -
                ysq.approximation[name, t]
            ),
        )
        # Bin3 upper bound: z ≤ ½·(zx + zy − zp2)
        bound_cons[name, 2, t] = JuMP.@constraint(
            model,
            z_var[name, t] <=
            0.5 *
            (
                xsq.approximation[name, t] + ysq.approximation[name, t] -
                zp2.approximation[name, t]
            ),
        )
    end

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
        approximation,
        xsq,
        ysq,
        p1_expr,
        p2_expr,
        zp1,
        zp2,
        z_var,
        bound_cons,
        mc,
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
        container,
        VariableSumExpression,
        C,
        collect(name_axis),
        time_axis;
        meta = meta * "_plus",
    )
    p2_target = add_expression_container!(
        container,
        VariableDifferenceExpression,
        C,
        collect(name_axis),
        time_axis;
        meta = meta * "_diff",
    )
    for name in name_axis, t in time_axis
        p1_target[name, t] = result.sum_expression[name, t]
        p2_target[name, t] = result.diff_expression[name, t]
    end

    z_target = add_variable_container!(
        container,
        BilinearProductVariable,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        z_target[name, t] = result.z_var[name, t]
    end

    bound_target = add_constraints_container!(
        container,
        HybSBoundConstraint,
        C,
        collect(name_axis),
        1:2,
        time_axis;
        sparse = true,
        meta,
    )
    for name in name_axis, k in 1:2, t in time_axis
        bound_target[(name, k, t)] = result.bound_constraints[name, k, t]
    end

    result_target = add_expression_container!(
        container,
        BilinearProductExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        result_target[name, t] = result.approximation[name, t]
    end

    if result.mccormick_constraints !== nothing
        register_mccormick_envelope!(container, C, result.mccormick_constraints, meta)
    end
    return
end
