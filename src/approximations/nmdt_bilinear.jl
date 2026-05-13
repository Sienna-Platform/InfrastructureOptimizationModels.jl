# NMDT bilinear approximations of z = x·y.
#   NMDTBilinearConfig  — discretize x only; y is just normalized.
#   DNMDTBilinearConfig — discretize both x and y, combine two estimates.

"""
Config for single-NMDT bilinear approximation (discretizes x only).

# Fields
- `depth::Int`: number of binary discretization levels L for x.
"""
struct NMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

"""
Config for double-NMDT bilinear approximation (discretizes both x and y).

# Fields
- `depth::Int`: number of binary discretization levels L for both x and y.
"""
struct DNMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

# --- NMDT (single discretization) ---

"""
Pure-JuMP result of `build_bilinear_approx(::NMDTBilinearConfig, ...)`.
"""
struct NMDTBilinearResult{A, XD, YN, BX, DZ} <: BilinearApproxResult
    approximation::A
    x_discretization::XD
    yh_expression::YN
    bx_yh_product::BX
    residual_product::DZ
end

"""
    build_bilinear_approx(config::NMDTBilinearConfig, model, x, y, x_bounds, y_bounds)

Approximate x·y via NMDT: discretize x, normalize y to yh ∈ [0,1], build the
binary-continuous product β·yh and residual δ·yh, reassemble x·y from
normalized components.
"""
function build_bilinear_approx(
    config::NMDTBilinearConfig,
    model::JuMP.Model,
    x,
    y,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    x_disc = build_discretization(model, x, x_bounds, config.depth)
    yh_expr = build_normed_variable(model, y, y_bounds)
    bx_yh = build_binary_continuous_product(
        model,
        x_disc.beta_var,
        yh_expr,
        0.0,
        1.0,
        config.depth,
    )
    dz = build_residual_product(
        model,
        x_disc.delta_var,
        yh_expr,
        1.0,
        config.depth,
    )
    approximation = build_assembled_product(
        model,
        [bx_yh.result_expression],
        dz.z_var,
        x_disc.norm_expr,
        yh_expr,
        x_bounds,
        y_bounds,
    )
    return NMDTBilinearResult(approximation, x_disc, yh_expr, bx_yh, dz)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::NMDTBilinearResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_discretization!(container, C, result.x_discretization, meta * "_x")

    yh_target = add_expression_container!(
        container,
        NormedVariableExpression,
        C,
        collect(axes(result.yh_expression, 1)),
        axes(result.yh_expression, 2);
        meta = meta * "_y",
    )
    for name in axes(result.yh_expression, 1), t in axes(result.yh_expression, 2)
        yh_target[name, t] = result.yh_expression[name, t]
    end

    register_binary_continuous_product!(container, C, result.bx_yh_product, meta)
    register_residual_product!(container, C, result.residual_product, meta)

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
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
    return
end

# --- DNMDT (double discretization) ---

"""
Pure-JuMP result of `build_bilinear_approx(::DNMDTBilinearConfig, ...)`.
"""
struct DNMDTBilinearResult{A, XD, YD, BXY, BYD, BYX, BXD, DZ} <: BilinearApproxResult
    approximation::A
    x_discretization::XD
    y_discretization::YD
    bx_yh_product::BXY
    by_dx_product::BYD
    by_xh_product::BYX
    bx_dy_product::BXD
    residual_product::DZ
end

"""
    build_bilinear_approx(config::DNMDTBilinearConfig, model, x, y, x_bounds, y_bounds)

DNMDT bilinear approximation: discretize both x and y, form all four cross
binary-continuous products, and convexly combine two NMDT estimates.
"""
function build_bilinear_approx(
    config::DNMDTBilinearConfig,
    model::JuMP.Model,
    x,
    y,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    x_disc = build_discretization(model, x, x_bounds, config.depth)
    y_disc = build_discretization(model, y, y_bounds, config.depth)
    bx_yh = build_binary_continuous_product(
        model,
        x_disc.beta_var,
        y_disc.norm_expr,
        0.0,
        1.0,
        config.depth,
    )
    by_dx = build_binary_continuous_product(
        model,
        y_disc.beta_var,
        x_disc.delta_var,
        0.0,
        2.0^(-config.depth),
        config.depth,
    )
    by_xh = build_binary_continuous_product(
        model,
        y_disc.beta_var,
        x_disc.norm_expr,
        0.0,
        1.0,
        config.depth,
    )
    bx_dy = build_binary_continuous_product(
        model,
        x_disc.beta_var,
        y_disc.delta_var,
        0.0,
        2.0^(-config.depth),
        config.depth,
    )
    dz = build_residual_product(
        model,
        x_disc.delta_var,
        y_disc.delta_var,
        2.0^(-config.depth),
        config.depth,
    )
    approximation = build_assembled_dnmdt(
        model,
        bx_yh.result_expression,
        by_dx.result_expression,
        by_xh.result_expression,
        bx_dy.result_expression,
        dz.z_var,
        x_disc,
        y_disc,
        x_bounds,
        y_bounds,
    )
    return DNMDTBilinearResult(
        approximation, x_disc, y_disc, bx_yh, by_dx, by_xh, bx_dy, dz,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::DNMDTBilinearResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_discretization!(container, C, result.x_discretization, meta * "_x")
    register_discretization!(container, C, result.y_discretization, meta * "_y")

    register_binary_continuous_product!(
        container, C, result.bx_yh_product, meta * "_bx_yh",
    )
    register_binary_continuous_product!(
        container, C, result.by_dx_product, meta * "_by_dx",
    )
    register_binary_continuous_product!(
        container, C, result.by_xh_product, meta * "_by_xh",
    )
    register_binary_continuous_product!(
        container, C, result.bx_dy_product, meta * "_bx_dy",
    )
    register_residual_product!(container, C, result.residual_product, meta)

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
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
    return
end
