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
struct NMDTBilinearResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    XD <: NMDTDiscretization,
    YN <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    BX <: NMDTBinaryContinuousProduct,
    DZ <: NMDTResidualProduct,
} <: BilinearApproxResult
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
    return _build_nmdt_with_precomputed(
        config, model, x_disc, yh_expr, x_bounds, y_bounds,
    )
end

# Shared math between the standard and precomputed-form entrypoints.
function _build_nmdt_with_precomputed(
    config::NMDTBilinearConfig,
    model::JuMP.Model,
    x_disc::NMDTDiscretization,
    yh_expr,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    bx_yh = build_binary_continuous_product(
        model, x_disc.beta_var, yh_expr, 0.0, 1.0, config.depth,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, yh_expr, 1.0, config.depth,
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

    name_axis = axes(result.yh_expression, 1)
    time_axis = axes(result.yh_expression, 2)
    yh_target = add_expression_container!(
        container, NormedVariableExpression, C, name_axis, time_axis;
        meta = meta * "_y",
    )
    yh_target.data .= result.yh_expression.data

    register_binary_continuous_product!(container, C, result.bx_yh_product, meta)
    register_residual_product!(container, C, result.residual_product, meta)

    result_name_axis = axes(result.approximation, 1)
    result_time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, BilinearProductExpression, C, result_name_axis, result_time_axis;
        meta,
    )
    result_target.data .= result.approximation.data
    return
end

"""
    add_bilinear_approx!(config::NMDTBilinearConfig, container, C, names, time_steps,
                         x_disc, yh_expr, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: accepts an already-built `x_disc::NMDTDiscretization`
and `yh_expr` (the normalized-y expression container) and builds only the
binary-continuous product, residual product, and final assembly on top.
"""
function add_bilinear_approx!(
    config::NMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    yh_expr::JuMP.Containers.DenseAxisArray,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result = _build_nmdt_with_precomputed(
        config, get_jump_model(container), x_disc, yh_expr, x_bounds, y_bounds,
    )
    register_binary_continuous_product!(container, C, result.bx_yh_product, meta)
    register_residual_product!(container, C, result.residual_product, meta)
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data
    return result_target
end

# --- DNMDT (double discretization) ---

"""
Pure-JuMP result of `build_bilinear_approx(::DNMDTBilinearConfig, ...)`.
"""
struct DNMDTBilinearResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    XD <: NMDTDiscretization,
    YD <: NMDTDiscretization,
    BXY <: NMDTBinaryContinuousProduct,
    BYD <: NMDTBinaryContinuousProduct,
    BYX <: NMDTBinaryContinuousProduct,
    BXD <: NMDTBinaryContinuousProduct,
    DZ <: NMDTResidualProduct,
} <: BilinearApproxResult
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
    return _build_dnmdt_with_precomputed(
        config, model, x_disc, y_disc, x_bounds, y_bounds,
    )
end

function _build_dnmdt_with_precomputed(
    config::DNMDTBilinearConfig,
    model::JuMP.Model,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    bx_yh = build_binary_continuous_product(
        model, x_disc.beta_var, y_disc.norm_expr, 0.0, 1.0, config.depth,
    )
    by_dx = build_binary_continuous_product(
        model, y_disc.beta_var, x_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth,
    )
    by_xh = build_binary_continuous_product(
        model, y_disc.beta_var, x_disc.norm_expr, 0.0, 1.0, config.depth,
    )
    bx_dy = build_binary_continuous_product(
        model, x_disc.beta_var, y_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, y_disc.delta_var,
        2.0^(-config.depth), config.depth,
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

    register_binary_continuous_product!(container, C, result.bx_yh_product, meta * "_bx_yh")
    register_binary_continuous_product!(container, C, result.by_dx_product, meta * "_by_dx")
    register_binary_continuous_product!(container, C, result.by_xh_product, meta * "_by_xh")
    register_binary_continuous_product!(container, C, result.bx_dy_product, meta * "_bx_dy")
    register_residual_product!(container, C, result.residual_product, meta)

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data
    return
end

"""
    add_bilinear_approx!(config::DNMDTBilinearConfig, container, C, names, time_steps,
                         x_disc, y_disc, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: accepts already-built `x_disc` and `y_disc`
NMDT discretizations and builds only the four cross binary-continuous
products, the shared residual product, and the final convex assembly.
"""
function add_bilinear_approx!(
    config::DNMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result = _build_dnmdt_with_precomputed(
        config, get_jump_model(container), x_disc, y_disc, x_bounds, y_bounds,
    )
    register_binary_continuous_product!(container, C, result.bx_yh_product, meta * "_bx_yh")
    register_binary_continuous_product!(container, C, result.by_dx_product, meta * "_by_dx")
    register_binary_continuous_product!(container, C, result.by_xh_product, meta * "_by_xh")
    register_binary_continuous_product!(container, C, result.bx_dy_product, meta * "_bx_dy")
    register_residual_product!(container, C, result.residual_product, meta)
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data
    return result_target
end
