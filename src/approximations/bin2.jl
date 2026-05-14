# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity x·y = ½·((x+y)² − x² − y²).
# Composes a quadratic approximation (chosen via `quad_config`) for x², y²,
# and (x+y)². Optionally adds reformulated McCormick cuts to tighten the LP
# relaxation in terms of the three quadratic approximations.

"""
Config for Bin2 bilinear approximation using z = ½·((x+y)² − x² − y²).

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for x², y², and (x+y)².
- `add_mccormick::Bool`: whether to add reformulated McCormick cuts (default true).
"""
struct Bin2Config{QC <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::QC
    add_mccormick::Bool
end
function Bin2Config(quad_config::QuadraticApproxConfig)
    return Bin2Config(quad_config, true)
end

"""
Pure-JuMP result of `build_bilinear_approx(::Bin2Config, ...)`.
"""
struct Bin2BilinearResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    XSQ <: QuadraticApproxResult,
    YSQ <: QuadraticApproxResult,
    PSQ <: QuadraticApproxResult,
    P <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    MC <: Union{
        Nothing,
        Tuple{
            <:JuMP.Containers.DenseAxisArray,
            <:JuMP.Containers.DenseAxisArray,
            <:JuMP.Containers.DenseAxisArray,
            <:JuMP.Containers.DenseAxisArray,
        },
    },
} <: BilinearApproxResult
    approximation::A
    xsq_result::XSQ
    ysq_result::YSQ
    psq_result::PSQ
    sum_expression::P
    mccormick_constraints::MC
end

"""
    build_bilinear_approx(config::Bin2Config, model, x, y, x_bounds, y_bounds)

Bin2 separable bilinear approximation: build x², y², and (x+y)² via the
chosen quadratic method, then combine via z = ½·(psq − xsq − ysq).
If `config.add_mccormick`, append the four reformulated McCormick cuts.
"""
function build_bilinear_approx(
    config::Bin2Config,
    model::JuMP.Model,
    x,
    y,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    xsq = build_quadratic_approx(config.quad_config, model, x, x_bounds)
    ysq = build_quadratic_approx(config.quad_config, model, y, y_bounds)

    name_axis = axes(x, 1)
    time_axis = axes(x, 2)

    p_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] + y[name, t]
    )
    p_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]
    psq = build_quadratic_approx(config.quad_config, model, p_expr, p_bounds)

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        0.5 * (
            psq.approximation[name, t] - xsq.approximation[name, t] -
            ysq.approximation[name, t]
        )
    )
    mc = if config.add_mccormick
        build_reformulated_mccormick(
            model, x, y,
            psq.approximation, xsq.approximation, ysq.approximation,
            x_bounds, y_bounds,
        )
    else
        nothing
    end

    return Bin2BilinearResult(approximation, xsq, ysq, psq, p_expr, mc)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::Bin2BilinearResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    register_in_container!(container, C, result.xsq_result, meta * "_x")
    register_in_container!(container, C, result.ysq_result, meta * "_y")
    register_in_container!(container, C, result.psq_result, meta * "_plus")

    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)

    p_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    p_target.data .= result.sum_expression.data

    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    register_reformulated_mccormick!(container, C, result.mccormick_constraints, meta)
    return
end

"""
    add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps,
                         xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: accepts already-built quadratic approximation
expression containers `xsq` ≈ x² and `ysq` ≈ y² (rather than re-computing
them). The Bin2 identity z = ½·((x+y)² − xsq − ysq) is built on top, along
with the (x+y)² approximation, sum expression, and optional reformulated
McCormick cuts.
"""
function add_bilinear_approx!(
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
    model = get_jump_model(container)
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)

    p_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x_var[name, t] + y_var[name, t]
    )
    p_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]
    psq = build_quadratic_approx(config.quad_config, model, p_expr, p_bounds)
    register_in_container!(container, C, psq, meta * "_plus")

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        0.5 * (psq.approximation[name, t] - xsq[name, t] - ysq[name, t])
    )

    p_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    p_target.data .= p_expr.data

    result_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= approximation.data

    if config.add_mccormick
        mc = build_reformulated_mccormick(
            model, x_var, y_var, psq.approximation, xsq, ysq,
            x_bounds, y_bounds,
        )
        register_reformulated_mccormick!(container, C, mc, meta)
    end
    return result_target
end
