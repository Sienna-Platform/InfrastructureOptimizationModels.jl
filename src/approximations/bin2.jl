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
struct Bin2Config <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    add_mccormick::Bool
end
function Bin2Config(quad_config::QuadraticApproxConfig)
    return Bin2Config(quad_config, true)
end

"""
Pure-JuMP result of `build_bilinear_approx(::Bin2Config, ...)`.
"""
struct Bin2BilinearResult{A, XSQ, YSQ, PSQ, P, MC} <: BilinearApproxResult
    approximation::A
    xsq_result::XSQ
    ysq_result::YSQ
    psq_result::PSQ
    sum_expression::P
    mccormick_constraints::MC  # Union{Nothing, DenseAxisArray}
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
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)

    xsq = build_quadratic_approx(config.quad_config, model, x, x_bounds)
    ysq = build_quadratic_approx(config.quad_config, model, y, y_bounds)

    p_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] + y[name, t]
    )
    p_bounds = [
        MinMax((
            min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max,
        )) for i in eachindex(x_bounds)
    ]
    psq = build_quadratic_approx(config.quad_config, model, p_expr, p_bounds)

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        0.5 *
        (
            psq.approximation[name, t] - xsq.approximation[name, t] -
            ysq.approximation[name, t]
        )
    )
    mc = if config.add_mccormick
        build_reformulated_mccormick(
        model,
        x,
        y,
        psq.approximation,
        xsq.approximation,
        ysq.approximation,
        x_bounds,
        y_bounds,
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
        container,
        VariableSumExpression,
        C,
        collect(name_axis),
        time_axis;
        meta = meta * "_plus",
    )
    for name in name_axis, t in time_axis
        p_target[name, t] = result.sum_expression[name, t]
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
        register_reformulated_mccormick!(
            container, C, result.mccormick_constraints, meta,
        )
    end
    return
end
