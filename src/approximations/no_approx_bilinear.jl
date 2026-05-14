# No-op bilinear approximation: returns the exact x·y as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for bilinear approximation: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

"Pure-JuMP result of the no-op bilinear approximation."
struct NoBilinearApproxResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.QuadExpr, 2},
} <: BilinearApproxResult
    approximation::A
end

"""
    build_bilinear_approx(::NoBilinearApproxConfig, model, x, y, x_bounds, y_bounds)

Build the exact x·y product. Bounds are accepted for signature parity and unused.
"""
function build_bilinear_approx(
    ::NoBilinearApproxConfig,
    model::JuMP.Model,
    x,
    y,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] * y[name, t]
    )
    return NoBilinearApproxResult(approximation)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::NoBilinearApproxResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    target.data .= result.approximation.data
    return
end

"""
    add_bilinear_approx!(::NoBilinearApproxConfig, container, C, names, time_steps,
                         xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: signature-compatible with the precomputed-form
of `Bin2Config` / `HybSConfig`, so a caller can swap configs without
changing the call site. `xsq` and `ysq` are accepted but ignored — the
no-op approximation just returns the exact `x·y` product as a `QuadExpr`.
"""
function add_bilinear_approx!(
    ::NoBilinearApproxConfig,
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
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    target.data .=
        JuMP.@expression(
            get_jump_model(container),
            [name = name_axis, t = time_axis],
            x_var[name, t] * y_var[name, t]
        ).data
    return target
end
