# No-op bilinear approximation: returns the exact x·y as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for bilinear approximation: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

"Pure-JuMP result of the no-op bilinear approximation."
struct NoBilinearApproxResult{A} <: BilinearApproxResult
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
        container,
        BilinearProductExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    for name in name_axis, t in time_axis
        target[name, t] = result.approximation[name, t]
    end
    return
end
