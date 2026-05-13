# No-op quadratic approximation: returns the exact x² as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for quadratic approximation: returns exact x² as a QuadExpr."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

"Pure-JuMP result of the no-op quadratic approximation."
struct NoQuadApproxResult{A} <: QuadraticApproxResult
    approximation::A
end

"""
    build_quadratic_approx(::NoQuadApproxConfig, model, x, bounds)

Build the exact x² expression for each (name, t) and wrap in a result struct.
`bounds` is accepted for signature parity but is unused.
"""
function build_quadratic_approx(
    ::NoQuadApproxConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x[name, t] * x[name, t]
    )
    return NoQuadApproxResult(approximation)
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::NoQuadApproxResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    target = add_expression_container!(
        container,
        QuadraticExpression,
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
