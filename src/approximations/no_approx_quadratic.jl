# No-op quadratic approximation: returns the exact x² as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for quadratic approximation: returns exact x² as a QuadExpr."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

"""
    build_quadratic_approx(::NoQuadApproxConfig, model, x, x_min, x_max)

Scalar form: return `(; approximation = x*x)` for a single JuMP scalar.
Bounds accepted for signature parity with other quadratic methods, unused.
"""
function build_quadratic_approx(
    ::NoQuadApproxConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    return (; approximation = x * x)
end

"""
    add_quadratic_approx!(::NoQuadApproxConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate a `QuadraticExpression` container with axes `(name, t)`, loop,
call the scalar build per cell, write the exact `x*x` per cell.
"""
function add_quadratic_approx!(
    ::NoQuadApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    @assert length(name_axis) == length(x_bounds)
    model = get_jump_model(container)
    target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(
                NoQuadApproxConfig(),
                model,
                x_var[name, t],
                xmn,
                xmx,
            )
            target[name, t] = r.approximation
        end
    end
    return target
end
