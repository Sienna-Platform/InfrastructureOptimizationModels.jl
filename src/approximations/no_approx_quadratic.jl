# No-op quadratic approximation: returns the exact x² as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for quadratic approximation: returns exact x² as a QuadExpr."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_quadratic_approx(::NoQuadApproxConfig, model, x, x_min, x_max)

Scalar form: return `(; approximation = x*x)` for a single JuMP scalar.
Bounds are accepted for signature parity with other quadratic methods and unused.
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

# --- IOM adapter (allocate, loop, write) ---

"""
    add_quadratic_approx!(::NoQuadApproxConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate a `QuadraticExpression` container with axes `(name, t)`, loop over
the cells, call the scalar `build_quadratic_approx(::NoQuadApproxConfig, ...)`,
and write the exact `x*x` QuadExpr per cell.
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
    IS.@assert_op length(name_axis) == length(x_bounds)
    model = get_jump_model(container)
    target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(NoQuadApproxConfig(), model, x_var[name, t], xmn, xmx)
            target[name, t] = r.approximation
        end
    end
    return target
end

# --- Legacy vectorized build + register (kept for the generic add_quadratic_approx!
# wrapper in common.jl until callers migrate; removed in sweep) ---

"Pure-JuMP result of the no-op quadratic approximation (legacy)."
struct NoQuadApproxResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.QuadExpr, 2},
} <: QuadraticApproxResult
    approximation::A
end

"""
    build_quadratic_approx(::NoQuadApproxConfig, model, x, bounds)

Legacy vectorized form. Returns a `NoQuadApproxResult` wrapping a 2D
`DenseAxisArray{QuadExpr}` of `x[name,t]^2`.
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
        container, QuadraticExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    target.data .= result.approximation.data
    return
end
