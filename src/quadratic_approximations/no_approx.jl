# No-op quadratic approximation: returns exact x² as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op config: returns exact x² as a QuadExpr (for NLP-capable solvers or testing)."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

# No tolerance-form constructor: this is an exact (NLP) formulation, so
# `NoQuadApproxConfig(; tolerance=…)` produces Julia's natural MethodError.
# (A hand-rolled kw stub here would clobber the auto-generated `NoQuadApproxConfig()`
# constructor — they share the same lowered positional signature.)

"""
    _add_quadratic_approx!(::NoQuadApproxConfig, container, C, names, time_steps, x_var, bounds, meta)

No-op quadratic approximation: returns exact x² as a QuadExpr.

# Arguments
- `::NoQuadApproxConfig`: no-op configuration (no fields)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: variable type identifier for the approximation
"""
function _add_quadratic_approx!(
    ::NoQuadApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        names,
        time_steps;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    for name in names, t in time_steps
        result_expr[name, t] = x_var[name, t] * x_var[name, t]
    end
    return result_expr
end
