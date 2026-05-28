# No-op bilinear approximation: returns exact x·y as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op bilinear config: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

# No tolerance-form constructor — see the matching note in
# src/quadratic_approximations/no_approx.jl: an empty-struct kw constructor
# would clobber the auto-generated `NoBilinearApproxConfig()` constructor.

"""
    _add_bilinear_approx!(::NoBilinearApproxConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

No-op bilinear approximation: returns exact x·y as a QuadExpr.

# Arguments
- `::NoBilinearApproxConfig`: no-op configuration (no fields)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y domain
- `meta::String`: variable type identifier for the approximation
"""
function _add_bilinear_approx!(
    ::NoBilinearApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression,
        C,
        names,
        time_steps;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    for name in names, t in time_steps
        result_expr[name, t] = x_var[name, t] * y_var[name, t]
    end
    return result_expr
end

"""
    _add_bilinear_approx!(::NoBilinearApproxConfig, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form no-op bilinear approximation: returns exact x·y as a QuadExpr.
`xsq` and `ysq` are accepted for signature parity with the precomputed-form
dispatches of `Bin2Config` and `HybSConfig` (so a caller can swap configs
without changing the call site) but are unused here.

# Arguments
- `::NoBilinearApproxConfig`: no-op configuration (no fields)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `xsq`: precomputed x² container (ignored)
- `ysq`: precomputed y² container (ignored)
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-component bounds on x
- `y_bounds::Vector{MinMax}`: per-component bounds on y
- `meta::String`: variable type identifier for the approximation
"""
function _add_bilinear_approx!(
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
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression,
        C,
        names,
        time_steps;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    for name in names, t in time_steps
        result_expr[name, t] = x_var[name, t] * y_var[name, t]
    end
    return result_expr
end
