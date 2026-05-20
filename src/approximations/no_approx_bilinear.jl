# No-op bilinear approximation: returns the exact x·y as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for bilinear approximation: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_bilinear_approx(::NoBilinearApproxConfig, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: return `(; approximation = x*y)` for a single JuMP scalar pair.
Bounds are accepted for signature parity with other bilinear methods and unused.
"""
function build_bilinear_approx(
    ::NoBilinearApproxConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    return (; approximation = x * y)
end

# --- IOM adapter (allocate, loop, write) ---

"""
    add_bilinear_approx!(::NoBilinearApproxConfig, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Allocate a `BilinearProductExpression` container with axes `(name, t)`, loop
over the cells, and write the exact `x*y` QuadExpr per cell.
"""
function add_bilinear_approx!(
    ::NoBilinearApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    model = get_jump_model(container)
    target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis;
        meta, expr_type = JuMP.QuadExpr,
    )
    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_bilinear_approx(
                NoBilinearApproxConfig(), model,
                x_var[name, t], y_var[name, t],
                xmn, xmx, ymn, ymx,
            )
            target[name, t] = r.approximation
        end
    end
    return target
end

# --- Legacy vectorized build + register + precomputed-form entrypoint
# (kept for the generic add_bilinear_approx! wrapper in common.jl and the
# Bin2/HybS swap-in pattern, until callers migrate; removed in sweep) ---

"Pure-JuMP result of the no-op bilinear approximation (legacy)."
struct NoBilinearApproxResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.QuadExpr, 2},
} <: BilinearApproxResult
    approximation::A
end

"""
    build_bilinear_approx(::NoBilinearApproxConfig, model, x, y, x_bounds, y_bounds)

Legacy vectorized form. Returns a `NoBilinearApproxResult` wrapping a 2D
`DenseAxisArray{QuadExpr}` of `x[name,t]*y[name,t]`.
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

Legacy precomputed-form entrypoint: signature-compatible with the
precomputed-form of `Bin2Config` / `HybSConfig`, so a caller can swap
configs without changing the call site. `xsq` and `ysq` are accepted but
ignored — the no-op approximation just returns the exact `x·y` product as
a `QuadExpr`.
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
