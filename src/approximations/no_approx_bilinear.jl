# No-op bilinear approximation: returns the exact x·y as a JuMP.QuadExpr.
# For NLP-capable solvers or testing.

"No-op config for bilinear approximation: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

"""
    build_bilinear_approx(::NoBilinearApproxConfig, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: return `(; approximation = x*y)` for a single JuMP scalar pair.
Bounds accepted for signature parity, unused.
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

"""
    add_bilinear_approx!(::NoBilinearApproxConfig, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Allocate a `BilinearProductExpression` container with axes `(name, t)`,
loop, and write the exact `x*y` per cell.
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
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
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

"""
    add_bilinear_approx!(::NoBilinearApproxConfig, container, ::Type{C}, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form entrypoint: signature-compatible with the precomputed-form
of `Bin2Config` / `HybSConfig`, so a caller can swap configs without
changing the call site. `xsq` and `ysq` are accepted but ignored — the
no-op approximation just returns the exact `x*y` product.
"""
function add_bilinear_approx!(
    ::NoBilinearApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    xsq,
    ysq,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    return add_bilinear_approx!(
        NoBilinearApproxConfig(), container, C, x_var, y_var, x_bounds, y_bounds, meta,
    )
end
