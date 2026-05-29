"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

# --- Quadratic approximation config hierarchy ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

"""
    _ceil_positive(x::Float64)::Int

Smallest integer ≥ x, clamped to ≥ 1. Used by every `tolerance_depth` helper
to convert a real-valued depth bound (e.g. `log₂(Δ²/τ)/2`) into a usable depth.
"""
_ceil_positive(x::Float64)::Int = max(1, ceil(Int, x))

"""
    _normed_variable!(container, C, names, time_steps, x_var, bounds, meta)

Create an affine expression for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1].

Stores results in a `NormedVariableExpression` expression container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _normed_variable!(
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
        NormedVariableExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        b = bounds[i]
        IS.@assert_op b.max > b.min
        lx = b.max - b.min
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        # add_proportional + add_constant accept any AbstractJuMPScalar for x_var,
        # so this works for both VariableRef and AffExpr inputs (the latter is
        # used by Bin2 to normalize the (x+y) expression).
        add_proportional_to_jump_expression!(result, x_var[name, t], 1.0 / lx)
        add_constant_to_jump_expression!(result, -b.min / lx)
    end
    return result_expr
end
