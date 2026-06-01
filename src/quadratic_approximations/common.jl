"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

# --- Quadratic approximation config hierarchy ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

"""
    tolerance_depth(::Type{<:QuadraticApproxConfig}; tolerance, max_delta)::Int

Smallest depth `L` whose worst-case approximation error on a domain of length
`max_delta = Δ` is ≤ `tolerance`. Each concrete config implements this method
with its own depth-to-error formula.

Every config's worst-case error has the form `Δ²·c(L)`, where `c(L)` is the
method's unit-domain error coefficient (Sawtooth: `2^{-2L-2}`, NMDT:
`2^{-L-2}`, DNMDT: `2^{-2L-2}`, SOS2: `1/(4·L²)`, Epigraph: `2^{-2L-2}`,
…). The `Δ²` prefactor comes from unnormalization: each method normalizes
`x = a + Δ·xh` with `xh ∈ [0, 1]`, approximates the only nonlinear term
`Δ²·xh²` in `x² = a² + 2a·Δ·xh + Δ²·xh²`, and inherits the unit-domain
error scaled by `Δ²`.
"""
function tolerance_depth end

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
