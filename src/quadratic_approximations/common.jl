"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

# --- Quadratic approximation config hierarchy ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

# --- Sidedness trait ---
#
# Classifies where a quadratic approximation `z` sits relative to the true `x²`.
# This is the property that governs which methods compose as inner approximations
# (e.g. Bin2/HybS require one-sided-over inner Qs). Encoding it as a trait replaces
# hand-maintained `Union{...}` constraints scattered across the bilinear methods.

"Supertype for the sidedness trait of a quadratic approximation."
abstract type ApproxSidedness end
"`z ≥ x²` at every feasible point (over-approximation)."
struct OneSidedOver <: ApproxSidedness end
"`z ≤ x²` at every feasible point (under-approximation)."
struct OneSidedUnder <: ApproxSidedness end
"`|z − x²|` is bounded but `z` may sit on either side of `x²`."
struct TwoSided <: ApproxSidedness end

"""
    sidedness(::Type{<:QuadraticApproxConfig})::ApproxSidedness

The sidedness of the approximation a config produces. Defaults to the conservative
`TwoSided()` so that any config without an explicit method is simply rejected by
one-sided-over composition checks rather than silently mis-treated. Each concrete
config that participates in composition overrides this.
"""
sidedness(::Type{<:QuadraticApproxConfig}) = TwoSided()

# --- SOS2 backend tag types ---

"Supertype for the adjacency-enforcement backend of the SOS2 quadratic approximation."
abstract type SOS2Backend end
"Enforce SOS2 adjacency with solver-native `MOI.SOS2` constraints."
struct SolverBackend <: SOS2Backend end
"Enforce SOS2 adjacency manually with binary segment-selection variables."
struct ManualBackend <: SOS2Backend end

# --- Tighteners ---
#
# A `Tightener` is an optional relaxation strengthener attached to an approximation
# config via its `tightener` field, replacing the previously inconsistent knobs
# (`pwmcc_segments::Int`, `epigraph_depth::Int`, `add_mccormick::Bool`). Tighteners are
# applied at config-specific points (post-hoc cuts for McCormick; structurally inline for
# the epigraph), so there is no single apply entry point — each config reads its field.

"Optional relaxation strengthener attached to an approximation config."
abstract type Tightener end

"No tightening."
struct NoTightener <: Tightener end

"""
    McCormickTightener(partitions = 1; backend = ManualBackend())

McCormick relaxation cuts. `partitions = 1` is the standard McCormick envelope; `partitions
> 1` is piecewise McCormick (PWMCC), which partitions the domain into K sub-intervals — only
meaningful for the SOS2 quadratic approximation, where it tightens the concave `−v²` term.
`backend` selects how PWMCC's interval selection is enforced (`ManualBackend` → binaries;
`SolverBackend` → solver-native `MOI.SOS1`); it is ignored for `partitions = 1` and for the
separable (Bin2/HybS) configs, which only use the standard envelope. Preserves the
MIP-optimal solution (a valid-inequality cut).
"""
struct McCormickTightener <: Tightener
    partitions::Int
    backend::SOS2Backend

    function McCormickTightener(partitions::Int = 1; backend::SOS2Backend = ManualBackend())
        partitions >= 1 ||
            throw(ArgumentError("McCormickTightener partitions must be ≥ 1, got $(partitions)"))
        return new(partitions, backend)
    end
end

"""
    EpigraphTightener(depth)

Epigraph (Q^{L1}) lower-bound tightening at the given sawtooth depth. Unlike McCormick,
this is a **structural** change: it enlarges/reshapes the MIP-feasible set (a free result
variable bounded below by the epigraph envelope) rather than only cutting the LP relaxation.
"""
struct EpigraphTightener <: Tightener
    depth::Int

    function EpigraphTightener(depth::Int)
        depth >= 1 ||
            throw(ArgumentError("EpigraphTightener depth must be ≥ 1, got $(depth)"))
        return new(depth)
    end
end

"""
    preserves_mip_optimum(::Tightener)::Bool

Whether the tightener leaves the MIP-feasible set (hence the MIP optimum) unchanged. McCormick
cuts (incl. PWMCC) are valid inequalities and preserve it; the epigraph tightener is structural
and does not.
"""
preserves_mip_optimum(::NoTightener) = true
preserves_mip_optimum(::McCormickTightener) = true
preserves_mip_optimum(::EpigraphTightener) = false

"""
    supports_tightener(::Type{<:ApproxConfig}, ::Tightener)::Bool

Whether a config kind accepts a tightener kind. Defaults to `false`; `NoTightener` is always
accepted. Each config that supports a specific tightener overrides this. Numeric constraints
(e.g. PWMCC partitions dividing the SOS2 depth) are checked separately in the config
constructor. `ApproxConfig` is the union of the quadratic and bilinear config supertypes.
"""
supports_tightener(::Type{<:QuadraticApproxConfig}, ::Tightener) = false
supports_tightener(::Type{<:QuadraticApproxConfig}, ::NoTightener) = true

# --- NMDT discretization variants ---

"Single- vs double-discretization variant of the NMDT family."
abstract type NMDTVariant end
"Single NMDT: discretizes one factor (worst-case error exponent `2^{-L-2}`)."
struct SingleNMDT <: NMDTVariant end
"Double NMDT (DNMDT): discretizes both factors (worst-case error exponent `2^{-2L-2}`)."
struct DoubleNMDT <: NMDTVariant end

"""
    add_quadratic_approx!(config, container, ::Type{C}, names, time_steps, x_var, bounds, meta)

Approximate `x²` for each `(name, t)`, storing the result in an expression container
and returning it. This is the uniform-arity interface every `QuadraticApproxConfig`
implements; methods differ only by the config type. Internal staged builders that take
precomputed inputs (e.g. an `NMDTDiscretization`) use distinct names and are not part
of this interface.
"""
function add_quadratic_approx! end

# Fallback: a config without a concrete `add_quadratic_approx!` method lands here.
add_quadratic_approx!(config::QuadraticApproxConfig, args...) = error(
    "add_quadratic_approx! is not implemented for $(typeof(config)); required signature: " *
    "(config, container, ::Type{C}, names::Vector{String}, time_steps::UnitRange{Int}, " *
    "x_var, bounds::Vector{MinMax}, meta::String)",
)

"""
    tolerance_depth(::Type{<:QuadraticApproxConfig}; tolerance, max_delta)::Int

Smallest depth `L` whose worst-case approximation error on a domain of length
`max_delta = Δ` is ≤ `tolerance`. Each concrete config implements this method
with its own depth-to-error formula.

Every config's worst-case error has the form `Δ²·c(L)`, where `c(L)` is the
method's unit-domain error coefficient (Sawtooth: `2^{-2L-2}`, NMDT:
`2^{-L-2}`, DNMDT: `2^{-2L-2}`, SOS2: `1/(4·L²)`, Epigraph: `2^{-2L-4}`,
…). The `Δ²` prefactor comes from unnormalization: each method normalizes
`x = a + Δ·xh` with `xh ∈ [0, 1]`, approximates the only nonlinear term
`Δ²·xh²` in `x² = a² + 2a·Δ·xh + Δ²·xh²`, and inherits the unit-domain
error scaled by `Δ²`.
"""
function tolerance_depth end

"""
    _check_tolerance_args(tolerance::Float64, deltas::Float64...)

Validate the common arguments of every `tolerance_depth` helper. Throws an
informative `ArgumentError` if `tolerance` or any domain length in `deltas` is
not strictly positive, so callers get a clear message instead of a low-level
`DomainError`/`InexactError` surfacing from `sqrt`/`log2`/`ceil`.
"""
function _check_tolerance_args(tolerance::Float64, deltas::Float64...)
    tolerance > 0 ||
        throw(ArgumentError("tolerance must be strictly positive, got $(tolerance)"))
    for d in deltas
        d > 0 || throw(
            ArgumentError("domain length (max_delta) must be strictly positive, got $(d)"),
        )
    end
    return nothing
end

"""
    _ceil_positive(x::Float64)::Int

Smallest integer ≥ x, clamped to ≥ 1. Used by every `tolerance_depth` helper
to convert a real-valued depth bound (e.g. `log₂(Δ²/τ)/2`) into a usable depth.
Guards against non-finite `x` (which would otherwise throw an opaque
`InexactError` from `ceil(Int, Inf)`).
"""
function _ceil_positive(x::Float64)::Int
    isfinite(x) || throw(
        ArgumentError(
            "tolerance_depth produced a non-finite depth ($(x)); " *
            "check that tolerance and max_delta are positive and finite",
        ),
    )
    return max(1, ceil(Int, x))
end

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
