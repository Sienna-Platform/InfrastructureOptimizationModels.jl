# Epigraph (Q^{L1}) LP-only lower bound for x² using tangent-line cuts.
# Pure LP — zero binary variables. Creates a variable z ≥ x² (approximately)
# bounded from below by supporting hyperplanes of the parabola.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024), Q^{L1} relaxation.

"Expression container for epigraph quadratic approximation results."
struct EpigraphExpression <: ExpressionType end

"Variable representing a lower-bounded approximation of x² in epigraph relaxation."
struct EpigraphVariable <: VariableType end
"Tangent-line lower-bound constraints in epigraph relaxation."
struct EpigraphTangentConstraint <: ConstraintType end
"Tangent-line lower-bound expression fL used in the epigraph formulation."
struct EpigraphTangentExpression <: ExpressionType end

"""
Config for epigraph (Q^{L1}) LP-only lower-bound quadratic approximation.

# Fields
- `depth::Int`: sawtooth recursion depth `L`; pure LP, zero binary variables

The worst-case underestimation gap is `Δ²·2^{-2L-4}` — the
sawtooth-epigraph relaxation bound from Beach, Burlacu, Hager,
Hildebrand 2024 (Definition 6, Proposition 2). The construction adds
tangent cuts `z ≥ fʲ(x, g) − 2^{-2j-2}` for `j = 0, 1, …, L` plus the
two boundary tangents to x² at x = x_min and x = x_max. The `j = 0`
cut is the tangent to x² at the segment midpoint; without it, the gap
degrades to `Δ²·2^{-2L-2}`. See
`tolerance_depth(::Type{EpigraphQuadConfig}; …)` to derive `depth` from
a target tolerance.
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int

    EpigraphQuadConfig(; depth::Int) = new(depth)
end

"""
    tolerance_depth(::Type{EpigraphQuadConfig}; tolerance, max_delta)::Int

Smallest epigraph depth `L` whose worst-case underestimation gap on `[a, a+Δ]`
falls within `tolerance`. Inverts the closed-form bound `Δ²·2^{-2L-4} ≤ τ`:
```
L = ⌈(log₂(Δ²/τ) − 4) / 2⌉
```
clamped to `L ≥ 1`.
"""
function tolerance_depth(
    ::Type{EpigraphQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    return _ceil_positive((log2(max_delta^2 / tolerance) - 4) / 2)
end

"""
    _add_quadratic_approx!(::EpigraphQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Create a variable z that lower-bounds x² using the sawtooth-epigraph
Q^{L1} relaxation (Beach, Burlacu, Hager, Hildebrand 2024, Definition 6).

For each (name, t), creates a variable z, the sawtooth aux variables
g_0, …, g_L with `g_j ≤ 2·g_{j-1}` and `g_j ≤ 2·(1 − g_{j-1})`, the
linking constraint `g_0 = (x − x_min)/Δ`, and `L + 1` sawtooth-encoded
tangent cuts `z ≥ fʲ(x, g) − 2^{-2j-2}` for `j = 0, …, L`, plus the two
boundary tangents to x² at x = x_min and x = x_max. Pure LP — zero
binary variables.

Stores affine expressions that lower-bound x² in an `EpigraphExpression` expression container.

The maximum underestimation gap between the tangent envelope and x² is
Δ²·2^{−2·depth−4} where Δ = x_max − x_min.

# Arguments
- `config::EpigraphQuadConfig`: configuration with `depth` field controlling the number of tangent-line breakpoints (2^depth + 1 tangent lines)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: variable type identifier for the approximated variable
"""
function _add_quadratic_approx!(
    config::EpigraphQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op config.depth >= 1
    jump_model = get_jump_model(container)
    g_levels = 0:(config.depth)

    z_var = add_variable_container!(
        container,
        EpigraphVariable,
        C,
        names,
        time_steps;
        meta,
    )
    g_var = add_variable_container!(
        container,
        SawtoothAuxVariable,
        C,
        names,
        g_levels,
        time_steps;
        meta,
    )
    lp_cons = add_constraints_container!(
        container,
        SawtoothLPConstraint,
        C,
        names,
        1:2,
        time_steps;
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SawtoothLinkingConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    fL_expr = add_expression_container!(
        container,
        EpigraphTangentExpression,
        C,
        names,
        time_steps;
        meta,
    )
    tangent_cons = add_constraints_container!(
        container,
        EpigraphTangentConstraint,
        C,
        names,
        1:(config.depth + 3),
        time_steps;
        sparse = true,
        meta,
    )
    result_expr = add_expression_container!(
        container,
        EpigraphExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        b = bounds[i]
        IS.@assert_op b.max > b.min
        delta = b.max - b.min
        z_ub = max(b.min^2, b.max^2)
        x = x_var[name, t]

        # Auxiliary variables g_0,...,g_L ∈ [0, 1]
        for j in g_levels
            g_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothAux_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
        end
        g0 = g_var[name, 0, t]

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_cons[name, t] = JuMP.@constraint(
            jump_model,
            g0 == (x - b.min) / delta,
        )

        # T^L constraints for j = 1,...,L
        for j in 1:(config.depth)
            g_prev = g_var[name, j - 1, t]
            g_curr = g_var[name, j, t]

            # g_j ≤ 2 g_{j-1}
            lp_cons[name, 1, t] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            lp_cons[name, 2, t] =
                JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
        end

        # Create the epigraph variable (bounded from below by tangent cuts)
        z =
            z_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "EpigraphVar_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = z_ub,
            )

        # Sawtooth-epigraph cuts j = 0,...,L (paper Def 6 eq. 15):
        #   z ≥ fʲ(x, g) − 2^{-2j-2}
        # where fʲ(x, g) = x − Σ_{k=1}^{j} 2^{-2k}·g_k (empty sum at j = 0).
        # The j = 0 cut is the tangent to x² at the segment midpoint and is
        # required for the paper's worst-case bound `Δ²·2^{-2L-4}`.
        fL = fL_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 0:(config.depth)
            if j > 0
                add_proportional_to_jump_expression!(
                    fL,
                    g_var[name, j, t],
                    delta * delta * 2.0^(-2j),
                )
            end
            tangent_cons[(name, j + 1, t)] = JuMP.@constraint(
                jump_model,
                z >=
                b.min * (2 * delta * g0 + b.min) - fL + delta^2 * (g0 - 2.0^(-2j - 2))
            )
        end
        # Boundary tangents at x = b.min and x = b.max:
        # tangent to x² at a is z ≥ 2·a·x − a²; substituting x = b.min + Δ·g_0
        # gives the formulas below.
        b_max = b.min + delta
        tangent_cons[name, config.depth + 2, t] = JuMP.@constraint(
            jump_model,
            z >= b.min^2 + 2.0 * b.min * delta * g0
        )
        tangent_cons[name, config.depth + 3, t] = JuMP.@constraint(
            jump_model,
            z >= 2.0 * b_max * b.min - b_max^2 + 2.0 * b_max * delta * g0
        )

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    return result_expr
end
