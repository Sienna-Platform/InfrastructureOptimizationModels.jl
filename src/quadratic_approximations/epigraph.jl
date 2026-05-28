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

Construct with either `depth` directly or `(tolerance, max_delta)`; the latter
inverts the closed-form bound `Δ²·2^{-2·depth-2}` to pick the smallest `depth`
whose worst-case underestimation gap is within `tolerance`.

# Fields
- `depth::Int`: number of tangent-line breakpoints (2^depth + 1 tangent lines); pure LP, zero binary variables
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int

    function EpigraphQuadConfig(;
        depth::Union{Int, Nothing} = nothing,
        tolerance::Union{Float64, Nothing} = nothing,
        max_delta::Union{Float64, Nothing} = nothing,
    )
        if depth !== nothing
            return new(depth)
        elseif tolerance !== nothing && max_delta !== nothing
            return new(max(1, ceil(Int, (log2(max_delta^2 / tolerance) - 2) / 2)))
        else
            error(
                "EpigraphQuadConfig requires either `depth` or both `tolerance` and `max_delta`.",
            )
        end
    end
end

"""
    _add_quadratic_approx!(::EpigraphQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Create a variable z that lower-bounds x² using tangent-line cuts (Q^{L1} relaxation).

For each (name, t), creates a variable z and adds 2^depth + 1 tangent-line
constraints of the form `z ≥ 2·aₖ·x − aₖ²` at uniformly spaced breakpoints
aₖ = x_min + k·Δ/2^depth for k = 0,…,2^depth. Pure LP — zero binary variables.

Stores affine expressions that lower-bound x² in an `EpigraphExpression` expression container.

The maximum underestimation gap between the tangent envelope and x² is
Δ²·2^{−2·depth−2} where Δ = x_max − x_min.

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
        1:(config.depth + 2),
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

        fL = fL_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:(config.depth)
            add_proportional_to_jump_expression!(
                fL,
                g_var[name, j, t],
                delta * delta * 2.0^(-2j),
            )
            tangent_cons[(name, j + 1, t)] = JuMP.@constraint(
                jump_model,
                z >=
                b.min * (2 * delta * g0 + b.min) - fL + delta^2 * (g0 - 2.0^(-2j - 2))
            )
        end
        tangent_cons[name, 1, t] = JuMP.@constraint(jump_model, z >= 0)
        tangent_cons[name, config.depth + 1, t] = JuMP.@constraint(
            jump_model,
            z >= 2.0 * b.min - 1.0 + 2.0 * delta * g0
        )

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    return result_expr
end
