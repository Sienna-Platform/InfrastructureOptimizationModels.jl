# Epigraph (Q^{L1}) LP-only lower bound for x² using tangent-line cuts.
# Pure LP — zero binary variables. Creates a variable z that lower-bounds
# x² (approximately) bounded from below by supporting hyperplanes of the
# parabola.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024), Q^{L1} relaxation.

"Expression container for epigraph quadratic approximation results."
struct EpigraphExpression <: ExpressionType end

"Auxiliary continuous variables (g₀, …, g_L) for tooth-based PWL approximations."
struct SawtoothAuxVariable <: VariableType end

"LP relaxation constraints (g_j ≤ 2g_{j-1}, g_j ≤ 2(1−g_{j-1}))."
struct SawtoothLPConstraint <: ConstraintType end

"Links g₀ to the normalized x value."
struct SawtoothLinkingConstraint <: ConstraintType end

"Variable representing a lower-bounded approximation of x² in epigraph relaxation."
struct EpigraphVariable <: VariableType end

"Tangent-line lower-bound constraints in epigraph relaxation."
struct EpigraphTangentConstraint <: ConstraintType end

"Tangent-line lower-bound expression fL used in the epigraph formulation."
struct EpigraphTangentExpression <: ExpressionType end

"""
Config for epigraph (Q^{L1}) LP-only lower-bound quadratic approximation.

# Fields
- `depth::Int`: number of tangent-line breakpoints (2^depth + 1 tangent lines);
  pure LP, zero binary variables.
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int
end

"""
Pure-JuMP result of `build_quadratic_approx(::EpigraphQuadConfig, ...)`.
"""
struct EpigraphQuadResult{A, Z, G, LP, LC, FL, TC} <: QuadraticApproxResult
    approximation::A
    z_var::Z
    g_var::G
    lp_constraints::LP
    link_constraints::LC
    tangent_expressions::FL
    tangent_constraints::TC
end

"""
    build_quadratic_approx(config::EpigraphQuadConfig, model, x, bounds)

LP-only lower bound for x² via 2^depth + 1 tangent-line cuts on the
parabola at uniformly spaced breakpoints in [x_min, x_max].
"""
function build_quadratic_approx(
    config::EpigraphQuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    IS.@assert_op config.depth >= 1
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.max > b.min
    end

    g_levels = 0:(config.depth)
    delta = JuMP.Containers.DenseAxisArray(
        [b.max - b.min for b in bounds],
        name_axis,
    )
    x_min_arr = JuMP.Containers.DenseAxisArray(
        [b.min for b in bounds],
        name_axis,
    )
    z_ub_arr = JuMP.Containers.DenseAxisArray(
        [max(b.min^2, b.max^2) for b in bounds],
        name_axis,
    )

    g_var = JuMP.@variable(
        model,
        [name = name_axis, j = g_levels, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "SawtoothAux",
    )

    link_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        g_var[name, 0, t] == (x[name, t] - x_min_arr[name]) / delta[name]
    )

    # T^L constraints: g_j ≤ 2 g_{j-1} and g_j ≤ 2(1 − g_{j-1}) for j = 1..L.
    # Indexed by (name, j, k, t) with k ∈ 1:2.
    lp_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, 1:(config.depth), 1:2, time_axis,
    )
    for name in name_axis, j in 1:(config.depth), t in time_axis
        lp_cons[name, j, 1, t] = JuMP.@constraint(
            model,
            g_var[name, j, t] <= 2.0 * g_var[name, j - 1, t],
        )
        lp_cons[name, j, 2, t] = JuMP.@constraint(
            model,
            g_var[name, j, t] <= 2.0 * (1.0 - g_var[name, j - 1, t]),
        )
    end

    # z is bounded below by x² via tangent cuts; pure-LP variable.
    z_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        lower_bound = 0.0,
        base_name = "EpigraphVar",
    )
    for name in name_axis, t in time_axis
        JuMP.set_upper_bound(z_var[name, t], z_ub_arr[name])
    end

    # fL[j] = Σ_{k=1..j} δ² · 2^{−2k} · g_k  (partial sum used in the j-th tangent cut).
    # Built as a 2D container with time axis only (full-depth sum); the partial
    # sums for each tangent constraint are formed inline below.
    fL_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(delta[name]^2 * 2.0^(-2j) * g_var[name, j, t] for j in 1:(config.depth))
    )

    # Tangent-line cuts: z ≥ 0, z ≥ 2·x_min + 2·δ·g₀ − 1, plus depth more cuts
    # at j = 1..L of the form z ≥ x_min·(2·δ·g₀ + x_min) − fL[j] + δ²·(g₀ − 2^{−2j−2}).
    tangent_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, 1:(config.depth + 2), time_axis,
    )
    for name in name_axis, t in time_axis
        tangent_cons[name, 1, t] = JuMP.@constraint(model, z_var[name, t] >= 0)
        tangent_cons[name, 2, t] = JuMP.@constraint(
            model,
            z_var[name, t] >=
            2.0 * x_min_arr[name] - 1.0 + 2.0 * delta[name] * g_var[name, 0, t],
        )
        for j in 1:(config.depth)
            tangent_cons[name, j + 2, t] = JuMP.@constraint(
                model,
                z_var[name, t] >=
                x_min_arr[name] *
                (2.0 * delta[name] * g_var[name, 0, t] + x_min_arr[name]) -
                sum(
                    delta[name]^2 * 2.0^(-2k) * g_var[name, k, t] for k in 1:j
                ) +
                delta[name]^2 * (g_var[name, 0, t] - 2.0^(-2j - 2)),
            )
        end
    end

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        1.0 * z_var[name, t]
    )

    return EpigraphQuadResult(
        approximation,
        z_var,
        g_var,
        lp_cons,
        link_cons,
        fL_expr,
        tangent_cons,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::EpigraphQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    g_levels = axes(result.g_var, 2)

    z_target = add_variable_container!(
        container,
        EpigraphVariable,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        z_target[name, t] = result.z_var[name, t]
    end

    g_target = add_variable_container!(
        container,
        SawtoothAuxVariable,
        C,
        collect(name_axis),
        g_levels,
        time_axis;
        meta,
    )
    for name in name_axis, j in g_levels, t in time_axis
        g_target[name, j, t] = result.g_var[name, j, t]
    end

    link_target = add_constraints_container!(
        container,
        SawtoothLinkingConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    fL_target = add_expression_container!(
        container,
        EpigraphTangentExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    result_target = add_expression_container!(
        container,
        EpigraphExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        link_target[name, t] = result.link_constraints[name, t]
        fL_target[name, t] = result.tangent_expressions[name, t]
        result_target[name, t] = result.approximation[name, t]
    end

    lp_target = add_constraints_container!(
        container,
        SawtoothLPConstraint,
        C,
        collect(name_axis),
        1:2,
        time_axis;
        meta,
    )
    lp_lvl_axis = axes(result.lp_constraints, 2)
    for name in name_axis, j in lp_lvl_axis, k in 1:2, t in time_axis
        lp_target[name, k, t] = result.lp_constraints[name, j, k, t]
    end

    tangent_axis = axes(result.tangent_constraints, 2)
    tangent_target = add_constraints_container!(
        container,
        EpigraphTangentConstraint,
        C,
        collect(name_axis),
        tangent_axis,
        time_axis;
        sparse = true,
        meta,
    )
    for name in name_axis, k in tangent_axis, t in time_axis
        tangent_target[(name, k, t)] = result.tangent_constraints[name, k, t]
    end
    return
end
