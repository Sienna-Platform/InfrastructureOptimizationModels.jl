# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end

"Variable result in tightened version."
struct SawtoothTightenedVariable <: VariableType end

"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end

"Bounds tightened sawtooth variable."
struct SawtoothTightenedConstraint <: ConstraintType end

"""
Config for sawtooth MIP quadratic approximation.

# Fields
- `depth::Int`: recursion depth L; uses L binary variables for 2^L + 1 breakpoints.
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound;
  0 to disable (default 0).
"""
struct SawtoothQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
function SawtoothQuadConfig(depth::Int)
    return SawtoothQuadConfig(depth, 0)
end

"""
Pure-JuMP result of `build_quadratic_approx(::SawtoothQuadConfig, ...)`.
"""
struct SawtoothQuadResult{A, G, AL, LC, MC, ZV, TC, EPI} <: QuadraticApproxResult
    approximation::A
    g_var::G
    alpha_var::AL
    link_constraints::LC
    mip_constraints::MC
    tightened_z_var::ZV          # Union{Nothing, DenseAxisArray}
    tightened_constraints::TC    # Union{Nothing, DenseAxisArray}
    epigraph::EPI                # Union{Nothing, EpigraphQuadResult}
end

"""
    build_quadratic_approx(config::SawtoothQuadConfig, model, x, bounds)

PWL approximation of x² with sawtooth tooth functions and L binary variables.
If `config.epigraph_depth > 0`, also builds an epigraph Q^{L1} lower bound and
tightens the approximation: z ≤ x² (sawtooth, upper) and z ≥ epigraph (lower).
"""
function build_quadratic_approx(
    config::SawtoothQuadConfig,
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
    alpha_levels = 1:(config.depth)
    delta = JuMP.Containers.DenseAxisArray(
        [b.max - b.min for b in bounds],
        name_axis,
    )
    x_min_arr = JuMP.Containers.DenseAxisArray(
        [b.min for b in bounds],
        name_axis,
    )

    g_var = JuMP.@variable(
        model,
        [name = name_axis, j = g_levels, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "SawtoothAux",
    )
    alpha_var = JuMP.@variable(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        binary = true,
        base_name = "SawtoothBin",
    )

    link_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        g_var[name, 0, t] == (x[name, t] - x_min_arr[name]) / delta[name],
    )

    # S^L constraints for j = 1..L: 4 inequalities per level.
    # Indexed by (name, j, k, t) with k ∈ 1:4.
    mip_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, alpha_levels, 1:4, time_axis,
    )
    for name in name_axis, j in alpha_levels, t in time_axis
        g_prev = g_var[name, j - 1, t]
        g_curr = g_var[name, j, t]
        a_j = alpha_var[name, j, t]
        mip_cons[name, j, 1, t] =
            JuMP.@constraint(model, g_curr <= 2.0 * g_prev)
        mip_cons[name, j, 2, t] =
            JuMP.@constraint(model, g_curr <= 2.0 * (1.0 - g_prev))
        mip_cons[name, j, 3, t] =
            JuMP.@constraint(model, g_curr >= 2.0 * (g_prev - a_j))
        mip_cons[name, j, 4, t] =
            JuMP.@constraint(model, g_curr >= 2.0 * (a_j - g_prev))
    end

    # x² ≈ x_min² + (2·x_min·δ + δ²)·g₀ − Σ_{j=1..L} δ²·2^{−2j}·g_j
    x_sq_approx = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        x_min_arr[name]^2 +
        (2.0 * x_min_arr[name] * delta[name] + delta[name]^2) * g_var[name, 0, t] -
        sum(delta[name]^2 * 2.0^(-2j) * g_var[name, j, t] for j in alpha_levels)
    )

    if config.epigraph_depth > 0
        epi_result = build_quadratic_approx(
            EpigraphQuadConfig(config.epigraph_depth),
            model,
            x,
            bounds,
        )
        z_min_arr = JuMP.Containers.DenseAxisArray(
            [(b.min <= 0.0 <= b.max) ? 0.0 : min(b.min^2, b.max^2) for b in bounds],
            name_axis,
        )
        z_max_arr = JuMP.Containers.DenseAxisArray(
            [max(b.min^2, b.max^2) for b in bounds],
            name_axis,
        )
        z_var = JuMP.@variable(
            model,
            [name = name_axis, t = time_axis],
            base_name = "TightenedSawtooth",
        )
        for name in name_axis, t in time_axis
            JuMP.set_lower_bound(z_var[name, t], z_min_arr[name])
            JuMP.set_upper_bound(z_var[name, t], z_max_arr[name])
        end
        tight_cons = JuMP.Containers.DenseAxisArray{Any}(
            undef, name_axis, 1:2, time_axis,
        )
        for name in name_axis, t in time_axis
            tight_cons[name, 1, t] =
                JuMP.@constraint(model, z_var[name, t] <= x_sq_approx[name, t])
            tight_cons[name, 2, t] = JuMP.@constraint(
                model,
                z_var[name, t] >= epi_result.approximation[name, t],
            )
        end
        approximation = JuMP.@expression(
            model,
            [name = name_axis, t = time_axis],
            1.0 * z_var[name, t]
        )
        return SawtoothQuadResult(
            approximation,
            g_var,
            alpha_var,
            link_cons,
            mip_cons,
            z_var,
            tight_cons,
            epi_result,
        )
    end

    return SawtoothQuadResult(
        x_sq_approx,
        g_var,
        alpha_var,
        link_cons,
        mip_cons,
        nothing,
        nothing,
        nothing,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::SawtoothQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    g_levels = axes(result.g_var, 2)
    alpha_levels = axes(result.alpha_var, 2)

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

    alpha_target = add_variable_container!(
        container,
        SawtoothBinaryVariable,
        C,
        collect(name_axis),
        alpha_levels,
        time_axis;
        meta,
    )
    for name in name_axis, j in alpha_levels, t in time_axis
        alpha_target[name, j, t] = result.alpha_var[name, j, t]
    end

    link_target = add_constraints_container!(
        container,
        SawtoothLinkingConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        link_target[name, t] = result.link_constraints[name, t]
    end

    mip_target = add_constraints_container!(
        container,
        SawtoothMIPConstraint,
        C,
        collect(name_axis),
        1:4,
        time_axis;
        sparse = true,
        meta,
    )
    for name in name_axis, j in alpha_levels, k in 1:4, t in time_axis
        mip_target[(name, k, t)] = result.mip_constraints[name, j, k, t]
    end

    result_target = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        result_target[name, t] = result.approximation[name, t]
    end

    if result.tightened_z_var !== nothing
        z_target = add_variable_container!(
            container,
            SawtoothTightenedVariable,
            C,
            collect(name_axis),
            time_axis;
            meta,
        )
        for name in name_axis, t in time_axis
            z_target[name, t] = result.tightened_z_var[name, t]
        end
        tight_target = add_constraints_container!(
            container,
            SawtoothTightenedConstraint,
            C,
            collect(name_axis),
            1:2,
            time_axis;
            meta,
        )
        for name in name_axis, k in 1:2, t in time_axis
            tight_target[name, k, t] = result.tightened_constraints[name, k, t]
        end
        register_in_container!(container, C, result.epigraph, meta * "_lb")
    end
    return
end
