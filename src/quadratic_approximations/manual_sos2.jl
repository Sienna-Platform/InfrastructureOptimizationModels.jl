# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses manually-implemented SOS2 adjacency via binary variables and linear constraints.

"Binary segment-selection variables (z) for manual SOS2 quadratic approximation."
struct ManualSOS2BinaryVariable <: SparseVariableType end
"Ensures exactly one segment is active (∑zⱼ = 1) in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end
"Expression for the segment selection sum Σ z_j in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionExpression <: ExpressionType end
"Links active segment to lambda variables."
struct ManualSOS2AdjacencyConstraint <: ConstraintType end

"""
Config for manual binary-variable SOS2 quadratic approximation.

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1)
- `pwmcc_segments::Int`: number of piecewise McCormick cut partitions; 0 to disable (default 0)

The worst-case PWL overestimation gap is `Δ²/(4·depth²)`. `pwmcc_segments` is
an LP-relaxation tightener (adds piecewise-McCormick cuts on the concave
relaxation surface); it does **not** change the MIP-optimal worst-case error,
only the LP relaxation given to branch-and-bound.

**Constraint:** `pwmcc_segments ≤ depth`. The PWMCC chord cuts treat `q` as if
`q = x²`, but with SoS2 PWL `q` is the coarse-PWL-chord-of-x² which over-estimates
x². When PWMCC sub-segments are finer than PWL segments, the PWMCC chord (on a
fine sub-segment) is smaller than the PWL chord (on the wider segment), and the
cut chops off MIP-feasible solutions — the model goes infeasible. The constructor
enforces this. See `tolerance_depth(::Type{ManualSOS2QuadConfig}; …)` to derive
`depth` from a target tolerance.
"""
struct ManualSOS2QuadConfig <: QuadraticApproxConfig
    depth::Int
    pwmcc_segments::Int

    function ManualSOS2QuadConfig(; depth::Int, pwmcc_segments::Int = 0)
        if pwmcc_segments > depth
            throw(
                ArgumentError(
                    "ManualSOS2QuadConfig requires pwmcc_segments ≤ depth " *
                    "(got pwmcc_segments=$(pwmcc_segments), depth=$(depth)); " *
                    "finer PWMCC sub-segments chop off MIP-feasible PWL solutions.",
                ),
            )
        end
        return new(depth, pwmcc_segments)
    end
end

"""
    tolerance_depth(::Type{ManualSOS2QuadConfig}; tolerance, max_delta)::Int

Smallest SOS2 segment count `d` whose worst-case PWL gap on `[a, a+Δ]` falls
within `tolerance`. Inverts `Δ²/(4·d²) ≤ τ`:
```
d = ⌈Δ / (2·√τ)⌉
```
clamped to `d ≥ 1`. `pwmcc_segments` does not enter the error bound, so it is
left to the constructor.
"""
function tolerance_depth(
    ::Type{ManualSOS2QuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive(max_delta / (2 * sqrt(tolerance)))
end

"""
    _add_quadratic_approx!(config::ManualSOS2QuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using a piecewise linear function with manually-implemented SOS2 constraints.

Creates lambda (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and manual adjacency constraints using binary variables,
and stores affine expressions approximating x² in a `QuadraticExpression`
expression container.

# Arguments
- `config::ManualSOS2QuadConfig`: configuration with `depth` (number of PWL segments) and `pwmcc_segments` (PWMCC cut partitions; 0 to disable, default 4)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_quadratic_approx!(
    config::ManualSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(
            0.0,
            1.0,
            _square;
            num_segments = config.depth,
        )
    n_points = config.depth + 1
    n_bins = n_points - 1
    jump_model = get_jump_model(container)

    # Create all containers upfront
    lambda_container = add_variable_container!(
        container,
        QuadraticVariable,
        C,
        names,
        1:n_points,
        time_steps;
        meta,
    )
    z_container = add_variable_container!(
        container,
        ManualSOS2BinaryVariable,
        C,
        names,
        1:n_bins,
        time_steps;
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SOS2LinkingConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    link_expr = add_expression_container!(
        container,
        SOS2LinkingExpression,
        C,
        names,
        time_steps;
        meta,
    )
    norm_cons = add_constraints_container!(
        container,
        SOS2NormConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    norm_expr = add_expression_container!(
        container,
        SOS2NormExpression,
        C,
        names,
        time_steps;
        meta,
    )
    seg_cons = add_constraints_container!(
        container,
        ManualSOS2SegmentSelectionConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    seg_expr = add_expression_container!(
        container,
        ManualSOS2SegmentSelectionExpression,
        C,
        names,
        time_steps;
        meta,
    )
    adj_cons = add_constraints_container!(
        container,
        ManualSOS2AdjacencyConstraint,
        C,
        names,
        1:n_points,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        b = bounds[i]
        IS.@assert_op b.max > b.min
        lx = b.max - b.min
        x = x_var[name, t]

        # Create lambda variables: λ_i ∈ [0, 1]
        lambda = Vector{JuMP.VariableRef}(undef, n_points)
        for i in 1:n_points
            lambda[i] =
                lambda_container[name, i, t] = JuMP.@variable(
                    jump_model,
                    base_name = "QuadraticVariable_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                )
        end

        # x = Σ λ_i * x_i
        link = link_expr[name, t] = JuMP.AffExpr(0.0)
        for i in eachindex(x_bkpts)
            add_proportional_to_jump_expression!(link, lambda[i], x_bkpts[i])
        end
        link_cons[name, t] = JuMP.@constraint(jump_model, (x - b.min) / lx == link)

        # Σ λ_i = 1
        norm = norm_expr[name, t] = JuMP.AffExpr(0.0)
        for l in lambda
            add_proportional_to_jump_expression!(norm, l, 1.0)
        end
        norm_cons[name, t] = JuMP.@constraint(jump_model, norm == 1.0)

        # Create binary segment-selection variables z_j
        z_vars = Vector{JuMP.VariableRef}(undef, n_bins)
        for j in 1:n_bins
            z_vars[j] =
                z_container[name, j, t] = JuMP.@variable(
                    jump_model,
                    base_name = "ManualSOS2Binary_$(C)_{$(name), $(j), $(t)}",
                    binary = true,
                )
        end

        # Σ z_j = 1 (segment selection)
        seg = seg_expr[name, t] = JuMP.AffExpr(0.0)
        for z in z_vars
            add_proportional_to_jump_expression!(seg, z, 1.0)
        end
        seg_cons[name, t] = JuMP.@constraint(jump_model, seg == 1)

        # Adjacency constraints: λ_i ≤ z_{i-1} + z_i (with boundary cases)
        # λ_1 ≤ z_1
        adj_cons[name, 1, t] = JuMP.@constraint(jump_model, lambda[1] <= z_vars[1])
        # λ_i ≤ z_{i-1} + z_i for i = 2..n-1
        for i in 2:(n_points - 1)
            adj_cons[name, i + 1, t] =
                JuMP.@constraint(jump_model, lambda[i] <= z_vars[i - 1] + z_vars[i])
        end
        # λ_n ≤ z_{n-1}
        adj_cons[name, n_points, t] =
            JuMP.@constraint(jump_model, lambda[n_points] <= z_vars[n_bins])

        # Build x̂² = Σ λ_i * x_i² as an affine expression
        x_hat_sq = JuMP.AffExpr(0.0)
        for i in 1:n_points
            add_proportional_to_jump_expression!(x_hat_sq, lambda[i], x_sq_bkpts[i])
        end
        x_sq = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(x_sq, x_hat_sq, lx * lx)
        add_proportional_to_jump_expression!(x_sq, x, 2 * b.min)
        add_constant_to_jump_expression!(x_sq, -b.min * b.min)
        result_expr[name, t] = x_sq
    end

    if config.pwmcc_segments > 0
        _add_pwmcc_concave_cuts!(
            container, C, names, time_steps,
            x_var, result_expr, bounds,
            config.pwmcc_segments, meta * "_pwmcc",
        )
    end

    return result_expr
end
