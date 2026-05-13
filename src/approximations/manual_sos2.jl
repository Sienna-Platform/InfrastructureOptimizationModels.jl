# SOS2 piecewise linear approximation of x² with manually-implemented adjacency
# via binary segment-selector variables. Useful when the solver does not
# natively support MOI.SOS2.

"Binary segment-selection variables (z) for manual SOS2 quadratic approximation."
struct ManualSOS2BinaryVariable <: SparseVariableType end

"Ensures exactly one segment is active (∑ z_j = 1) in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end

"Expression for the segment selection sum Σ z_j in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionExpression <: ExpressionType end

"Links active segment to lambda variables."
struct ManualSOS2AdjacencyConstraint <: ConstraintType end

"""
Config for manual binary-variable SOS2 quadratic approximation.

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1).
- `pwmcc_segments::Int`: number of piecewise McCormick cut partitions;
  0 to disable (default 4).
"""
struct ManualSOS2QuadConfig <: QuadraticApproxConfig
    depth::Int
    pwmcc_segments::Int
end
function ManualSOS2QuadConfig(depth::Int)
    return ManualSOS2QuadConfig(depth, 4)
end

"""
Pure-JuMP result of `build_quadratic_approx(::ManualSOS2QuadConfig, ...)`.
"""
struct ManualSOS2QuadResult{A, L, Z, LC, NC, ZSUM, AC, LE, NE, ZE, PWMCC} <:
       QuadraticApproxResult
    approximation::A
    lambda::L
    z_var::Z
    link_constraints::LC
    norm_constraints::NC
    segment_sum_constraints::ZSUM
    adjacency_constraints::AC
    link_expressions::LE
    norm_expressions::NE
    segment_sum_expressions::ZE
    pwmcc::PWMCC
end

"""
    build_quadratic_approx(config::ManualSOS2QuadConfig, model, x, bounds)

PWL approximation of x² with manually-enforced SOS2 adjacency via binary
segment-selectors z_j and adjacency constraints λ_i ≤ z_{i-1} + z_i. If
`config.pwmcc_segments > 0`, also adds piecewise McCormick concave cuts.
"""
function build_quadratic_approx(
    config::ManualSOS2QuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.max > b.min
    end
    n_points = config.depth + 1
    n_bins = n_points - 1
    x_bkpts, x_sq_bkpts = _get_breakpoints_for_pwl_function(
        0.0,
        1.0,
        _square;
        num_segments = config.depth,
    )

    lx = JuMP.Containers.DenseAxisArray(
        [b.max - b.min for b in bounds],
        name_axis,
    )
    x_min = JuMP.Containers.DenseAxisArray(
        [b.min for b in bounds],
        name_axis,
    )

    lambda = JuMP.@variable(
        model,
        [name = name_axis, i = 1:n_points, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "QuadraticVariable",
    )
    z_var = JuMP.@variable(
        model,
        [name = name_axis, j = 1:n_bins, t = time_axis],
        binary = true,
        base_name = "ManualSOS2Binary",
    )

    link_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(x_bkpts[i] * lambda[name, i, t] for i in 1:n_points)
    )
    link_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        (x[name, t] - x_min[name]) / lx[name] == link_expr[name, t]
    )
    norm_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(lambda[name, i, t] for i in 1:n_points)
    )
    norm_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        norm_expr[name, t] == 1.0
    )
    seg_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(z_var[name, j, t] for j in 1:n_bins)
    )
    seg_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        seg_expr[name, t] == 1
    )

    # Adjacency constraints: λ_i ≤ z_{i-1} + z_i with boundary cases.
    # Store as a 3D container keyed by (name, i, t) where i ∈ 1:n_points.
    adj_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, 1:n_points, time_axis,
    )
    for name in name_axis, t in time_axis
        adj_cons[name, 1, t] =
            JuMP.@constraint(model, lambda[name, 1, t] <= z_var[name, 1, t])
        for i in 2:(n_points - 1)
            adj_cons[name, i, t] = JuMP.@constraint(
                model,
                lambda[name, i, t] <= z_var[name, i - 1, t] + z_var[name, i, t],
            )
        end
        adj_cons[name, n_points, t] = JuMP.@constraint(
            model,
            lambda[name, n_points, t] <= z_var[name, n_bins, t],
        )
    end

    approximation = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        lx[name] * lx[name] *
        sum(x_sq_bkpts[i] * lambda[name, i, t] for i in 1:n_points) +
        2.0 * x_min[name] * x[name, t] - x_min[name] * x_min[name]
    )

    pwmcc = if config.pwmcc_segments > 0
        build_pwmcc_concave_cuts(model, x, approximation, bounds, config.pwmcc_segments)
    else
        nothing
    end

    return ManualSOS2QuadResult(
        approximation,
        lambda,
        z_var,
        link_cons,
        norm_cons,
        seg_cons,
        adj_cons,
        link_expr,
        norm_expr,
        seg_expr,
        pwmcc,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::ManualSOS2QuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    n_points_axis = axes(result.lambda, 2)
    n_bins_axis = axes(result.z_var, 2)

    lambda_target = add_variable_container!(
        container,
        QuadraticVariable,
        C,
        collect(name_axis),
        n_points_axis,
        time_axis;
        meta,
    )
    for name in name_axis, i in n_points_axis, t in time_axis
        lambda_target[name, i, t] = result.lambda[name, i, t]
    end

    z_target = add_variable_container!(
        container,
        ManualSOS2BinaryVariable,
        C,
        collect(name_axis),
        n_bins_axis,
        time_axis;
        meta,
    )
    for name in name_axis, j in n_bins_axis, t in time_axis
        z_target[name, j, t] = result.z_var[name, j, t]
    end

    link_cons_target = add_constraints_container!(
        container,
        SOS2LinkingConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    norm_cons_target = add_constraints_container!(
        container,
        SOS2NormConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    seg_cons_target = add_constraints_container!(
        container,
        ManualSOS2SegmentSelectionConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    link_expr_target = add_expression_container!(
        container,
        SOS2LinkingExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    norm_expr_target = add_expression_container!(
        container,
        SOS2NormExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    seg_expr_target = add_expression_container!(
        container,
        ManualSOS2SegmentSelectionExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    result_target = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        link_cons_target[name, t] = result.link_constraints[name, t]
        norm_cons_target[name, t] = result.norm_constraints[name, t]
        seg_cons_target[name, t] = result.segment_sum_constraints[name, t]
        link_expr_target[name, t] = result.link_expressions[name, t]
        norm_expr_target[name, t] = result.norm_expressions[name, t]
        seg_expr_target[name, t] = result.segment_sum_expressions[name, t]
        result_target[name, t] = result.approximation[name, t]
    end

    adj_target = add_constraints_container!(
        container,
        ManualSOS2AdjacencyConstraint,
        C,
        collect(name_axis),
        n_points_axis,
        time_axis;
        meta,
    )
    for name in name_axis, i in n_points_axis, t in time_axis
        adj_target[name, i, t] = result.adjacency_constraints[name, i, t]
    end

    if result.pwmcc !== nothing
        register_pwmcc!(container, C, result.pwmcc, meta * "_pwmcc")
    end
    return
end
