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
    build_quadratic_approx(config::ManualSOS2QuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² with manually-enforced SOS2 adjacency
via binary segment-selectors z_j and constraints λ_i ≤ z_{i-1} + z_i (with
boundary cases at i=1 and i=n_points). If `config.pwmcc_segments > 0`,
also adds piecewise McCormick concave cuts per cell.

Returns a NamedTuple with `(approximation, lambda, z_var, link_constraint,
norm_constraint, segment_sum_constraint, adjacency_constraints,
link_expression, norm_expression, segment_sum_expression, pwmcc)`.
"""
function build_quadratic_approx(
    config::ManualSOS2QuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    @assert x_max > x_min
    n_points = config.depth + 1
    n_bins = n_points - 1
    x_bkpts, x_sq_bkpts = _get_breakpoints_for_pwl_function(
        0.0, 1.0, _square; num_segments = config.depth,
    )
    lx = x_max - x_min

    lambda = JuMP.@variable(
        model, [i = 1:n_points],
        lower_bound = 0.0, upper_bound = 1.0,
        base_name = "QuadraticVariable",
    )
    z_var = JuMP.@variable(
        model, [j = 1:n_bins], binary = true, base_name = "ManualSOS2Binary",
    )

    link_expr = JuMP.@expression(
        model, sum(x_bkpts[i] * lambda[i] for i in 1:n_points),
    )
    link_con = JuMP.@constraint(model, (x - x_min) / lx == link_expr)
    norm_expr = JuMP.@expression(model, sum(lambda[i] for i in 1:n_points))
    norm_con = JuMP.@constraint(model, norm_expr == 1.0)
    seg_expr = JuMP.@expression(model, sum(z_var[j] for j in 1:n_bins))
    seg_con = JuMP.@constraint(model, seg_expr == 1)

    adj_first = JuMP.@constraint(model, lambda[1] <= z_var[1])
    adj_interior = JuMP.@constraint(
        model, [i = 2:(n_points - 1)], lambda[i] <= z_var[i - 1] + z_var[i],
    )
    adj_last = JuMP.@constraint(model, lambda[n_points] <= z_var[n_bins])

    adj_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(undef, 1:n_points)
    adj_cons[1] = adj_first
    if n_points >= 3
        for i in 2:(n_points - 1)
            adj_cons[i] = adj_interior[i]
        end
    end
    adj_cons[n_points] = adj_last

    approximation = JuMP.@expression(
        model,
        lx * lx * sum(x_sq_bkpts[i] * lambda[i] for i in 1:n_points) +
        2.0 * x_min * x - x_min * x_min,
    )

    pwmcc = if config.pwmcc_segments > 0
        build_pwmcc_concave_cuts(
            model, x, approximation, x_min, x_max, config.pwmcc_segments,
        )
    else
        nothing
    end

    return (;
        approximation,
        lambda,
        z_var,
        link_constraint = link_con,
        norm_constraint = norm_con,
        segment_sum_constraint = seg_con,
        adjacency_constraints = adj_cons,
        link_expression = link_expr,
        norm_expression = norm_expr,
        segment_sum_expression = seg_expr,
        pwmcc,
    )
end

"""
    add_quadratic_approx!(config::ManualSOS2QuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate manual-SOS2 containers (λ, z, link/norm/seg/adjacency, expressions,
approximation) plus, when `config.pwmcc_segments > 0`, the PWMCC containers
under `meta * "_pwmcc"`. Loop `(name, t)`.
"""
function add_quadratic_approx!(
    config::ManualSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    n_points = config.depth + 1
    n_bins = n_points - 1
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
    end

    model = get_jump_model(container)

    lambda_target = add_variable_container!(
        container, QuadraticVariable, C, name_axis, 1:n_points, time_axis; meta,
    )
    z_target = add_variable_container!(
        container, ManualSOS2BinaryVariable, C, name_axis, 1:n_bins, time_axis; meta,
    )
    link_cons_target = add_constraints_container!(
        container, SOS2LinkingConstraint, C, name_axis, time_axis; meta,
    )
    norm_cons_target = add_constraints_container!(
        container, SOS2NormConstraint, C, name_axis, time_axis; meta,
    )
    seg_cons_target = add_constraints_container!(
        container, ManualSOS2SegmentSelectionConstraint, C, name_axis, time_axis; meta,
    )
    link_expr_target = add_expression_container!(
        container, SOS2LinkingExpression, C, name_axis, time_axis; meta,
    )
    norm_expr_target = add_expression_container!(
        container, SOS2NormExpression, C, name_axis, time_axis; meta,
    )
    seg_expr_target = add_expression_container!(
        container, ManualSOS2SegmentSelectionExpression, C, name_axis, time_axis; meta,
    )
    approx_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    adj_target = add_constraints_container!(
        container, ManualSOS2AdjacencyConstraint, C, name_axis, 1:n_points, time_axis;
        meta,
    )

    use_pwmcc = config.pwmcc_segments > 0
    K = config.pwmcc_segments
    pwmcc_targets = use_pwmcc ?
        _alloc_pwmcc_targets!(container, C, name_axis, time_axis, K, meta * "_pwmcc") :
        nothing

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            for ip in 1:n_points
                lambda_target[name, ip, t] = r.lambda[ip]
                adj_target[name, ip, t] = r.adjacency_constraints[ip]
            end
            for j in 1:n_bins
                z_target[name, j, t] = r.z_var[j]
            end
            link_cons_target[name, t] = r.link_constraint
            norm_cons_target[name, t] = r.norm_constraint
            seg_cons_target[name, t] = r.segment_sum_constraint
            link_expr_target[name, t] = r.link_expression
            norm_expr_target[name, t] = r.norm_expression
            seg_expr_target[name, t] = r.segment_sum_expression
            approx_target[name, t] = r.approximation
            if use_pwmcc
                _write_pwmcc_cell!(pwmcc_targets, name, t, r.pwmcc, K)
            end
        end
    end
    return approx_target
end
