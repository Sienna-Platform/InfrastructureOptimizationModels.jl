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

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_quadratic_approx(config::ManualSOS2QuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² with manually-enforced SOS2 adjacency
via binary segment-selectors z_j and constraints λ_i ≤ z_{i-1} + z_i (with
boundary cases at i=1 and i=n_points). If `config.pwmcc_segments > 0`,
also adds piecewise McCormick concave cuts.

Returns a NamedTuple:
- `approximation`             :: scalar AffExpr
- `lambda`                    :: DenseAxisArray{VariableRef, 1} over `1:n_points`
- `z_var`                     :: DenseAxisArray{VariableRef, 1} over `1:n_bins` (binary)
- `link_constraint`           :: scalar
- `norm_constraint`           :: scalar
- `segment_sum_constraint`    :: scalar
- `adjacency_constraints`     :: DenseAxisArray{Constraint, 1} over `1:n_points`
- `link_expression`           :: scalar AffExpr
- `norm_expression`           :: scalar AffExpr
- `segment_sum_expression`    :: scalar AffExpr
- `pwmcc`                     :: `nothing` or NamedTuple from scalar PWMCC build
"""
function build_quadratic_approx(
    config::ManualSOS2QuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    IS.@assert_op x_max > x_min
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
        model, [j = 1:n_bins],
        binary = true,
        base_name = "ManualSOS2Binary",
    )

    link_expr = JuMP.@expression(
        model, sum(x_bkpts[i] * lambda[i] for i in 1:n_points),
    )
    link_con = JuMP.@constraint(model, (x - x_min) / lx == link_expr)
    norm_expr = JuMP.@expression(model, sum(lambda[i] for i in 1:n_points))
    norm_con = JuMP.@constraint(model, norm_expr == 1.0)
    seg_expr = JuMP.@expression(model, sum(z_var[j] for j in 1:n_bins))
    seg_con = JuMP.@constraint(model, seg_expr == 1)

    # Adjacency: λ_1 ≤ z_1, λ_n ≤ z_{n-1}, and λ_i ≤ z_{i-1}+z_i for interior.
    adj_first = JuMP.@constraint(model, lambda[1] <= z_var[1])
    adj_interior = JuMP.@constraint(
        model, [i = 2:(n_points - 1)], lambda[i] <= z_var[i - 1] + z_var[i],
    )
    adj_last = JuMP.@constraint(model, lambda[n_points] <= z_var[n_bins])

    adj_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
        undef, 1:n_points,
    )
    adj_cons[1] = adj_first
    if n_points >= 3
        @views adj_cons.data[2:(n_points - 1)] .= adj_interior.data
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

# --- IOM adapter (allocate, loop, write) ---

"""
    add_quadratic_approx!(config::ManualSOS2QuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate manual-SOS2 containers (λ, z, link/norm/seg/adjacency, expressions,
approximation) plus, when `config.pwmcc_segments > 0`, the PWMCC containers
under `meta * "_pwmcc"`. Loop `(name, t)`, call scalar build per cell, write.
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
    IS.@assert_op length(name_axis) == length(x_bounds)
    for b in x_bounds
        IS.@assert_op b.max > b.min
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
    local pw_delta_target, pw_vd_target, pw_selector_target, pw_linking_target,
        pw_interval_lb_target, pw_interval_ub_target,
        pw_chord_target, pw_tangent_l_target, pw_tangent_r_target
    if use_pwmcc
        pwm_meta = meta * "_pwmcc"
        pw_delta_target = add_variable_container!(
            container, PiecewiseMcCormickBinary, C, name_axis, 1:K, time_axis;
            meta = pwm_meta,
        )
        pw_vd_target = add_variable_container!(
            container, PiecewiseMcCormickDisaggregated, C, name_axis, 1:K, time_axis;
            meta = pwm_meta,
        )
        pw_selector_target = add_constraints_container!(
            container, PiecewiseMcCormickSelectorSum, C, name_axis, time_axis;
            meta = pwm_meta,
        )
        pw_linking_target = add_constraints_container!(
            container, PiecewiseMcCormickLinking, C, name_axis, time_axis;
            meta = pwm_meta,
        )
        pw_interval_lb_target = add_constraints_container!(
            container, PiecewiseMcCormickIntervalLB, C, name_axis, 1:K, time_axis;
            meta = pwm_meta,
        )
        pw_interval_ub_target = add_constraints_container!(
            container, PiecewiseMcCormickIntervalUB, C, name_axis, 1:K, time_axis;
            meta = pwm_meta,
        )
        pw_chord_target = add_constraints_container!(
            container, PiecewiseMcCormickChordUB, C, name_axis, time_axis;
            meta = pwm_meta,
        )
        pw_tangent_l_target = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBL, C, name_axis, time_axis;
            meta = pwm_meta,
        )
        pw_tangent_r_target = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBR, C, name_axis, time_axis;
            meta = pwm_meta,
        )
    end

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
                pw = r.pwmcc
                for k in 1:K
                    pw_delta_target[name, k, t] = pw.delta_var[k]
                    pw_vd_target[name, k, t] = pw.vd_var[k]
                    pw_interval_lb_target[name, k, t] = pw.interval_lb_constraints[k]
                    pw_interval_ub_target[name, k, t] = pw.interval_ub_constraints[k]
                end
                pw_selector_target[name, t] = pw.selector_constraint
                pw_linking_target[name, t] = pw.linking_constraint
                pw_chord_target[name, t] = pw.chord_ub_constraint
                pw_tangent_l_target[name, t] = pw.tangent_lb_l_constraint
                pw_tangent_r_target[name, t] = pw.tangent_lb_r_constraint
            end
        end
    end
    return approx_target
end

# --- Legacy result + vectorized build + register (kept until callers
# migrate; removed in sweep) ---

"""
Pure-JuMP result of legacy vectorized `build_quadratic_approx(::ManualSOS2QuadConfig, ...)`.
"""
struct ManualSOS2QuadResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    L <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    Z <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    LC <: JuMP.Containers.DenseAxisArray,
    NC <: JuMP.Containers.DenseAxisArray,
    ZSUM <: JuMP.Containers.DenseAxisArray,
    AC <: JuMP.Containers.DenseAxisArray,
    LE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    NE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    ZE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    PWMCC <: Union{Nothing, PWMCCResult},
} <: QuadraticApproxResult
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
        0.0, 1.0, _square; num_segments = config.depth,
    )

    lx = JuMP.Containers.DenseAxisArray([b.max - b.min for b in bounds], name_axis)
    x_min = JuMP.Containers.DenseAxisArray([b.min for b in bounds], name_axis)

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

    adj_first = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        lambda[name, 1, t] <= z_var[name, 1, t],
    )
    adj_interior = JuMP.@constraint(
        model,
        [name = name_axis, i = 2:(n_points - 1), t = time_axis],
        lambda[name, i, t] <= z_var[name, i - 1, t] + z_var[name, i, t],
    )
    adj_last = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        lambda[name, n_points, t] <= z_var[name, n_bins, t],
    )
    adj_cons = JuMP.Containers.DenseAxisArray{eltype(adj_first.data)}(
        undef, name_axis, 1:n_points, time_axis,
    )
    @views adj_cons.data[:, 1, :] .= adj_first.data
    if n_points >= 3
        @views adj_cons.data[:, 2:(n_points - 1), :] .= adj_interior.data
    end
    @views adj_cons.data[:, n_points, :] .= adj_last.data

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
        approximation, lambda, z_var,
        link_cons, norm_cons, seg_cons, adj_cons,
        link_expr, norm_expr, seg_expr, pwmcc,
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
        container, QuadraticVariable, C, name_axis, n_points_axis, time_axis; meta,
    )
    lambda_target.data .= result.lambda.data

    z_target = add_variable_container!(
        container, ManualSOS2BinaryVariable, C, name_axis, n_bins_axis, time_axis; meta,
    )
    z_target.data .= result.z_var.data

    link_cons_target = add_constraints_container!(
        container, SOS2LinkingConstraint, C, name_axis, time_axis; meta,
    )
    link_cons_target.data .= result.link_constraints.data

    norm_cons_target = add_constraints_container!(
        container, SOS2NormConstraint, C, name_axis, time_axis; meta,
    )
    norm_cons_target.data .= result.norm_constraints.data

    seg_cons_target = add_constraints_container!(
        container, ManualSOS2SegmentSelectionConstraint, C, name_axis, time_axis; meta,
    )
    seg_cons_target.data .= result.segment_sum_constraints.data

    link_expr_target = add_expression_container!(
        container, SOS2LinkingExpression, C, name_axis, time_axis; meta,
    )
    link_expr_target.data .= result.link_expressions.data

    norm_expr_target = add_expression_container!(
        container, SOS2NormExpression, C, name_axis, time_axis; meta,
    )
    norm_expr_target.data .= result.norm_expressions.data

    seg_expr_target = add_expression_container!(
        container, ManualSOS2SegmentSelectionExpression, C, name_axis, time_axis; meta,
    )
    seg_expr_target.data .= result.segment_sum_expressions.data

    result_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    adj_target = add_constraints_container!(
        container, ManualSOS2AdjacencyConstraint, C, name_axis, n_points_axis,
        time_axis; meta,
    )
    adj_target.data .= result.adjacency_constraints.data

    register_pwmcc!(container, C, result.pwmcc, meta * "_pwmcc")
    return
end
