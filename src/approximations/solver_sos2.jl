# Solver-native SOS2 piecewise linear approximation of x² for use in constraints.
# Uses solver-native MOI.SOS2 constraints for adjacency enforcement among λ weights.

"Lambda (λ) convex-combination weight variables for SOS2 quadratic approximation."
struct QuadraticVariable <: SparseVariableType end

"Links x to the weighted sum of breakpoints in SOS2 quadratic approximation."
struct SOS2LinkingConstraint <: ConstraintType end

"Expression for the weighted sum of breakpoints Σ λ_i · x_i linking x to λ variables."
struct SOS2LinkingExpression <: ExpressionType end

"Ensures the sum of λ weights equals 1 in SOS2 quadratic approximation."
struct SOS2NormConstraint <: ConstraintType end

"Expression for the normalization sum Σ λ_i in SOS2 quadratic approximation."
struct SOS2NormExpression <: ExpressionType end

"Solver-native MOI.SOS2 adjacency constraint on lambda variables."
struct SolverSOS2Constraint <: ConstraintType end

"""
Config for solver-native SOS2 quadratic approximation (MOI.SOS2 adjacency).

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1).
- `pwmcc_segments::Int`: number of piecewise McCormick cut partitions;
  0 to disable (default 4).
"""
struct SolverSOS2QuadConfig <: QuadraticApproxConfig
    depth::Int
    pwmcc_segments::Int
end
function SolverSOS2QuadConfig(depth::Int)
    return SolverSOS2QuadConfig(depth, 4)
end

"""
    build_quadratic_approx(config::SolverSOS2QuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² for a single JuMP scalar `x` using
solver-native MOI.SOS2 adjacency. If `config.pwmcc_segments > 0`, also
adds piecewise McCormick concave cuts per cell.

Returns a NamedTuple with `(approximation, lambda, link_constraint,
norm_constraint, sos_constraint, link_expression, norm_expression, pwmcc)`.
"""
function build_quadratic_approx(
    config::SolverSOS2QuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    @assert x_max > x_min
    n_points = config.depth + 1
    x_bkpts, x_sq_bkpts = _get_breakpoints_for_pwl_function(
        0.0, 1.0, _square; num_segments = config.depth,
    )
    lx = x_max - x_min

    lambda = JuMP.@variable(
        model, [i = 1:n_points],
        lower_bound = 0.0, upper_bound = 1.0,
        base_name = "QuadraticVariable",
    )
    link_expr = JuMP.@expression(
        model, sum(x_bkpts[i] * lambda[i] for i in 1:n_points),
    )
    link_con = JuMP.@constraint(model, (x - x_min) / lx == link_expr)
    norm_expr = JuMP.@expression(model, sum(lambda[i] for i in 1:n_points))
    norm_con = JuMP.@constraint(model, norm_expr == 1.0)
    sos_con = JuMP.@constraint(
        model, [lambda[i] for i in 1:n_points] in MOI.SOS2(collect(1:n_points)),
    )
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
        link_constraint = link_con,
        norm_constraint = norm_con,
        sos_constraint = sos_con,
        link_expression = link_expr,
        norm_expression = norm_expr,
        pwmcc,
    )
end

"""
    add_quadratic_approx!(config::SolverSOS2QuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate SOS2 containers (λ, link/norm/sos constraints, expressions,
approximation) plus, when `config.pwmcc_segments > 0`, the full set of
PWMCC containers under `meta * "_pwmcc"`. Loop `(name, t)`.
"""
function add_quadratic_approx!(
    config::SolverSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    n_points = config.depth + 1
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
    end

    model = get_jump_model(container)

    lambda_target = add_variable_container!(
        container, QuadraticVariable, C, name_axis, 1:n_points, time_axis; meta,
    )
    link_cons_target = add_constraints_container!(
        container, SOS2LinkingConstraint, C, name_axis, time_axis; meta,
    )
    norm_cons_target = add_constraints_container!(
        container, SOS2NormConstraint, C, name_axis, time_axis; meta,
    )
    sos_cons_target = add_constraints_container!(
        container, SolverSOS2Constraint, C, name_axis, time_axis; meta,
    )
    link_expr_target = add_expression_container!(
        container, SOS2LinkingExpression, C, name_axis, time_axis; meta,
    )
    norm_expr_target = add_expression_container!(
        container, SOS2NormExpression, C, name_axis, time_axis; meta,
    )
    approx_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
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
            end
            link_cons_target[name, t] = r.link_constraint
            norm_cons_target[name, t] = r.norm_constraint
            sos_cons_target[name, t] = r.sos_constraint
            link_expr_target[name, t] = r.link_expression
            norm_expr_target[name, t] = r.norm_expression
            approx_target[name, t] = r.approximation
            if use_pwmcc
                _write_pwmcc_cell!(pwmcc_targets, name, t, r.pwmcc, K)
            end
        end
    end
    return approx_target
end
