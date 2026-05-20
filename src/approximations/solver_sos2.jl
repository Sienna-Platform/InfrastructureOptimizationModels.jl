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

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_quadratic_approx(config::SolverSOS2QuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² for a single JuMP scalar `x` using
solver-native MOI.SOS2 adjacency. If `config.pwmcc_segments > 0`, also
adds piecewise McCormick concave cuts (per cell).

Returns a NamedTuple:
- `approximation`     :: scalar AffExpr (the PWL estimate of x²)
- `lambda`            :: DenseAxisArray{VariableRef, 1} over `1:n_points`
- `link_constraint`   :: scalar
- `norm_constraint`   :: scalar
- `sos_constraint`    :: scalar (MOI.SOS2 adjacency)
- `link_expression`   :: scalar AffExpr (Σ x_bkpts[i] · λ_i)
- `norm_expression`   :: scalar AffExpr (Σ λ_i)
- `pwmcc`             :: `nothing` or NamedTuple from scalar `build_pwmcc_concave_cuts`
"""
function build_quadratic_approx(
    config::SolverSOS2QuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    IS.@assert_op x_max > x_min
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
    # x² = lx² · Σ λ_i · x_bkpts[i]² + 2·x_min·x − x_min²
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

# --- IOM adapter (allocate, loop, write) ---

"""
    add_quadratic_approx!(config::SolverSOS2QuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate SOS2 containers (λ, link/norm/sos constraints, expressions,
approximation) plus, when `config.pwmcc_segments > 0`, the full set of
PWMCC containers under `meta * "_pwmcc"`. Loop `(name, t)` calling the
scalar build per cell and writing all refs.

Returns the registered `QuadraticExpression` container.
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
    IS.@assert_op length(name_axis) == length(x_bounds)
    for b in x_bounds
        IS.@assert_op b.max > b.min
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
            end
            link_cons_target[name, t] = r.link_constraint
            norm_cons_target[name, t] = r.norm_constraint
            sos_cons_target[name, t] = r.sos_constraint
            link_expr_target[name, t] = r.link_expression
            norm_expr_target[name, t] = r.norm_expression
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
Pure-JuMP result of legacy vectorized `build_quadratic_approx(::SolverSOS2QuadConfig, ...)`.
"""
struct SOS2QuadResult{
    A <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    L <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    LC <: JuMP.Containers.DenseAxisArray,
    NC <: JuMP.Containers.DenseAxisArray,
    SC <: JuMP.Containers.DenseAxisArray,
    LE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    NE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    PWMCC <: Union{Nothing, PWMCCResult},
} <: QuadraticApproxResult
    approximation::A
    lambda::L
    link_constraints::LC
    norm_constraints::NC
    sos_constraints::SC
    link_expressions::LE
    norm_expressions::NE
    pwmcc::PWMCC
end

function build_quadratic_approx(
    config::SolverSOS2QuadConfig,
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
    sos_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        [lambda[name, i, t] for i in 1:n_points] in MOI.SOS2(collect(1:n_points))
    )
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

    return SOS2QuadResult(
        approximation, lambda, link_cons, norm_cons, sos_cons, link_expr, norm_expr, pwmcc,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::SOS2QuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    n_points_axis = axes(result.lambda, 2)

    lambda_target = add_variable_container!(
        container, QuadraticVariable, C, name_axis, n_points_axis, time_axis; meta,
    )
    lambda_target.data .= result.lambda.data

    link_cons_target = add_constraints_container!(
        container, SOS2LinkingConstraint, C, name_axis, time_axis; meta,
    )
    link_cons_target.data .= result.link_constraints.data

    norm_cons_target = add_constraints_container!(
        container, SOS2NormConstraint, C, name_axis, time_axis; meta,
    )
    norm_cons_target.data .= result.norm_constraints.data

    sos_cons_target = add_constraints_container!(
        container, SolverSOS2Constraint, C, name_axis, time_axis; meta,
    )
    sos_cons_target.data .= result.sos_constraints.data

    link_expr_target = add_expression_container!(
        container, SOS2LinkingExpression, C, name_axis, time_axis; meta,
    )
    link_expr_target.data .= result.link_expressions.data

    norm_expr_target = add_expression_container!(
        container, SOS2NormExpression, C, name_axis, time_axis; meta,
    )
    norm_expr_target.data .= result.norm_expressions.data

    result_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    register_pwmcc!(container, C, result.pwmcc, meta * "_pwmcc")
    return
end
