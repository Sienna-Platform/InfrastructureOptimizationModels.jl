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
Pure-JuMP result of `build_quadratic_approx(::SolverSOS2QuadConfig, ...)`.
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

"""
    build_quadratic_approx(config::SolverSOS2QuadConfig, model, x, bounds)

PWL approximation of x² with solver-native MOI.SOS2 adjacency on the
convex-combination weights. If `config.pwmcc_segments > 0`, also adds
piecewise McCormick concave cuts to tighten the LP relaxation.
"""
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
    # x² = x_min² + 2·x_min·(x − x_min) + lx² · xh² where xh² ≈ Σ λ_i · x_bkpts[i]²
    #     = lx² · Σ λ_i · x_bkpts[i]² + 2·x_min·x − x_min²
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
