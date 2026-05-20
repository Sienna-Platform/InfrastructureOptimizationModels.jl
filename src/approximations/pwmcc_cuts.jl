# Piecewise McCormick (PWMCC) cuts for concave terms in PWL approximations of x².
# Adds K local chord upper bounds on the v² PWL approximation by partitioning
# v's domain into K sub-intervals. The LP gap shrinks from Delta²/4 to
# Delta²/(4·K²). These cuts supplement (do not replace) the underlying PWL
# (SOS2 or manual-SOS2) constraints.

# --- Container key types ---

"Binary interval selector for piecewise McCormick cuts."
struct PiecewiseMcCormickBinary <: SparseVariableType end

"Disaggregated variable for piecewise McCormick cuts."
struct PiecewiseMcCormickDisaggregated <: SparseVariableType end

"Selector sum constraint: sum_k delta_k = 1."
struct PiecewiseMcCormickSelectorSum <: ConstraintType end

"Disaggregation linking constraint: v = sum_k v^d_k."
struct PiecewiseMcCormickLinking <: ConstraintType end

"Interval activation lower bound: t_{k-1} * delta_k <= v^d_k."
struct PiecewiseMcCormickIntervalLB <: ConstraintType end

"Interval activation upper bound: v^d_k <= t_k * delta_k."
struct PiecewiseMcCormickIntervalUB <: ConstraintType end

"Piecewise McCormick chord upper-bound constraint on v² approximation."
struct PiecewiseMcCormickChordUB <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (left endpoint)."
struct PiecewiseMcCormickTangentLBL <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (right endpoint)."
struct PiecewiseMcCormickTangentLBR <: ConstraintType end

"""
    build_pwmcc_concave_cuts(model, v, q_expr, v_min, v_max, K)

Scalar form: build the K-segment PWMCC cuts at a single cell. `v` is the
JuMP scalar variable and `q_expr` is the scalar expression for the
existing PWL v² approximation at this cell.

Returns a NamedTuple with `(delta_var, vd_var, selector_constraint,
linking_constraint, interval_lb_constraints, interval_ub_constraints,
chord_ub_constraint, tangent_lb_l_constraint, tangent_lb_r_constraint)`.
"""
function build_pwmcc_concave_cuts(
    model::JuMP.Model,
    v::JuMP.AbstractJuMPScalar,
    q_expr,
    v_min::Float64,
    v_max::Float64,
    K::Int,
)
    @assert K >= 1
    @assert v_min < v_max

    brk = [v_min + k * (v_max - v_min) / K for k in 0:K]

    delta_var = JuMP.@variable(
        model, [k = 1:K], binary = true, base_name = "PwMcCBin",
    )
    vd_var = JuMP.@variable(model, [k = 1:K], base_name = "PwMcCDis")

    selector_con = JuMP.@constraint(model, sum(delta_var[k] for k in 1:K) == 1.0)
    linking_con = JuMP.@constraint(model, sum(vd_var[k] for k in 1:K) == v)

    interval_lb = JuMP.@constraint(
        model, [k = 1:K], brk[k] * delta_var[k] <= vd_var[k],
    )
    interval_ub = JuMP.@constraint(
        model, [k = 1:K], vd_var[k] <= brk[k + 1] * delta_var[k],
    )
    chord_ub = JuMP.@constraint(
        model,
        q_expr <= sum(
            (brk[k] + brk[k + 1]) * vd_var[k] -
            brk[k] * brk[k + 1] * delta_var[k] for k in 1:K
        ),
    )
    tangent_lb_l = JuMP.@constraint(
        model,
        q_expr >= sum(
            2.0 * brk[k] * vd_var[k] - brk[k]^2 * delta_var[k] for k in 1:K
        ),
    )
    tangent_lb_r = JuMP.@constraint(
        model,
        q_expr >= sum(
            2.0 * brk[k + 1] * vd_var[k] - brk[k + 1]^2 * delta_var[k] for k in 1:K
        ),
    )

    return (;
        delta_var,
        vd_var,
        selector_constraint = selector_con,
        linking_constraint = linking_con,
        interval_lb_constraints = interval_lb,
        interval_ub_constraints = interval_ub,
        chord_ub_constraint = chord_ub,
        tangent_lb_l_constraint = tangent_lb_l,
        tangent_lb_r_constraint = tangent_lb_r,
    )
end

# --- Allocation + per-cell write helpers (used by SOS2 adapters) ---

function _alloc_pwmcc_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, K::Int, meta,
) where {C <: IS.InfrastructureSystemsComponent}
    return (
        delta = add_variable_container!(
            container, PiecewiseMcCormickBinary, C, name_axis, 1:K, time_axis; meta,
        ),
        vd = add_variable_container!(
            container, PiecewiseMcCormickDisaggregated, C,
            name_axis, 1:K, time_axis; meta,
        ),
        selector = add_constraints_container!(
            container, PiecewiseMcCormickSelectorSum, C, name_axis, time_axis; meta,
        ),
        linking = add_constraints_container!(
            container, PiecewiseMcCormickLinking, C, name_axis, time_axis; meta,
        ),
        interval_lb = add_constraints_container!(
            container, PiecewiseMcCormickIntervalLB, C,
            name_axis, 1:K, time_axis; meta,
        ),
        interval_ub = add_constraints_container!(
            container, PiecewiseMcCormickIntervalUB, C,
            name_axis, 1:K, time_axis; meta,
        ),
        chord = add_constraints_container!(
            container, PiecewiseMcCormickChordUB, C, name_axis, time_axis; meta,
        ),
        tangent_l = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBL, C, name_axis, time_axis; meta,
        ),
        tangent_r = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBR, C, name_axis, time_axis; meta,
        ),
    )
end

function _write_pwmcc_cell!(targets, name, t, r, K::Int)
    for k in 1:K
        targets.delta[name, k, t] = r.delta_var[k]
        targets.vd[name, k, t] = r.vd_var[k]
        targets.interval_lb[name, k, t] = r.interval_lb_constraints[k]
        targets.interval_ub[name, k, t] = r.interval_ub_constraints[k]
    end
    targets.selector[name, t] = r.selector_constraint
    targets.linking[name, t] = r.linking_constraint
    targets.chord[name, t] = r.chord_ub_constraint
    targets.tangent_l[name, t] = r.tangent_lb_l_constraint
    targets.tangent_r[name, t] = r.tangent_lb_r_constraint
    return
end
