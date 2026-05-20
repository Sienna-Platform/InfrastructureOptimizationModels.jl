# Piecewise McCormick (PWMCC) cuts for concave terms in PWL approximations of x².
# Adds K local chord upper bounds on the v² PWL approximation by partitioning
# v's domain into K sub-intervals. The LP gap shrinks from Delta²/4 to
# Delta²/(4·K²). These cuts supplement (do not replace) the underlying PWL
# (SOS2 or manual-SOS2) constraints.
#
# Shared between solver_sos2.jl and manual_sos2.jl — both call the scalar
# `build_pwmcc_concave_cuts` per cell from inside their own scalar build.

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

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_pwmcc_concave_cuts(model, v, q_expr, v_min, v_max, K)

Scalar form: build the K-segment PWMCC cuts at a single cell. `v` is a
JuMP scalar (the original variable) and `q_expr` is the scalar expression
for the existing PWL v² approximation at this cell.

Returns a NamedTuple:
- `delta_var`               :: DenseAxisArray{VariableRef, 1} over `1:K` (binary)
- `vd_var`                  :: DenseAxisArray{VariableRef, 1} over `1:K` (continuous)
- `selector_constraint`     :: scalar (Σ_k δ_k == 1)
- `linking_constraint`      :: scalar (Σ_k vd_k == v)
- `interval_lb_constraints` :: DenseAxisArray{Constraint, 1} over `1:K`
- `interval_ub_constraints` :: DenseAxisArray{Constraint, 1} over `1:K`
- `chord_ub_constraint`     :: scalar
- `tangent_lb_l_constraint` :: scalar
- `tangent_lb_r_constraint` :: scalar
"""
function build_pwmcc_concave_cuts(
    model::JuMP.Model,
    v::JuMP.AbstractJuMPScalar,
    q_expr,
    v_min::Float64,
    v_max::Float64,
    K::Int,
)
    IS.@assert_op K >= 1
    IS.@assert_op v_min < v_max

    brk = [v_min + k * (v_max - v_min) / K for k in 0:K]  # length K+1, indexed 1..K+1

    delta_var = JuMP.@variable(
        model, [k = 1:K],
        binary = true,
        base_name = "PwMcCBin",
    )
    vd_var = JuMP.@variable(
        model, [k = 1:K],
        base_name = "PwMcCDis",
    )

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

# --- Legacy vectorized build + register (kept for SolverSOS2/ManualSOS2's
# vectorized build_quadratic_approx until those callers migrate; removed in
# sweep) ---

"""
Pure-JuMP result of legacy vectorized `build_pwmcc_concave_cuts`. Fields are
JuMP container arrays indexed by (name, k, t) for the K-segment pieces or
(name, t) for the once-per-element constraints.
"""
struct PWMCCResult{
    DV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    VDV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    SC <: JuMP.Containers.DenseAxisArray,
    LC <: JuMP.Containers.DenseAxisArray,
    ILBC <: JuMP.Containers.DenseAxisArray,
    IUBC <: JuMP.Containers.DenseAxisArray,
    CUBC <: JuMP.Containers.DenseAxisArray,
    TLLC <: JuMP.Containers.DenseAxisArray,
    TLRC <: JuMP.Containers.DenseAxisArray,
}
    delta_var::DV
    vd_var::VDV
    selector_constraints::SC
    linking_constraints::LC
    interval_lb_constraints::ILBC
    interval_ub_constraints::IUBC
    chord_ub_constraints::CUBC
    tangent_lb_l_constraints::TLLC
    tangent_lb_r_constraints::TLRC
end

"""
    build_pwmcc_concave_cuts(model, v_var, q_expr, bounds, K) -> PWMCCResult

Legacy vectorized form. Build piecewise McCormick cuts on the concave term
(−v²) to tighten the LP relaxation of a PWL approximation `q_expr ≈ v²`.
Partitions each name's [v_min, v_max] into K uniform sub-intervals.
"""
function build_pwmcc_concave_cuts(
    model::JuMP.Model,
    v_var,
    q_expr,
    bounds::Vector{MinMax},
    K::Int,
)
    IS.@assert_op K >= 1
    name_axis = axes(v_var, 1)
    time_axis = axes(v_var, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.min < b.max
    end

    brk = JuMP.Containers.DenseAxisArray(
        [
            bounds[i].min + k * (bounds[i].max - bounds[i].min) / K
            for i in eachindex(name_axis), k in 0:K
        ],
        name_axis,
        0:K,
    )

    delta_var = JuMP.@variable(
        model,
        [name = name_axis, k = 1:K, t = time_axis],
        binary = true,
        base_name = "PwMcCBin",
    )
    vd_var = JuMP.@variable(
        model,
        [name = name_axis, k = 1:K, t = time_axis],
        base_name = "PwMcCDis",
    )

    selector_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        sum(delta_var[name, k, t] for k in 1:K) == 1.0
    )
    linking_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        sum(vd_var[name, k, t] for k in 1:K) == v_var[name, t]
    )
    interval_lb = JuMP.@constraint(
        model,
        [name = name_axis, k = 1:K, t = time_axis],
        brk[name, k - 1] * delta_var[name, k, t] <= vd_var[name, k, t]
    )
    interval_ub = JuMP.@constraint(
        model,
        [name = name_axis, k = 1:K, t = time_axis],
        vd_var[name, k, t] <= brk[name, k] * delta_var[name, k, t]
    )
    chord_ub = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        q_expr[name, t] <= sum(
            (brk[name, k - 1] + brk[name, k]) * vd_var[name, k, t] -
            brk[name, k - 1] * brk[name, k] * delta_var[name, k, t] for k in 1:K
        )
    )
    tangent_lb_l = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        q_expr[name, t] >= sum(
            2.0 * brk[name, k - 1] * vd_var[name, k, t] -
            brk[name, k - 1]^2 * delta_var[name, k, t] for k in 1:K
        )
    )
    tangent_lb_r = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        q_expr[name, t] >= sum(
            2.0 * brk[name, k] * vd_var[name, k, t] -
            brk[name, k]^2 * delta_var[name, k, t] for k in 1:K
        )
    )

    return PWMCCResult(
        delta_var,
        vd_var,
        selector_cons,
        linking_cons,
        interval_lb,
        interval_ub,
        chord_ub,
        tangent_lb_l,
        tangent_lb_r,
    )
end

"""
    register_pwmcc!(container, ::Type{C}, pwmcc::PWMCCResult, meta)

Legacy registration helper for the vectorized PWMCC result.
"""
function register_pwmcc!(
    container::OptimizationContainer,
    ::Type{C},
    pwmcc::PWMCCResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(pwmcc.delta_var, 1)
    k_axis = axes(pwmcc.delta_var, 2)
    time_axis = axes(pwmcc.delta_var, 3)

    delta_target = add_variable_container!(
        container, PiecewiseMcCormickBinary, C, name_axis, k_axis, time_axis; meta,
    )
    delta_target.data .= pwmcc.delta_var.data

    vd_target = add_variable_container!(
        container, PiecewiseMcCormickDisaggregated, C, name_axis, k_axis, time_axis;
        meta,
    )
    vd_target.data .= pwmcc.vd_var.data

    selector_target = add_constraints_container!(
        container, PiecewiseMcCormickSelectorSum, C, name_axis, time_axis; meta,
    )
    selector_target.data .= pwmcc.selector_constraints.data

    linking_target = add_constraints_container!(
        container, PiecewiseMcCormickLinking, C, name_axis, time_axis; meta,
    )
    linking_target.data .= pwmcc.linking_constraints.data

    chord_target = add_constraints_container!(
        container, PiecewiseMcCormickChordUB, C, name_axis, time_axis; meta,
    )
    chord_target.data .= pwmcc.chord_ub_constraints.data

    tangent_l_target = add_constraints_container!(
        container, PiecewiseMcCormickTangentLBL, C, name_axis, time_axis; meta,
    )
    tangent_l_target.data .= pwmcc.tangent_lb_l_constraints.data

    tangent_r_target = add_constraints_container!(
        container, PiecewiseMcCormickTangentLBR, C, name_axis, time_axis; meta,
    )
    tangent_r_target.data .= pwmcc.tangent_lb_r_constraints.data

    interval_lb_target = add_constraints_container!(
        container, PiecewiseMcCormickIntervalLB, C, name_axis, k_axis, time_axis; meta,
    )
    interval_lb_target.data .= pwmcc.interval_lb_constraints.data

    interval_ub_target = add_constraints_container!(
        container, PiecewiseMcCormickIntervalUB, C, name_axis, k_axis, time_axis; meta,
    )
    interval_ub_target.data .= pwmcc.interval_ub_constraints.data
    return
end

# No-op when the caller did not build PWMCC cuts.
register_pwmcc!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing
