# Piecewise McCormick (PWMCC) cuts for concave terms in PWL approximations of x².
# Adds K local chord upper bounds on the v² PWL approximation by partitioning
# v's domain into K sub-intervals. The LP gap shrinks from Delta²/4 to
# Delta²/(4·K²). These cuts supplement (do not replace) the underlying PWL
# (SOS2 or manual-SOS2) constraints.
#
# Shared between solver_sos2.jl and manual_sos2.jl — both reference these
# container keys and the build/register helpers below.

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

# --- Result struct ---

"""
Pure-JuMP result of `build_pwmcc_concave_cuts`. All fields are JuMP container
arrays indexed by (name, k, t) for the K-segment pieces or (name, t) for the
once-per-element constraints.
"""
struct PWMCCResult{DV, VDV, SC, LC, ILBC, IUBC, CUBC, TLLC, TLRC}
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

# --- Pure-JuMP build ---

"""
    build_pwmcc_concave_cuts(model, v_var, q_expr, bounds, K) -> PWMCCResult

Build piecewise McCormick cuts on the concave term (−v²) to tighten the LP
relaxation of a PWL approximation `q_expr ≈ v²`. Partitions each name's
[v_min, v_max] into K uniform sub-intervals.

# Arguments
- `model::JuMP.Model`: JuMP model.
- `v_var`: 2D container of the original variable v, indexed by (name, t).
- `q_expr`: 2D container of the existing PWL approximation expressions for v².
- `bounds`: per-name (v_min, v_max).
- `K::Int`: number of sub-intervals (K = 1 is degenerate; K ≥ 2 useful).
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

    # Per-name breakpoint coefficients
    v_min_arr = JuMP.Containers.DenseAxisArray([b.min for b in bounds], name_axis)
    v_max_arr = JuMP.Containers.DenseAxisArray([b.max for b in bounds], name_axis)
    brk = JuMP.Containers.DenseAxisArray{Float64}(undef, name_axis, 0:K)
    for (i, name) in enumerate(name_axis)
        bmin = bounds[i].min
        bmax = bounds[i].max
        IS.@assert_op bmin < bmax
        for k in 0:K
            brk[name, k] = bmin + k * (bmax - bmin) / K
        end
    end

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
    # Chord upper bound: q ≤ Σ_k (brk[k-1]+brk[k]) * vd_k − brk[k-1]*brk[k] * δ_k
    chord_ub = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        q_expr[name, t] <= sum(
            (brk[name, k - 1] + brk[name, k]) * vd_var[name, k, t] -
            brk[name, k - 1] * brk[name, k] * delta_var[name, k, t] for k in 1:K
        )
    )
    # Tangent lower bound at left endpoint: q ≥ Σ_k 2·brk[k-1]*vd_k − brk[k-1]²·δ_k
    tangent_lb_l = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        q_expr[name, t] >= sum(
            2.0 * brk[name, k - 1] * vd_var[name, k, t] -
            brk[name, k - 1]^2 * delta_var[name, k, t] for k in 1:K
        )
    )
    # Tangent lower bound at right endpoint: q ≥ Σ_k 2·brk[k]*vd_k − brk[k]²·δ_k
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

# --- IOM-side register helper ---

"""
    register_pwmcc!(container, ::Type{C}, pwmcc::PWMCCResult, meta)

Register all PWMCC variables and constraints in the optimization container
under the corresponding key types, suffixed by `meta`.
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
        container,
        PiecewiseMcCormickBinary,
        C,
        collect(name_axis),
        k_axis,
        time_axis;
        meta,
    )
    for name in name_axis, k in k_axis, t in time_axis
        delta_target[name, k, t] = pwmcc.delta_var[name, k, t]
    end

    vd_target = add_variable_container!(
        container,
        PiecewiseMcCormickDisaggregated,
        C,
        collect(name_axis),
        k_axis,
        time_axis;
        meta,
    )
    for name in name_axis, k in k_axis, t in time_axis
        vd_target[name, k, t] = pwmcc.vd_var[name, k, t]
    end

    selector_target = add_constraints_container!(
        container,
        PiecewiseMcCormickSelectorSum,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    linking_target = add_constraints_container!(
        container,
        PiecewiseMcCormickLinking,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    chord_target = add_constraints_container!(
        container,
        PiecewiseMcCormickChordUB,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    tangent_l_target = add_constraints_container!(
        container,
        PiecewiseMcCormickTangentLBL,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    tangent_r_target = add_constraints_container!(
        container,
        PiecewiseMcCormickTangentLBR,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        selector_target[name, t] = pwmcc.selector_constraints[name, t]
        linking_target[name, t] = pwmcc.linking_constraints[name, t]
        chord_target[name, t] = pwmcc.chord_ub_constraints[name, t]
        tangent_l_target[name, t] = pwmcc.tangent_lb_l_constraints[name, t]
        tangent_r_target[name, t] = pwmcc.tangent_lb_r_constraints[name, t]
    end

    interval_lb_target = add_constraints_container!(
        container,
        PiecewiseMcCormickIntervalLB,
        C,
        collect(name_axis),
        k_axis,
        time_axis;
        meta,
    )
    interval_ub_target = add_constraints_container!(
        container,
        PiecewiseMcCormickIntervalUB,
        C,
        collect(name_axis),
        k_axis,
        time_axis;
        meta,
    )
    for name in name_axis, k in k_axis, t in time_axis
        interval_lb_target[name, k, t] = pwmcc.interval_lb_constraints[name, k, t]
        interval_ub_target[name, k, t] = pwmcc.interval_ub_constraints[name, k, t]
    end
    return
end
