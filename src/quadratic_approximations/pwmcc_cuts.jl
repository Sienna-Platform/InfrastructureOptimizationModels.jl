# Piecewise McCormick (PWMCC) cuts for concave terms in Bin2 bilinear approximation.
# Adds K local chord upper bounds on the v^2 SoS2 approximation by partitioning
# each concave term's domain into K sub-intervals.
# LP gap shrinks from Delta^2/4 to Delta^2/(4K^2).
# These cuts supplement (do not replace) existing SoS2 constraints.

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

"Piecewise McCormick chord upper-bound constraint on v^2 approximation."
struct PiecewiseMcCormickChordUB <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (left endpoint)."
struct PiecewiseMcCormickTangentLBL <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (right endpoint)."
struct PiecewiseMcCormickTangentLBR <: ConstraintType end

"Solver-native SOS1 interval-selection constraint for piecewise McCormick cuts."
struct PiecewiseMcCormickSOS1 <: ConstraintType end

"""
    _pwmcc_selector_var!(backend, jump_model, C, name, k, t)

Create one interval-selection variable `δ_k` for the piecewise McCormick cuts. The
`ManualBackend` makes it binary; the `SolverBackend` makes it continuous in `[0,1]` (paired
with a `MOI.SOS1` constraint over the K selectors, which together with `Σ δ_k = 1` forces
exactly one selector to 1 — the same feasible set as the binaries).
"""
_pwmcc_selector_var!(::ManualBackend, jump_model, ::Type{C}, name, k, t) where {C} =
    JuMP.@variable(
        jump_model,
        base_name = "PwMcCBin_$(C)_{$(name), $(k), $(t)}",
        binary = true,
    )

_pwmcc_selector_var!(::SolverBackend, jump_model, ::Type{C}, name, k, t) where {C} =
    JuMP.@variable(
        jump_model,
        base_name = "PwMcCSel_$(C)_{$(name), $(k), $(t)}",
        lower_bound = 0.0,
        upper_bound = 1.0,
    )

"""
    _add_pwmcc_concave_cuts!(container, C, names, time_steps, v_var, q_expr, bounds, K, meta)

Add piecewise McCormick cuts on a concave term (-v^2) to tighten its SoS2 LP relaxation.

Partitions each name's [v_min, v_max] into K uniform sub-intervals and adds disaggregated
variables, binary interval selectors, and chord/tangent constraints that cut off
the interior of the SoS2 relaxation polytope.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `v_var`: container of the original variable indexed by (name, t)
- `q_expr`: expression container for the SoS2 approximation of v^2 (indexed by (name, t))
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of v domain
- `K::Int`: number of sub-intervals (K=2 is the minimal useful choice)
- `meta::String`: unique key prefix, e.g. "pwmcc_x" or "pwmcc_y"
"""
function _add_pwmcc_concave_cuts!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    v_var,
    q_expr,
    bounds::Vector{MinMax},
    K::Int,
    meta::String;
    backend::SOS2Backend = ManualBackend(),
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op K >= 1

    jump_model = get_jump_model(container)

    # Create containers
    delta_var = add_variable_container!(
        container,
        PiecewiseMcCormickBinary,
        C,
        names,
        1:K,
        time_steps;
        meta,
    )
    vd_var = add_variable_container!(
        container,
        PiecewiseMcCormickDisaggregated,
        C,
        names,
        1:K,
        time_steps;
        meta,
    )
    selector_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickSelectorSum,
        C,
        names,
        time_steps;
        meta,
    )
    sos1_cons =
        if backend isa SolverBackend
            add_constraints_container!(
            container,
            PiecewiseMcCormickSOS1,
            C,
            names,
            time_steps;
            meta,
        )
        else
            nothing
        end
    linking_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickLinking,
        C,
        names,
        time_steps;
        meta,
    )
    interval_lb_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickIntervalLB,
        C,
        names,
        1:K,
        time_steps;
        meta,
    )
    interval_ub_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickIntervalUB,
        C,
        names,
        1:K,
        time_steps;
        meta,
    )
    chord_ub_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickChordUB,
        C,
        names,
        time_steps;
        meta,
    )
    tangent_lb_l_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickTangentLBL,
        C,
        names,
        time_steps;
        meta,
    )
    tangent_lb_r_cons = add_constraints_container!(
        container,
        PiecewiseMcCormickTangentLBR,
        C,
        names,
        time_steps;
        meta,
    )

    delta = Vector{JuMP.VariableRef}(undef, K)
    vd = Vector{JuMP.VariableRef}(undef, K)

    for (idx, name) in enumerate(names), t in time_steps
        b = bounds[idx]
        IS.@assert_op b.min < b.max
        v_min = b.min
        v_max = b.max

        # Compute breakpoints and derived coefficients for this name
        brk = [v_min + k * (v_max - v_min) / K for k in 0:K]
        sum_brk = [brk[k] + brk[k + 1] for k in 1:K]
        prod_brk = [brk[k] * brk[k + 1] for k in 1:K]
        two_brk_l = [2.0 * brk[k] for k in 1:K]
        sq_brk_l = [brk[k]^2 for k in 1:K]
        two_brk_r = [2.0 * brk[k + 1] for k in 1:K]
        sq_brk_r = [brk[k + 1]^2 for k in 1:K]

        v = v_var[name, t]
        q = q_expr[name, t]

        for k in 1:K
            delta[k] =
                delta_var[name, k, t] =
                    _pwmcc_selector_var!(backend, jump_model, C, name, k, t)
            vd[k] =
                vd_var[name, k, t] = JuMP.@variable(
                    jump_model,
                    base_name = "PwMcCDis_$(C)_{$(name), $(k), $(t)}",
                )
        end

        sel_expr = JuMP.AffExpr(0.0)
        for k in 1:K
            JuMP.add_to_expression!(sel_expr, delta[k])
        end
        selector_cons[name, t] = JuMP.@constraint(jump_model, sel_expr == 1.0)
        if backend isa SolverBackend
            sos1_cons[name, t] =
                JuMP.@constraint(jump_model, delta in MOI.SOS1(collect(1:K)))
        end

        link_expr = JuMP.AffExpr(0.0)
        for k in 1:K
            JuMP.add_to_expression!(link_expr, vd[k])
        end
        linking_cons[name, t] = JuMP.@constraint(jump_model, link_expr == v)

        # We copy v to whichever vd[k] is active, and the rest are all 0.
        # So when constructing the chords and tangents for "all" vd[k], we are
        # really only constructing one. (unless LP)
        for k in 1:K
            interval_lb_cons[name, k, t] = JuMP.@constraint(
                jump_model,
                brk[k] * delta[k] <= vd[k]
            )
            interval_ub_cons[name, k, t] = JuMP.@constraint(
                jump_model,
                vd[k] <= brk[k + 1] * delta[k]
            )
        end

        # Chord upper bound: prevents q from exceeding the local piecewise chord
        # of v^2 in the LP relaxation (tightens from global chord to piecewise).
        chord_rhs = JuMP.AffExpr(0.0)
        for k in 1:K
            JuMP.add_to_expression!(chord_rhs, sum_brk[k], vd[k])
            JuMP.add_to_expression!(chord_rhs, -prod_brk[k], delta[k])
        end
        chord_ub_cons[name, t] = JuMP.@constraint(jump_model, q <= chord_rhs)

        # Tangent lower bounds from convexity of v^2 at interval endpoints.
        tang_l_rhs = JuMP.AffExpr(0.0)
        for k in 1:K
            JuMP.add_to_expression!(tang_l_rhs, two_brk_l[k], vd[k])
            JuMP.add_to_expression!(tang_l_rhs, -sq_brk_l[k], delta[k])
        end
        tangent_lb_l_cons[name, t] = JuMP.@constraint(jump_model, q >= tang_l_rhs)

        tang_r_rhs = JuMP.AffExpr(0.0)
        for k in 1:K
            JuMP.add_to_expression!(tang_r_rhs, two_brk_r[k], vd[k])
            JuMP.add_to_expression!(tang_r_rhs, -sq_brk_r[k], delta[k])
        end
        tangent_lb_r_cons[name, t] = JuMP.@constraint(jump_model, q >= tang_r_rhs)
    end

    return
end
