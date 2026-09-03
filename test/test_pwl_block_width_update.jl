"""
Regression test for the stale PWL block-offer width defect.

`add_pwl_block_offer_constraints!` (`objective_function_pwl_delta.jl`) builds each
block-width upper bound `δ_k <= breakpoints[k+1] - breakpoints[k]` and discards the
`ConstraintRef`. Block-offer delta variables are created with `upper_bound = Inf`, so
that discarded row is their only upper bound. In a recurrent solve, updating the
breakpoint parameter data between solves has no way to reach that row, so the width
stays pinned to whatever was baked in at the first build.

This isolates the defect from the slope path: slopes are held fixed across the two
"periods" below (time-varying slopes are already handled correctly elsewhere, by
rebuilding the objective term, not by patching a coefficient), so the only thing
exercised here is the block-width bound.

As of the fix that stores each per-block width `ConstraintRef` in a sparse container
keyed `(name, block, t)` (`IOM.get_constraint(container, <WidthConstraintType>,
ComponentType)`), this test asserts the mechanism directly: fetch the width container,
`JuMP.set_normalized_rhs` a block's row to a new width, and confirm
`JuMP.normalized_rhs` reads the new value back. It then simulates the update PSI's
updater performs (this test lives in IOM and cannot exercise PSI's parameter-update
code, so it sets the RHS itself the same way that updater would): fix dispatch, solve
under one set of breakpoints, set every block's width RHS to the second period's
values, re-solve without rebuilding, and compare the solved cost to the ground truth
computed independently from the second period's curve data. The two periods are
chosen so the same fixed dispatch is feasible under both, isolating the comparison to
cost, not feasibility.
"""

# Merit-order fill: cheapest (lowest-index) segment first. Valid because the offer
# slopes here are strictly increasing, and the delta formulation's width bounds make
# that fill the unique cost-minimizing decomposition of `dispatch` into segments.
function _expected_merit_order_cost(
    breakpoints::Vector{Float64},
    slopes::Vector{Float64},
    dispatch::Float64,
)
    remaining = dispatch
    cost = 0.0
    for k in eachindex(slopes)
        width = breakpoints[k + 1] - breakpoints[k]
        used = min(width, remaining)
        cost += used * slopes[k]
        remaining -= used
        if remaining <= 0.0
            break
        end
    end
    @assert remaining <= 1e-9
    return cost
end

@testset "Stale PWL block-offer width after a recurrent-solve breakpoint update" begin
    time_steps = 1:1
    names = ["gen1"]
    base_power = 100.0

    # Same slopes for both periods: isolates the width defect from the (already
    # correct) slope-update path.
    slopes = [10.0, 20.0]

    # Period A widths: [50, 50] (breakpoints 0, 50, 100).
    breakpoints_a = [0.0, 50.0, 100.0]
    # Period B widths: [20, 70] (breakpoints 0, 20, 90). Same segment count as A, so
    # the already-built delta variables and (if reachable) width rows could in
    # principle carry the new period without a rebuild.
    breakpoints_b = [0.0, 20.0, 90.0]

    # Feasible under both periods' curve tops (100 and 90) so only cost is compared.
    dispatch_mw = 90.0

    # The width constraint's RHS lives in system per-unit, the same conversion
    # `_fill_pwl_data_from_arrays!` applies at build time (`NaturalUnit`, device and
    # system base powers equal here, so this is dividing by `base_power`).
    to_pu_breakpoints(bp) = first(
        IOM.get_piecewise_curve_per_system_unit(
            bp, slopes, IS.NaturalUnit(), base_power, base_power))

    container = make_test_container(time_steps; base_power = base_power)
    add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", 1)
    add_test_expression!(
        container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

    slopes_mat = [slopes for _ in 1:1, _ in time_steps]
    bp_mat = [breakpoints_a for _ in 1:1, _ in time_steps]
    setup_delta_pwl_parameters!(
        container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

    device = make_mock_thermal("gen1"; base_power = base_power)
    cost_fn = _make_ts_incremental_cost_curve()

    IOM.add_variable_cost_to_objective!(
        container, TestVariableType, device, cost_fn, TestDeviceFormulation)

    jm = IOM.get_jump_model(container)
    p = IOM.get_variable(container, TestVariableType, MockThermalGen)
    JuMP.set_optimizer(jm, HiGHS.Optimizer)
    JuMP.set_silent(jm)

    obj = IOM.get_objective_expression(container)
    variant = IOM.get_variant_terms(obj)
    JuMP.@objective(jm, JuMP.MOI.MIN_SENSE, variant)
    JuMP.fix(p["gen1", 1], dispatch_mw / base_power; force = true)

    JuMP.optimize!(jm)
    @test JuMP.termination_status(jm) == JuMP.MOI.OPTIMAL

    expected_a = _expected_merit_order_cost(breakpoints_a, slopes, dispatch_mw)
    @test isapprox(JuMP.objective_value(jm), expected_a; atol = 1e-6)

    # Direct mechanism check: the per-block width `ConstraintRef` is reachable and
    # mutable through the sparse width container, keyed (name, block, t).
    WidthType = IOM._block_width_constraint(IOM.IncrementalOffer())
    width_container = IOM.get_constraint(container, WidthType, MockThermalGen)
    block_one_ref = width_container[("gen1", 1, 1)]
    pu_breakpoints_a = to_pu_breakpoints(breakpoints_a)
    @test isapprox(
        JuMP.normalized_rhs(block_one_ref), pu_breakpoints_a[2] - pu_breakpoints_a[1];
        atol = 1e-9)
    JuMP.set_normalized_rhs(block_one_ref, 12.5)
    @test isapprox(JuMP.normalized_rhs(block_one_ref), 12.5; atol = 1e-9)
    # Restore before the recurrent-solve comparison below so it isn't polluted.
    JuMP.set_normalized_rhs(block_one_ref, pu_breakpoints_a[2] - pu_breakpoints_a[1])

    # Recurrent solve: the breakpoint data for the same (name, t) slot is updated in
    # place, as a rolling-horizon driver would between simulation steps, then the
    # SAME model is re-solved without rebuilding variables or constraints. This test
    # lives in IOM and cannot exercise PSI's parameter-update code, so it performs the
    # width-RHS update the same way that updater does: `JuMP.set_normalized_rhs` on
    # each block's stored ref, driven by the new breakpoint values.
    BPParam = IOM._breakpoint_param(IOM.IncrementalOffer())
    bp_param_container = IOM.get_parameter(container, BPParam, MockThermalGen)
    for (k, bp) in enumerate(breakpoints_b)
        IOM.set_parameter!(bp_param_container, jm, bp, "gen1", k, 1)
    end
    pu_breakpoints_b = to_pu_breakpoints(breakpoints_b)
    for k in eachindex(slopes)
        JuMP.set_normalized_rhs(
            width_container[("gen1", k, 1)], pu_breakpoints_b[k + 1] - pu_breakpoints_b[k])
    end

    JuMP.optimize!(jm)
    @test JuMP.termination_status(jm) == JuMP.MOI.OPTIMAL

    expected_b = _expected_merit_order_cost(breakpoints_b, slopes, dispatch_mw)
    @test isapprox(JuMP.objective_value(jm), expected_b; atol = 1e-6)
end
