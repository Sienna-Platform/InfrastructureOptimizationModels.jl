# Fix: time-varying PWL block-offer widths are frozen after the first solve

Date: 2026-09-03. Primary repo: `InfrastructureOptimizationModels.jl` (IOM), branch `jd/market_model` @ `f81e97c` (13 commits ahead of `origin/main`, clean).
Secondary repo: `PowerSimulations.jl` (PSI), branch `jd/pom_excision`. No changes expected in `PowerOperationsModels.jl` (POM), but its suite must be re-run.

## 1. The defect

`src/objective_function/objective_function_pwl_delta.jl`, `add_pwl_block_offer_constraints!` (~line 132):

```julia
con_container[name, t] = JuMP.@constraint(jump_model, power_var == sum_pwl)   # linking constraint: ref STORED
for (ix, var) in enumerate(pwl_vars)
    JuMP.@constraint(jump_model, var <= breakpoints[ix + 1] - breakpoints[ix])  # ref DISCARDED
end
```

The per-block width bound is built once and its `ConstraintRef` is thrown away. Nothing can reach the row afterwards, so in a recurrent solve the width stays at whatever the first period produced.

**Why it is silent rather than loud.** The block-offer delta variables are created with `upper_bound = Inf` (`add_pwl_variables_delta!`, called from `value_curve_cost.jl:353` and both POM call sites, whose docstring says *"use `Inf` for block offer formulation where segment capacity is enforced by constraints instead"*). The discarded constraint is therefore the **only** upper bound on those variables. A stale width does not error; it lets the solver dispatch the wrong quantity in each tranche at the wrong marginal price. The model stays feasible and the answer is quietly wrong.

**Measured.** Two independent agents reproduced the same numbers on the import/export-cost fixture: solved objective delta across two simulation steps froze at `[-220.68, -220.68]` against a ground-truth delta of `[-220.68, -229.08]`.

**Scope of wrongness.** Time-varying **breakpoints** only. Time-varying **slopes** are correct, because PSI rebuilds the objective term each step (`_update_pwl_cost_expression` → `add_to_objective_variant_expression!`) rather than patching a coefficient. Nothing analogous exists for a constraint RHS.

**The bug is documented in PSI as if it were intended.** `PowerSimulations.jl/src/parameters/update_cost_parameters.jl:121-128` carries this comment above a deliberate no-op:

> *"Breakpoints re-parametrize the block-width constraint (`_update_pwl_cost_expression` handles that at build time), not an objective term, so updating one here is a genuine no-op."*

That claim is false. `_update_pwl_cost_expression` only rebuilds the objective from slopes. No code re-parametrizes the width constraint at any point after the initial build. The comment records an assumption that was never implemented, which is why the gap survived review.

## 2. Options considered

Breakpoints are **always `Float64`** in practice. `AbstractPiecewiseLinearBreakpointParameter <: ObjectiveFunctionParameter` (not `TimeSeriesParameter`), and `add_cost_function_parameter_container!` hardcodes `data_type = Float64` with no caller overriding it anywhere in IOM or POM. The permissive `Vector{<:JuMPOrFloat}` signature is not exercised with JuMP values.

| Option | Approach | Verdict |
|---|---|---|
| **A. Store the refs** | Keep the constraint; store each `ConstraintRef` in a sparse container keyed `(name, block, t)`; PSI updates with `JuMP.set_normalized_rhs`. | **Recommended.** |
| B. Use variable bounds | Drop the constraint; `JuMP.set_upper_bound(var, width)`; update the same way. | Rejected — see below. |
| C. JuMP-parameter breakpoints | Store breakpoints as fixed `VariableRef`s so the RHS updates via `JuMP.fix`. | Rejected — most invasive. |

**Why not B**, despite being the smallest diff and needing no new type: it converts rows into bounds, which changes the model's constraint counts. POM's suite carries ~200 exact-count assertions (`moi_tests`, `check_constraint_count`). Every block-offer count would shift, forcing churn in a repo owned by other people, to buy a marginally smaller diff. It would also contradict the existing docstring's stated design and discard the rows as a potential dual source. Option A preserves the formulation exactly: same rows, same counts, no POM test churn.

**Why not C**: cost parameters are deliberately walled off from JuMP-parameter storage (the `ObjectiveFunctionParameter` vs `TimeSeriesParameter` split is explicitly commented as such in `offer_curve_types.jl:36-39`). Adopting the recurrent-solve `VariableRef` path for breakpoints would diverge from how every other cost parameter works, enlarge the model, and change what `calculate_parameter_values` returns. Disproportionate to the defect.

## 3. RESOLVED by Task 1: Case 1, padded maximum

Task 1 (commit `3b4a347`) settled this empirically. **`n_segments` is hoisted out of the per-period loop** in `value_curve_cost.jl`'s `_add_ts_incremental_pwl_cost!` and sized once from the parameter array's fixed axis length — the batch-wide maximum, corroborated by `get_max_tranches`'s own docstring ("global maximum over time").

Evidence, from a two-period container with 2 real segments at t=1 and 4 at t=2:

| observation | t=1 | t=2 |
|---|---|---|
| delta variable keys created | `("gen1", 1..4, 1)` — **4**, not 2 | `("gen1", 1..4, 2)` — 4 |
| width rows | blocks 1-2 `<= 0.5` real; blocks 3-4 `<= 0.0` padding | all 4 real |
| objective slope coefficients | nonzero on 1-2 only | nonzero on all 4 |

**Consequence: `set_normalized_rhs` is sufficient.** Every period already has a row per block up to the global maximum, with surplus blocks pinned to zero width. No structural second defect exists on the code path Task 2 targets, and no loud-error guard is needed there.

The original framing below was correct in substance but imprecise: raw tranche counts *are* computed inside the per-period loop, yet both cited call sites read through fixed-shape padded arrays, so neither ever actually sees raggedness.

### Superseded original text



Block counts are **ragged**: POM computes `breakpoints, slopes = IOM._get_pwl_data(dir, container, component, t)` inside the `for t in time_steps` loop, so tranche count can differ per period. Meanwhile the parameter *arrays* are padded to a batch-wide maximum via `calc_additional_axes`.

So: at build time, are variables and width constraints created for **that period's** tranche count, or for the **padded maximum**? This decides the fix's sufficiency.

- If built to the padded maximum, `set_normalized_rhs` is sufficient; surplus blocks get a zero width.
- If built to the period's own count, a later period needing **more** tranches has no variable or row to update, and `set_normalized_rhs` cannot conjure one. That is a second, structural defect: such a case is only representable with `rebuild_model = true`.

Settle this empirically before writing the fix. If the second case holds, the fix must additionally **error loudly** when a period's tranche count exceeds what was built, rather than silently updating a subset — consistent with this line's rule against silent-failure patterns.

## 4. Global constraints

- No shims, no compat aliases. No version or compat bumps in any `Project.toml`.
- Never hand-edit generated code. Never modify `src/core/optimization_container.jl` beyond the additive container call the fix needs, and prefer the existing `add_constraints_container!` / `lazy_container_addition!` entry points over new plumbing.
- Style: no `isa`/`<:` runtime branching (use dispatch), no ternaries, explicit `function … end` with explicit `return`, `iszero` over `== 0`, no `isnothing(x) && continue` guards.
- Every `PSY` getter on a convertible field passes the unit system explicitly (`PSY.SU`).
- IOM is a library with multiple consumers. A function with no in-IOM caller is not dead code. Do not reshuffle imports or exports beyond what this fix requires.
- **Branch decision (repository owner, 2026-09-03): the fix lands directly on `jd/market_model`.** Per-task commits, no trailers of any kind. Do not open a separate fix branch.
- Formatter before every commit: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`.
- PSI's `[sources]` pins IOM by **local path**, so whatever branch IOM has checked out is what PSI compiles. Creating and checking out a fix branch changes PSI's build immediately — intended, but be aware.

---

## Task 1: Establish ground truth and a failing test

**Files:** Create `test/test_pwl_block_width_update.jl` (IOM).

- [ ] **Step 1: Settle the ragged-tranche question from §3.** Build a container with a two-period, time-varying block offer whose tranche counts differ between periods. Print, per period, the number of delta variables actually created and the keys present in the width path. Record which of the two cases in §3 holds. Do not proceed on assumption.

- [ ] **Step 2: Write a failing regression test.** Build a model with a time-varying breakpoint offer, solve, change the breakpoint data to the second period's values, re-solve, and assert the block-width RHS changed. Today it cannot, because the ref is unreachable — so first assert the observable proxy: that the solved objective differs between the two periods by the independently computed ground-truth delta. Use the measured `[-220.68, -229.08]` shape from the import/export fixture as the reference if it transfers, otherwise compute the expected delta from the curve data directly.

- [ ] **Step 3: Run it and confirm it fails for the right reason.** `julia --project=test test/runtests.jl` (or include the single file). Expected: the second-period objective equals the first, proving staleness. Capture the exact output — it is the before-picture for the fix.

- [ ] **Step 4: Commit.** `git commit -m "Add a failing regression test for stale PWL block-offer widths"`

## Task 2: Introduce the width-constraint type and store the refs

**Files:** Modify `src/objective_function/offer_curve_types.jl`, `src/objective_function/objective_function_pwl_delta.jl`, `src/objective_function/value_curve_cost.jl`, `src/InfrastructureOptimizationModels.jl`.

**Interfaces produced:** a sparse constraint container keyed `(name::String, block::Int, t::Int)` reachable as `get_constraint(container, <WidthConstraintType>, ComponentType)`, holding one ref per block-width row.

- [ ] **Step 1: Define the constraint types**, mirroring the existing offer pair in `offer_curve_types.jl:76-92` verbatim in style:

```julia
abstract type AbstractPiecewiseLinearBlockWidthConstraint <: ConstraintType end

"""
Struct to create the block-width limit associated with a piecewise-linear incremental offer.
"""
struct PiecewiseLinearBlockIncrementalWidthConstraint <:
       AbstractPiecewiseLinearBlockWidthConstraint end

"""
Struct to create the block-width limit associated with a piecewise-linear decremental offer.
"""
struct PiecewiseLinearBlockDecrementalWidthConstraint <:
       AbstractPiecewiseLinearBlockWidthConstraint end
```

- [ ] **Step 2: Extend the direction dispatch table.** `value_curve_cost.jl:119-120` already maps offer direction to the block-offer constraint type. Add the parallel mapping to the new width types, in the same form. Do not branch on direction with an `if`.

- [ ] **Step 3: Create the container and store the refs.** In `add_pwl_constraint_delta!`, create the width container alongside the existing linking container, using the sparse idiom that mirrors how `pwl_vars` are already keyed:

```julia
width_container = lazy_container_addition!(
    container,
    <WidthConstraintType>,
    T;
    sparse = true,
    meta = meta,
)
```

Thread it into `add_pwl_block_offer_constraints!` as a new argument and store each ref:

```julia
for (ix, var) in enumerate(pwl_vars)
    width_container[(name, ix, t)] =
        JuMP.@constraint(jump_model, var <= breakpoints[ix + 1] - breakpoints[ix])
end
```

Confirm the exact `lazy_container_addition!` / `add_constraints_container!` signature against `src/core/optimization_container.jl:840-863` and the sparse spec in `src/utils/jump_utils.jl:512-518` before writing it. Use a sparse container, not dense 3-D — tranche counts are ragged, and this mirrors the existing `pwl_vars` sparse keying rather than the padded parameter-array shape.

- [ ] **Step 4: Export the new types** in `src/InfrastructureOptimizationModels.jl`, next to the existing `export AbstractPiecewiseLinearBlockOfferConstraint, …` block. No manual docs entry is needed: `docs/src/reference/public.md` is a bare `@autodocs` over the module with `Public = true`, so an exported symbol with a docstring is picked up automatically.

- [ ] **Step 5: Verify the module loads and nothing regressed.**
```sh
julia --project=test -e 'using InfrastructureOptimizationModels; println("LOAD_OK")'
julia --project=test test/runtests.jl
```
The Task 1 test should still fail (no updater exists yet); everything else must stay green, including the Aqua checks and `Test.detect_ambiguities`.

- [ ] **Step 6: Commit.** `git commit -m "Store the PWL block-offer width constraint references"`

## Task 3: Make the widths updatable, and prove it in IOM

**Files:** Modify `test/test_pwl_block_width_update.jl`.

- [ ] **Step 1: Upgrade the Task 1 test** from the objective-value proxy to a direct assertion: fetch the width container, call `JuMP.set_normalized_rhs` on a block's row with a new width, and assert `JuMP.normalized_rhs` reads back the new value. This proves the ref is reachable and mutable from a consumer's position, which is the whole point of Task 2.

- [ ] **Step 2: Add the guard the §3 answer requires.** If Task 1 found that rows are built per-period rather than to the padded maximum, add a test asserting that a period demanding more tranches than were built raises a clear error naming the component, the period, the built count and the requested count. Silent partial updating is not acceptable on this line.

- [ ] **Step 3: Run the suite and commit.** `julia --project=test test/runtests.jl` must be fully green. `git commit -m "Assert PWL block-offer widths are reachable and updatable"`

## Task 4: Wire the update in PSI and remove the false comment

**Files:** Modify `PowerSimulations.jl/src/parameters/update_cost_parameters.jl`.

- [ ] **Step 1: Replace the no-op with a real updater.** The method at `update_cost_parameters.jl:129-139` currently returns `nothing`. Implement it: derive the breakpoints from the `PSY.PiecewiseStepData` argument, resolve the width constraint container for the component type and offer direction, and `JuMP.set_normalized_rhs` each `(name, block, time_period)` row to `breakpoints[ix+1] - breakpoints[ix]`. Keep the fully-specified signature — a bare `args...` is ambiguous with the general `ObjectiveFunctionParameter` method, which is a trap already hit and fixed once in this file.

- [ ] **Step 2: Delete the false comment** at lines 121-128 and replace it with an accurate one: breakpoints re-parametrize the block-width constraint, which is why the refs are stored in IOM and the RHS is set here. A comment that misdescribes the mechanism is how this bug survived; do not leave a softened version of it.

- [ ] **Step 3: Un-fence the blocked tests.** `grep -rn "UPSTREAM-IOM-BUG" test/` lists the branches disabled because of this bug, in `test/test_market_bid_cost.jl` and `test/test_import_export_cost.jl`. Restore each to its full runner set and delete the marker comment. These are the real acceptance tests for this fix.

- [ ] **Step 4: Verify.**
```sh
julia --project=. -e 'using PowerSimulations; println("LOAD_OK")'
julia --project=test -e 'include("test/includes.jl"); include("test/test_market_bid_cost.jl")'
julia --project=test -e 'include("test/includes.jl"); include("test/test_import_export_cost.jl")'
```
Both must pass with the previously blocked branches now running. A breakpoint-varying case must now show a per-period objective that tracks ground truth instead of freezing.

- [ ] **Step 5: Commit.** `git commit -m "Update PWL block-offer widths between solves"`

## Task 5: Cross-repo verification

- [ ] **Step 1: Full PSI suite.** `julia --project=test test/runtests.jl`. Baseline to beat: 24,468 passing, 0 failed, 0 errored. Expect the count to RISE, because previously blocked branches now run. Any drop is a regression to investigate, not to accept.

- [ ] **Step 2: POM suite.** `cd ../PowerOperationsModels.jl && julia --project=test test/runtests.jl --jobs=8`. POM should need no source change, but it is the largest consumer of the block-offer path and carries ~200 exact-count assertions. Option A was chosen precisely so these do not move — so if any count changes, the fix altered the formulation and must be re-examined.

- [ ] **Step 3: IOM docs build.** `julia --project=docs docs/make.jl` must finish clean, since two exported types were added.

- [ ] **Step 4: Record the outcome.** Note in IOM's `.claude/CLAUDE.md` that block-offer widths are stored and updatable, and remove this defect from PSI's `.claude/CLAUDE.md` follow-up list, replacing it with the fixed behaviour. Update `PowerSimulations.jl/.claude/plans/excision-progress.md`.

## 5. Blast radius

- **IOM**: additive only — two new exported constraint types, one new sparse container, one changed internal signature (`add_pwl_block_offer_constraints!` gains an argument). That function is exported, so treat the signature change as breaking for any external caller; only `add_pwl_constraint_delta!` calls it inside IOM and POM.
- **POM**: expected zero source change. Its two call sites go through `add_pwl_constraint_delta!`, whose public signature is unchanged. Must still run the suite.
- **PSI**: one method implemented, one comment corrected, several test branches re-enabled.
- **In-flight IOM work does not collide.** `bf625df` ("Keep must-run units out of the PWL block offer KeyError") touches the same file but only the `min_power_offset` branch of `add_pwl_constraint_delta!`, not the per-block loop. `588e490` (MarketModel container) touches entirely different files. Rebase onto `jd/market_model` rather than `main`, since the branch is 13 commits ahead.
- **Not fixed, deliberately**: the lambda formulation stores its refs correctly but has no update path at all, and is not reachable from the market-bid flow today. If it is ever wired into a recurrent-solve path it will need the same treatment. Recorded, not in scope.

## 6. A SECOND instance of the same defect, in POM (found by Task 1)

`PowerOperationsModels.jl/src/services_models/reserve_offers.jl`, `add_reserve_offer_costs!`:

```julia
:94    JuMP.@constraint(jump_model, v <= breakpoints[k + 1] - breakpoints[k])   # ref DISCARDED
:97    cons[(service_name, dev_name, t)] = JuMP.@constraint(                    # ref stored
```

Identical shape to the IOM bug: the linking constraint's ref is kept, the per-block width ref is thrown away. This path hand-rolls its own delta variables and width rows inline rather than calling `add_pwl_block_offer_constraints!`, so **the Task 2 fix does not reach it.**

It is also **worse than the IOM case in one respect**: it reads genuinely per-period curves directly via `PSY.get_services_bid` rather than through the padded fixed-shape arrays, so it is truly ragged. That means it is plausibly the Case 2 situation the IOM path turned out not to be — a later period needing more tranches may have no row at all — and a fix there may need the loud-error guard that Task 2 does not require.

**Status: recorded, not scheduled.** It lives in a different repository, on the reserve-offer path rather than the device-offer path, and was not part of the reported defect. It needs its own assessment, including whether the ragged case can actually arise in a built model. Raise with the repository owner before touching it.
