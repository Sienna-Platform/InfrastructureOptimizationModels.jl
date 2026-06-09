# Deep Review Corrections: Correctness and Performance Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax
> for tracking. Locate code by the **quoted snippets first**, line numbers second — line numbers
> drift as edits land. Read `.claude/Sienna.md` before starting (no `isa`/`<:` branching, concrete
> struct fields, `add_*!` returns `nothing`, run the formatter after every task).

**Origin:** Full-codebase review (June 2026) by Claude Fable 5: six parallel subsystem reviews
plus a manual pass over `optimization_container.jl`; every HIGH and behavior-changing finding
below was re-verified against source line-by-line before inclusion.

**Goal:** Fix all verified correctness bugs (silently-wrong results, broken/never-working APIs,
crash paths) and performance defects (O(N²) accumulation, hot-loop allocations, abstract fields,
type instability) found in the review.

**Baseline caveat:** The review sandbox could not reach the Julia package registry, so the test
suite was NOT run during the review. **Task 0 (establish a green baseline) is mandatory before
any change.** If the baseline is not green, stop and report.

**Testing:**

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=test test/runtests.jl                                           # full suite
julia scripts/formatter/formatter_code.jl                                       # after EVERY task
```

**Commit strategy:** One commit per task (or per tightly-coupled task pair), message prefixed
`fix:`/`perf:`/`docs:`/`test:`. Tasks marked **[DECISION]** change model results — implement the
recommended default, but list them prominently in the PR description for maintainer sign-off.

---

## Phase map and parallelism

Phases 1–3 are correctness; Phase 4 is performance; Phase 5 is cleanup/docs; Phase 6 is tests.
Within a phase, tasks touching different files can run in parallel; tasks touching the same file
are marked sequential.

```
Phase 0: Task 0 (baseline)                  — first, alone
Phase 1 (silent wrong results):
  1.1 parameter_container.jl                ─┐
  1.2 nmdt_common.jl (+ nmdt.jl)            ─┤
  1.3 solver_sos2.jl + manual_sos2.jl       ─┤  parallel
  1.4 rateofchange_constraints.jl           ─┤
  1.5 jump_utils.jl (sparse to_dataframe)   ─┤
  1.6 model_numerical_analysis_utils.jl     ─┤
  1.7 dataset.jl                            ─┤
  1.8 dual_processing.jl                    ─┤
  1.9 problem_outputs.jl + optimization_problem_outputs.jl ─┘
Phase 2 (broken APIs / crash paths):
  2.1 abstract_model_store.jl               ─┐
  2.2 jump_utils.jl (to_outputs_dataframe)  ─┤  (2.2 after 1.5 — same file)
  2.3 outputs_by_time.jl                    ─┤
  2.4 initial_conditions.jl                 ─┤
  2.5 parameter_container.jl + add_param_container.jl (after 1.1 — same file)
  2.6 add_constraint_dual.jl                ─┤
  2.7 optimization_problem_outputs.jl (after 1.9 — same file)
  2.8 optimization_model_interface.jl       ─┤
  2.9 jump_utils.jl (conflict status; after 2.2)
  2.10 dataset_container.jl + emulation_model_store.jl ─┤
  2.11 optimization_container_keys.jl       ─┤
  2.12 store_common.jl                      ─┤
  2.13 emulation_model.jl + problem_template.jl ─┤
  2.14 duration_constraints.jl              ─┤
  2.15 manual_sos2.jl/sawtooth.jl/epigraph.jl/incremental.jl (manual_sos2 after 1.3)
  2.16 add_variable.jl                      ─┤
  2.17 optimization_container.jl            ─┤
  2.18 datetime_utils.jl                    ─┘
Phase 3 (decision-flagged semantic alignments): 3.1–3.4, parallel, each [DECISION]
Phase 4 (performance): 4.1–4.14, mostly parallel (see per-task file lists)
Phase 5 (cleanup/dead code/docs): 5.1–5.8
Phase 6 (tests + final verification): 6.1–6.5
```

---

## Phase 0: Baseline

### Task 0: Environment + green baseline

- [ ] `julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`
- [ ] `julia --project=test test/runtests.jl` — record the pass count (README/CI history says ~1200 tests + Aqua). If not green, STOP and report; do not start fixes on a red baseline.
- [ ] Note: `test/test_model_emulation.jl` is currently disabled ("FIXME not working and not included in the tests"). Several Phase 2 fixes are prerequisites for re-enabling it (see Task 6.2).

---

## Phase 1: Correctness — silently wrong results

These produce wrong numbers without erroring. Highest priority.

### Task 1.1: Parameter multiplier applied twice in `calculate_parameter_values`

**File:** `src/core/parameter_container.jl` (~lines 140–200)

**Problem:** The generic chain double-multiplies. `calculate_parameter_values(::ParameterAttributes, ::DenseAxisArray, ::DenseAxisArray)`:

```julia
return get_parameter_values(attributes, param_array, multiplier_array) .*
       multiplier_array
```

while the generic `get_parameter_values(::ParameterAttributes, ::DenseAxisArray, ::DenseAxisArray)` **already** multiplies:

```julia
return (.*).(jump_value.(param_array), multiplier_array)
```

→ generic parameters (NoAttributes, VariableValueAttributes, CostFunctionAttributes) come out as
`param .* mult.^2`. The `TimeSeriesAttributes` and `EventParametersAttributes` overloads of
`get_parameter_values` correctly return **unmultiplied** values, as does the SparseAxisArray
`calculate_parameter_values`. Affected call paths: `write_model_parameter_outputs!`
(`src/operation/store_common.jl:114`, per solve step), `lookup_value`
(`src/core/optimization_container.jl:1473`, external evaluators), and `read_parameters`
(`src/core/optimization_container.jl:1000-1002`). With the common `-1.0` RHS multiplier, written
parameter outputs silently flip sign (-1² = 1).

- [ ] Fix: make the generic `get_parameter_values` return raw values, establishing the invariant "`get_parameter_values` = raw, `calculate_parameter_values` = × multiplier (once)":

```julia
function get_parameter_values(
    ::ParameterAttributes,
    param_array::DenseAxisArray,
    multiplier_array::DenseAxisArray,
)
    return jump_value.(param_array)
end
```

- [ ] Check `read_parameters` (`optimization_container.jl:992-1005`): after the fix, `_calculate_parameter_values(::ParameterKey{<:ParameterType}, ...)` multiplies once (correct) and the `ObjectiveFunctionParameter` overload returns raw params (correct). No change needed there, but verify with a test.
- [ ] **Downstream caveat:** `get_parameter_values` is exported. Grep POM (if available) for `get_parameter_values(` on `ParameterContainer`s with generic attributes and note the semantic change in the PR description.
- [ ] Add test: a `ParameterContainer` with `NoAttributes`, multiplier `-1.0`, parameter `5.0` → `calculate_parameter_values` returns `-5.0` (currently `+5.0`).

### Task 1.2: NMDT/DNMDT epigraph tightening mixes normalized and unnormalized frames

**Files:** `src/quadratic_approximations/nmdt_common.jl` (`_tighten_lower_bounds!`, ~lines 244–277), callers in `src/quadratic_approximations/nmdt.jl`

**Problem:** `_tighten_lower_bounds!` builds an epigraph approximation of the **normalized**
`xh² ∈ [0,1]` (it passes `x_disc.norm_expr` with bounds `(0,1)`) and then constrains the
**unnormalized** result:

```julia
epi_cons[name, t] = JuMP.@constraint(
    jump_model,
    result_expr[name, t] >= epi_expr[name, t],
)
```

`result_expr` is in original units (`lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min`, see
~line 400-410). The cut is only valid on domain exactly [0,1]. For domain [−1,1] at x=0 it makes
the model **infeasible**; for [0,2] it collapses the lower envelope (and `tighten` simultaneously
drops McCormick lower bounds), silently and massively under-estimating x². `epigraph_depth`
defaults to `3*depth` in both `NMDTQuadConfig` and `DNMDTQuadConfig`, so this is **on by default**.
Existing tests only use (0,1) bounds, which is why it passes.

- [ ] Fix: pass the per-name bounds into `_tighten_lower_bounds!` and unnormalize the cut. For the quadratic (x·x) case with bound `b` and `lx = b.max - b.min`:

```julia
epi_cons[name, t] = JuMP.@constraint(
    jump_model,
    result_expr[name, t] >=
    b.min^2 + 2.0 * b.min * lx * x_disc.norm_expr[name, t] + lx^2 * epi_expr[name, t],
)
```

  (This is `(b.min + lx·xh)²` expanded with `xh²` replaced by its epigraph under-estimator.
  `sawtooth.jl:~192-197` shows the correct pattern — it builds the epigraph on the original
  variable with original bounds.) Check whether `_tighten_lower_bounds!` is also used by the
  **bilinear** NMDT path (`src/bilinear_approximations/nmdt.jl`); if so the unnormalization is
  `x_min·y_min + ly·x_min·yh + lx·y_min·xh + lx·ly·epi` — derive per call site, do not guess.
- [ ] Add tests on non-[0,1] domains: x ∈ [−1,1] (must be feasible at x=0 and approximate x²≈0) and x ∈ [0,2] (approximation at x=1 must be ≥ ~1−tol), for both NMDT and DNMDT with default `epigraph_depth`.

### Task 1.3: PWMCC segment-count validation is too weak — misaligned chords cut off feasible points

**Files:** `src/quadratic_approximations/solver_sos2.jl` (constructor ~lines 42–52), `src/quadratic_approximations/manual_sos2.jl` (constructor ~lines 37–48)

**Problem:** Constructors validate `pwmcc_segments <= depth`, but the PWMCC chord upper bound
(`pwmcc_cuts.jl` ~line 212, `chord_ub_cons[name, t] = @constraint(jump_model, q <= chord_rhs)`)
is valid for the SOS2 PWL value `q` only when every PWMCC boundary coincides with a PWL
breakpoint — for uniform grids, **`depth % pwmcc_segments == 0`**. Counter-example: `depth=3,
pwmcc_segments=2` on [0,1] makes the band v ∈ (4/9, 5/9) MIP-infeasible (PWL forces q = v − 2/9 =
0.278 at v=0.5; both chords cap q at 0.25).

- [ ] Fix both constructors: replace the `pwmcc_segments > depth` check with

```julia
if pwmcc_segments != 0 && depth % pwmcc_segments != 0
    throw(ArgumentError(
        "pwmcc_segments must evenly divide depth so PWMCC boundaries coincide with " *
        "PWL breakpoints (got pwmcc_segments=$(pwmcc_segments), depth=$(depth))"))
end
```

- [ ] Add a constructor test asserting `SolverSOS2QuadConfig(depth=3, pwmcc_segments=2)` throws and `(depth=4, pwmcc_segments=2)` is accepted; same for `ManualSOS2QuadConfig`.

### Task 1.4: t=1 ramp big-M omits initial status in the OnStatusParameter path

**File:** `src/common_models/rateofchange_constraints.jl` (~lines 228–250)

**Problem:** In the OnStatusParameter (UC-commitment-as-parameter) ramp builder, t ≥ 2 uses
`big_m = power_limits.max * (2 - yprev - ycur)` (relaxed when off in either period), but t = 1 uses

```julia
big_m = power_limits.max * (1 - ycur)
```

— no initial-status term. A unit the upstream UC starts at t=1 (`ycur=1`, prior status off,
`ic_power≈0`) gets `big_m = 0`, so `var[1] − 0 ≤ r_up·dt + slack` is fully binding; with
`Pmin > r_up·dt` (common for thermal units) the dispatch model is **infeasible** (or slack-priced),
while the identical start at t=2 is correctly relaxed.

- [ ] Fix: mirror the t≥2 formula using initial status. `ic_power = ic_power_by_name[name]` is `Float64` in non-recurrent builds and `JuMP.VariableRef` in recurrent builds, so derive the status without branching on runtime values where impossible:
  - Preferred: look up the device's initial **status** initial condition if one exists in the container (search how `_get_initial_condition_type` maps; POM registers a status IC for UC formulations). Then `big_m = power_limits.max * (2 - y_init - ycur)` (works as an `AffExpr` when `y_init`/`ycur` are VariableRefs).
  - If no status IC is available in this path, derive from power for the `Float64` case (`y_init = ic_power > ABSOLUTE_TOLERANCE ? 1.0 : 0.0` — add a small dispatch helper, not an inline `isa`) and **document + assert** the recurrent-case behavior rather than silently keeping the wrong bound.
- [ ] Add test: 2-timestep model, unit off initially (`ic_power = 0`), `on_status[name,1] = 1`, `Pmin > ramp_up·dt` → model must remain feasible with `var[1] ≥ Pmin` (currently infeasible).

### Task 1.5: SparseAxisArray → DataFrame conversion misaligns column labels and breaks on sparse keys

**File:** `src/utils/jump_utils.jl` (`to_dataframe(::SparseAxisArray, key)` ~line 233, `_to_matrix` ~lines 83–93, `to_matrix` ~line 99, `get_column_names_from_axis_array(::SparseAxisArray)` ~lines 175–183)

**Problem (two coupled defects):**
1. `to_dataframe(array::SparseAxisArray, key)` pairs matrix columns ordered by `sort!(unique!(raw key tuples))` with labels ordered by sorted **encoded strings** — these diverge for mixed-type tuples (`("a",2) < ("a",10)` but `"a__10" < "a__2"`) → **silently mislabeled CSV exports** (export path: `_export_container_output!` in `store_common.jl`). The repo documents this exact hazard in `decision_model_store.jl:128-135` and fixed it there only.
2. `_to_matrix` indexes rows by the raw time **value** (`data[t, ix]`) and assumes a rectangular key set (`array.data[(col..., t)]` KeyErrors on genuinely sparse data).

- [ ] Fix `to_dataframe(::SparseAxisArray, key)` to derive ordering once, mirroring `decision_model_store.jl:144-146`:

```julia
function to_dataframe(array::SparseAxisArray{T, N, K}, key::OptimizationContainerKey) where {T, N, K}
    tuple_columns = sort!(unique!([k[1:(N - 1)] for k in keys(array.data)]))
    return DataFrame(_to_matrix(array, tuple_columns), encode_tuple_to_column.(tuple_columns))
end
```

- [ ] Fix `_to_matrix` row indexing and sparsity:

```julia
sorted_times = sort!(collect(Set{Int}(k[N] for k in keys(array.data))))
row_index = Dict(t => i for (i, t) in enumerate(sorted_times))
data = Matrix{Float64}(undef, length(sorted_times), length(columns))
for (ix, col) in enumerate(columns), t in sorted_times
    data[row_index[t], ix] = get(array.data, (col..., t), NaN)
end
```

- [ ] Add test: a SparseAxisArray with keys `("a",2,1),("a",10,1)` round-trips with values under the correct labels; one with time keys `{2,4}` doesn't throw.

### Task 1.6: `update_numerical_bounds` mixes signed values with abs comparisons

**File:** `src/operation/model_numerical_analysis_utils.jl` (~lines 64–75)

**Problem:**

```julia
if v.min > abs(value)
    set_min!(v, value)        # stores SIGNED value against an abs comparison
    set_min_index!(v, idx)
elseif v.max < abs(value)     # elseif: first value can never initialize max
    set_max!(v, value)
```

A stored negative min poisons all subsequent comparisons and the `max - min > 1e9` conditioning
checks in `optimization_model_interface.jl` warn spuriously or not at all.

- [ ] Fix:

```julia
a = abs(value)
if v.min > a
    set_min!(v, a); set_min_index!(v, idx)
end
if v.max < a
    set_max!(v, a); set_max_index!(v, idx)
end
```

- [ ] While in this file, add the missing `isassigned` guard to `get_detailed_constraint_numerical_bounds` (`optimization_debugging.jl` ~lines 77-78) to match its non-detailed twin (`model_numerical_analysis_utils.jl:122`).

### Task 1.7: `set_value!(::InMemoryDataset{2}, ::DenseAxisArray{Float64,2}, index)` slices the incoming array by the store row index

**File:** `src/core/dataset.jl` (~lines 177–180)

**Problem:** `s.values[:, index] = vals[:, index]` — `index` is the execution counter, not a column
of `vals`. At execution 3 it writes the 3rd time column of the current solve; once
`index > size(vals, 2)` it's a BoundsError. PSI guarded this with `@assert_op size(array)[2] == 1`;
the port replaced the guard with silently wrong behavior. (The comment above the method says
incoming vals are single-step.)

- [ ] Fix:

```julia
function set_value!(s::InMemoryDataset{2}, vals::DenseAxisArray{Float64, 2}, index::Int)
    IS.@assert_op size(vals, 2) == 1
    s.values[:, index] = vals[:, 1]
    return
end
```

### Task 1.8: Dual processing crashes on fixed integers and converts integer variables to binary

**File:** `src/core/dual_processing.jl`

**Problem 1 (~line 62):** `cache[key][:fixed_int_value] = jump_value.(v)` — `v` is the loop
variable of an **earlier, closed** loop; this is an `UndefVarError` whenever a fixed integer
variable is encountered.
**Problem 2 (~lines 102–106):** the restore loop runs `JuMP.unfix.(variable); JuMP.set_binary.(variable)`
unconditionally — general-integer variables are re-declared **binary** after dual processing
(the correct logic sits in the commented-out block right below); bounds are also not restored.

- [ ] Fix line 62: `cache[key][:fixed_int_value] = var_cache[key]` (the values were just cached).
- [ ] Fix the restore loop using the cached `:integer` flag and bounds (resurrect the commented block):

```julia
JuMP.unfix.(variable)
if haskey(cache[key], :lb)
    JuMP.set_lower_bound.(variable, cache[key][:lb])
end
if haskey(cache[key], :ub)
    JuMP.set_upper_bound.(variable, cache[key][:ub])
end
if cache[key][:integer]
    JuMP.set_integer.(variable)
else
    JuMP.set_binary.(variable)
end
if haskey(cache[key], :fixed_int_value)
    JuMP.fix.(variable, cache[key][:fixed_int_value]; force = true)
end
```

  Delete the now-dead commented block.
- [ ] While here (perf, same file): the first loop's SparseAxisArray branch (~lines 24–28) broadcasts `jump_value.(v)` and then re-writes every element in a loop — delete the redundant per-index loop; the broadcast alone is correct for both branches (drop the `isa` split entirely).
- [ ] Add test: MILP mock with one general-integer variable → after `process_duals`, `JuMP.is_integer(var)` is still true and bounds are intact.

### Task 1.9: Emulation outputs timestamps collapse to a single repeated initial_time

**Files:** `src/operation/problem_outputs.jl` (~line 74), `src/core/optimization_problem_outputs.jl` (`_process_timestamps`, ~lines 489–493)

**Problem:** `OptimizationProblemOutputs(::EmulationModel)` stores
`StepRange(initial_time, get_resolution(model), initial_time)` — one element — and the read path
papers over it with `repeat(get_timestamps(res), def_len)`, so every execution row is labeled
`initial_time`. Any `start_time > initial_time` then makes `findfirst` return `nothing` →
`MethodError` on `collect(nothing:def_len)`.

- [ ] Fix `problem_outputs.jl:74` to store the real range over written executions, e.g. `range(initial_time; step = get_resolution(model), length = <executions written to store>)` (derive the count from the store / `get_num_executions`).
- [ ] Remove the `repeat(...)` special case in `_process_timestamps` and make a too-short timestamp vector an explicit error.
- [ ] Covered by re-enabled emulation tests (Task 6.2).

---

## Phase 2: Correctness — broken APIs and crash paths

Verified never-working or crash-on-use code. All are straightforward; each needs a regression test
(most have none — that's why they shipped broken).

### Task 2.1: `EmulationModelStore` breaks all `list_keys`/`list_fields`/`get_value` (and `OptimizationProblemOutputs(::EmulationModel)`)

**File:** `src/core/abstract_model_store.jl` (~lines 75–99)

**Problem:** The three `@generated` functions emit `getfield(store, $field)`, bypassing the
`get_data_field` indirection that `EmulationModelStore` relies on (its containers live inside
`data_container`). `getfield(::EmulationModelStore, :variables)` → `ErrorException`. This breaks
`list_*_keys`, `list_*_names`, and the `OptimizationProblemOutputs(::EmulationModel)` constructor.

- [ ] Fix all three generated bodies to route through the overridable accessor:

```julia
return :(return collect(keys(get_data_field(store, Val($field)))))   # list_keys
return :(return keys(get_data_field(store, Val($field))))            # list_fields
return :(return get_data_field(store, Val($field))[$K(T, U)])        # get_value
```

  Confirm `EmulationModelStore` defines `get_data_field(store, ::Val{S})` overrides (see
  `emulation_model_store.jl:11-17`); add `Val`-typed overrides there if only the Symbol form exists.
- [ ] Test: `list_variable_keys(::EmulationModelStore)` on a store with one written key.

### Task 2.2: `to_outputs_dataframe` family — four broken methods in the public read path

**File:** `src/utils/jump_utils.jl` (after Task 1.5; same file)

Verified defects (independently found by two reviewers):
1. ~line 252: generic entry `to_outputs_dataframe(array, timestamps)` ends with `Val(TableFormat.LONG))()` — the trailing `()` calls the returned DataFrame → always throws. **Fix: delete the trailing `()`.**
2. ~lines 431–441: the 3-D `(Vector{String}, Vector{String}, UnitRange)` + `::Nothing` method shadows the working parametric method at ~395 and passes the raw enum (`TableFormat.LONG`) instead of `Val(...)` → `MethodError` on every `read_variable`/`read_dual`/`read_expression` of a 3-D output. **Fix: delete the entire method** (the parametric method at ~395 covers it; "fixing" it would self-recurse).
3. ~lines 256–266: 1-D LONG method hardcodes `:DateTime => [1]` (DimensionMismatch for >1 name), ignores `timestamps`, and uses `:DateTime` where the `::Nothing` convention is `:time_index`. **Fix:** mirror the 2-D variants — `::Nothing` → `:time_index => fill(1, length(array))`; with timestamps → `:DateTime => fill(only(timestamps), length(array))`. Add the missing `::Nothing` 1-D method if dispatch requires it.
4. Method gap: the with-timestamps 3-D method (~line 350) only accepts `Tuple{Vector{String}, Vector{String}, UnitRange{Int}}` while the `::Nothing` variant accepts `T <: Union{Vector{String}, UnitRange{Int}}` for axis 2 — `DecisionModelStore` stores `(Vector{String}, UnitRange, UnitRange)` arrays and `make_dataframe(::OutputsByTime{...,3})` passes real timestamps → `MethodError`. **Fix:** widen ~350's signature to the same parametric form; type `name2_col` via `eltype(T)`.
5. ~lines 443–446: `to_dataframe(array::SparseAxisArray)` (key-less) passes a tuple of encoded strings where `_to_matrix` needs raw key tuples, then calls a nonexistent `DataFrame(::Matrix, ::Tuple)` — triple-broken, zero callers. **Fix: delete it** (Task 1.5's keyed version is the supported path).

- [ ] Implement fixes 1–5.
- [ ] In the 2-D/3-D LONG converters (~lines 287–292, 375–385): iterate `enumerate(axes(array, 2))` and index `timestamps_arr[j]` by position, not by axis value (axes like `0:23` or `2:25` currently misindex); in the 3-D loop, read from `array.data` positionally instead of per-element axis-keyed `array[name, name2, time_index]` lookups.
- [ ] Tests: 1-D single/multi name; 3-D `(String,String,Int)` and `(String,Int,Int)` with and without timestamps; 2-D with a `0:23` integer axis.

### Task 2.3: `OutputsByTime` — dead validation, self-referential kwarg default, broken Matrix path

**File:** `src/core/outputs_by_time.jl`

Three verified defects:
1. ~lines 8–16: the "validating" outer constructor calls `OutputsByTime(key, data, resolution, column_names)` — for `NTuple` column_names the **implicit** constructor (more specific) wins, so `_check_column_consistency` is never executed (all 4 methods dead); for any other column_names type it recurses into itself → StackOverflowError.
2. ~line 149: `make_dataframes(outputs; table_format::TableFormat = table_format)` — self-referential default → `UndefVarError` when the kwarg is omitted.
3. ~line 132: `DataFrames.DataFrame(array, outputs.column_names)` passes an `NTuple` where a vector is required → MethodError for every `make_dataframe(::OutputsByTime{Matrix{Float64}})`.

- [ ] Fix 1: convert to an inner constructor (mirroring PSI's `ResultsByTime`):

```julia
mutable struct OutputsByTime{T, N}
    key::OptimizationContainerKey
    data::SortedDict{Dates.DateTime, T}
    resolution::Dates.Period
    column_names::NTuple{N, Vector{String}}

    function OutputsByTime(key, data::SortedDict{Dates.DateTime, T}, resolution, column_names) where {T}
        _check_column_consistency(data, column_names)
        new{T, length(column_names)}(key, data, resolution, column_names)
    end
end
```

3. - [ ] Fix 2: `table_format::TableFormat = TableFormat.LONG`.
- [ ] Fix 3: `DataFrames.DataFrame(array, outputs.column_names[1])`.
- [ ] Tests: constructor rejects mismatched column names; `make_dataframes(outputs)` (no kwarg) works; Matrix-backed `make_dataframe` works.

### Task 2.4: Key-based `InitialCondition` constructor instantiates an impossible type

**File:** `src/core/initial_conditions.jl` (~lines 20–30)

**Problem:** `InitialCondition{T, U}(component, value)` where `U` is the **component** type —
violates the struct bound `U <: Union{JuMP.VariableRef, Float64, Nothing}` → `TypeError` on every
call. Exported API (latent; POM-facing).

- [ ] Fix: `return InitialCondition{T, V}(component, value)`.
- [ ] Test: `InitialCondition(InitialConditionKey(SomeICType, MockThermalGen), mock_component, 1.0)` constructs.

### Task 2.5: `EventParametersAttributes` is unconstructible as used

**Files:** `src/core/parameter_container.jl` (~lines 83–94, after Task 1.1), `src/common_models/add_param_container.jl` (~line 99)

**Problem:** The struct has phantom type parameters `{T, U}` (neither appears in a field), no
explicit constructor, and its only call site passes a component **type**:
`attributes = EventParametersAttributes(V)` → `MethodError` on every
`add_param_container!` for `T <: EventParameter` (exported; `range_constraint.jl` consumes these
containers).

- [ ] Fix: give the struct a working constructor consistent with `get_param_type` usage:

```julia
struct EventParametersAttributes{T <: IS.InfrastructureSystemsComponent, U <: ParameterType} <: ParameterAttributes
    affected_devices::Vector{T}
end
EventParametersAttributes(::Type{T}, ::Type{U}) where {T <: IS.InfrastructureSystemsComponent, U <: ParameterType} =
    EventParametersAttributes{T, U}(T[])
```

  and update the call site to `EventParametersAttributes(V, T)`.
- [ ] Test: `add_param_container!` with a mock `EventParameter` type succeeds and `get_param_type(attrs)` returns the parameter type.

### Task 2.6: `_validate_keys` interpolates out-of-scope variables

**File:** `src/common_models/add_constraint_dual.jl` (~lines 76–81)

**Problem:** The error message interpolates `$constraint_type` and `$D`, neither in scope →
`UndefVarError` instead of the intended `InvalidValue` whenever a dual is requested for an
unbuilt constraint.

- [ ] Fix:

```julia
_validate_keys(keys, ::Type{T}, ::Type{D}) where {T, D} =
    isempty(keys) && throw(IS.InvalidValue(
        "No constraint of type $T for $D is stored; cannot assign a dual variable."))
```

  and update both call sites (`_validate_keys(constraint_keys, constraint_type, D)`).
- [ ] Test: requesting a dual for a never-built constraint type throws `IS.InvalidValue` with the type names in the message.

### Task 2.7: `optimization_problem_outputs.jl` — three broken error/IO paths + dropped kwargs

**File:** `src/core/optimization_problem_outputs.jl` (after Task 1.9; same file)

- [ ] ~lines 340–346 (`set_source_data!`): message interpolates undefined `$sys_uuid` and `$(res.source_uuid)`; the locals/fields are `source_uuid` / `source_data_uuid`. Fix the interpolations so a UUID mismatch produces the intended `InvalidValue`.
- [ ] ~lines 443–447 (`_read_outputs`): the row-drop diagnostic interpolates `$(outputs[key])` before `outputs[key]` is assigned → KeyError masks the message. Interpolate `$(df)` instead.
- [ ] ~lines 1066–1067 (`export_optimizer_stats`, JSON branch): references unimported `JSON` (only `JSON3` is a dependency) → `UndefVarError` for the documented `format="json"`. Fix: `open(joinpath(directory, "optimizer_stats.json"), "w") do io; JSON3.write(io, to_dict(data)); end`.
- [ ] ~lines 972–974 (`read_expressions(res::Outputs; kwargs...)`): drops `kwargs` instead of forwarding to `read_expression(res, x; kwargs...)` like its four siblings. Forward them.
- [ ] Tests for the JSON export and `read_expressions(res; len=...)` forwarding.

### Task 2.8: Mutating two-arg `get_simulation_info`

**File:** `src/operation/optimization_model_interface.jl` (~line 190)

**Problem:** `get_simulation_info(model::AbstractOptimizationModel, val) = model.simulation_info = val`
— a `get_` accessor that mutates; the intended `set_simulation_info!` does not exist anywhere.

- [ ] Fix: rename to `set_simulation_info!(model::AbstractOptimizationModel, val)` ending with bare `return`. Grep repo + tests for two-arg `get_simulation_info` callers (none in-repo at review time) and note the rename in the PR for POM.

### Task 2.9: Sparse `check_conflict_status` can never dispatch, and iterates values

**File:** `src/utils/jump_utils.jl` (~lines 674–687, after Task 2.2)

**Problem:** Sparse constraint containers have eltype `Union{Nothing, ConstraintRef}` (see
`sparse_container_spec`), so the `SparseAxisArray{JuMP.ConstraintRef}` signature never matches →
`MethodError`, swallowed by `compute_conflict!`'s try/catch and misreported as "solver doesn't
support IIS" for any model with sparse constraint containers. The body also destructures
`(index, constraint)` from an iterator that yields values only.

- [ ] Fix:

```julia
function check_conflict_status(
    jump_model::JuMP.Model,
    constraint_container::SparseAxisArray,
)
    conflict_indices = Vector{eltype(keys(constraint_container.data))}()
    for (index, constraint) in pairs(constraint_container.data)
        constraint === nothing && continue
        if MOI.get(jump_model, MOI.ConstraintConflictStatus(), constraint) != MOI.NOT_IN_CONFLICT
            push!(conflict_indices, index)
        end
    end
    return conflict_indices
end
```

- [ ] Also type the dense variant's accumulator (currently `Vector()` ⇒ `Vector{Any}`).
- [ ] Unit-test the function directly with a mock/dummy (CI solvers lack IIS support — test dispatch and iteration, not the MOI attribute).

### Task 2.10: `Base.empty!(::DatasetContainer)` and `EmulationModelStore` empty!/isempty dead branches

**Files:** `src/core/dataset_container.jl` (~lines 19–28), `src/operation/emulation_model_store.jl` (~lines 42–50, 67–72)

- [ ] `dataset_container.jl`: `empty!(val)` is called on dataset values but no `Base.empty!` exists for any `AbstractDataset` → MethodError on first non-empty call. Fix: `empty!(field_dict)` per field (drop the inner loop), matching the intent of resetting the container.
- [ ] `emulation_model_store.jl` `Base.empty!`: the `name in [:values, :timestamps]` membership test is always true for actual `DatasetContainer` fields, the `elseif :update_timestamp` branch is unreachable and would write a nonexistent field, and `Base.isempty`'s else branch has inverted logic (`iszero(val) && return false`). Rewrite both: iterate `fieldnames(DatasetContainer)`, `empty!` each dict, `empty!(store.optimizer_stats)`; `isempty` = all dicts empty && stats empty.
- [ ] Tests for `empty!`/`isempty` round-trip on a store with data.

### Task 2.11: Key meta validation shadowed; `Base.convert` methods reference undefined `decode_symbol`

**File:** `src/core/optimization_container_keys.jl`

- [ ] ~lines 81–100: the `meta::String` constructor method skips `check_meta_chars`, and since meta is always a String in practice, the validating method (meta untyped, ~81–90) is dead — metas containing `__` (`COMPONENT_NAME_DELIMITER`) silently corrupt the encode/decode round-trip of stored outputs. Fix: delete the redundant `meta::String` method (the untyped one handles it and validates), or add `check_meta_chars(meta)` to it. Also add validation to `make_key` (~102–113).
- [ ] ~lines 161–162: `Base.convert(::Type{ExpressionKey}, name::Symbol) = ExpressionKey(decode_symbol(name)...)` (and the ConstraintKey twin) call `decode_symbol`, which **does not exist** anywhere in the package → delete both methods (or implement decode; grep first — no in-repo callers).
- [ ] Test: `ConstraintKey(MockConstraint, MockComponent, "bad__meta")` throws `InvalidValue`.

### Task 2.12: Export path broken for emulation indices

**File:** `src/operation/store_common.jl` (~lines 10–23)

**Problem:** `_export_container_output!` does
`range(index; length = horizon_count, step = resolution::Millisecond)`; for emulation,
`index::Int` → `MethodError: +(::Int64, ::Millisecond)`. Even working, an Int index is not a
timestamp.

- [ ] Fix: dispatch on index type — `DecisionModelIndexType` (DateTime) keeps current behavior; for `EmulationModelIndexType` (Int) compute the timestamp (`initial_time + (index - 1) * resolution`, threaded through from the caller which already has `update_timestamp`) or use `update_timestamp` directly with `length = 1`.

### Task 2.13: `EmulationModel` mutates/aliases the caller's template; `_deepcopy_template` not exception-safe

**Files:** `src/operation/emulation_model.jl` (~line 37), `src/operation/problem_template.jl` (~lines 141–150)

- [ ] `emulation_model.jl`: mirror `DecisionModel` — `template_ = _deepcopy_template(template); finalize_template!(template_, sys)` and store `template_` (currently finalizes and stores the caller's template in place; two models share mutable state).
- [ ] `problem_template.jl`: `_deepcopy_template` nulls `PTDF_matrix`/`MODF_matrix` on the **input**, deepcopies, then restores — if `deepcopy` throws, the user's template is left stripped. Wrap restore in `try ... finally`.

### Task 2.14: Duration constraints: container axis built from up-column ICs, down-column written

**File:** `src/common_models/duration_constraints.jl` (`device_duration_retrospective!`, ~lines 50–101)

**Problem:** `device_name_sets` filters non-`nothing` ICs of `initial_duration[:, 1]` (up), but
the down loop iterates `initial_duration[:, 2]` with its own filter and writes
`con_down[name, t]`. Up-IC `nothing` + down-IC set → KeyError; up set + down `nothing` →
permanently `#undef` rows in `con_down` (UndefRefError on any full container read).

- [ ] Fix: build one filtered name set per column and pass each to its own `add_constraints_container!` call; or filter on both columns jointly if the formulation requires both. Check the other three duration functions (~158, ~257, ~379) — they use the unfiltered column-1 axis and have the undef-row half of the hazard; align them.
- [ ] Test: matrix of ICs where one device has up-IC `nothing`, another has down-IC `nothing` — constraint containers must have consistent axes and no `#undef` entries.

### Task 2.15: Approximation container bookkeeping (manual SOS2 / sawtooth / epigraph / incremental)

**Files:** `src/quadratic_approximations/manual_sos2.jl` (after Task 1.3), `sawtooth.jl`, `epigraph.jl`, `incremental.jl` — parallel-safe across files

- [ ] `manual_sos2.jl` ~lines 250–253: adjacency loop stores at `adj_cons[name, i + 1, t]`, leaving index 2 `#undef` and double-writing index `n_points`. Fix: store at `[name, i, t]`. (JuMP model is correct; container reads crash.)
- [ ] `sawtooth.jl` ~lines 252–267: `mip_cons[name, 1:4, t]` written inside `for j in alpha_levels` → only the last level's references survive. Fix: add the level axis — container axes `(names, alpha_levels, 1:4, time_steps)`, index `[name, j, c, t]`.
- [ ] `epigraph.jl` ~lines 185–193: same per-level overwrite for `lp_cons` (axes `(names, 1:2, time_steps)` written inside `for j in 1:depth`). Fix: add the level axis.
- [ ] `incremental.jl`:
  - ~lines 88–95: δ variables get bounds only from optional extension hooks that default to `nothing` — the formulation requires δ ∈ [0,1] (ordering constraints `δ_{i+1} ≤ z_i ≤ δ_i` enforce no lower bound by themselves) → unbounded δ silently leaves the PWL curve. Fix: default to `set_lower_bound(0.0)`/`set_upper_bound(1.0)` when the hooks return `nothing`.
  - ~lines 235–240: ordering constraints are anonymous (never stored). Store them in a constraints container per repo convention.
  - ~lines 77–95 (perf, same edit): delete the dead `pwlvars` array; hoist `ub`/`lb` hook calls out of the per-segment loop; keep the created `VariableRef` in a local instead of re-looking up `var_container[name, i, t]`.
- [ ] Tests: container completeness (`all(i -> isassigned(...), ...)`) for manual SOS2 and the level-indexed containers; incremental approximation value stays on the curve for an interior x.

### Task 2.16: Service variables ignore the warm-start setting

**File:** `src/common_models/add_variable.jl` (~lines 108–124)

**Problem:** `add_service_variables!` sets start values unconditionally; the device twin gates on
`get_warm_start(settings)`.

- [ ] Fix: hoist `settings = get_settings(container)`; apply the same `if get_warm_start(settings)` gate. (Loop restructure is Task 4.4; keep this one minimal or fold into 4.4 if done together.)
- [ ] Also note the lb/ub asymmetry at ~117–120 (`lb` guarded by `!binary`, `ub` not) in the PR for maintainer review — do not change behavior without confirmation.

### Task 2.17: `get_objective_expression(::ObjectiveFunction)` is non-idempotent (mutates `variant_terms`)

**File:** `src/core/optimization_container.jl` (~lines 32–47)

**Problem:** The AffExpr branch does
`JuMP.add_to_expression!(v.variant_terms, v.invariant_terms)` — calling the exported getter twice
between variant resets double-counts the invariant terms in the objective. The QuadExpr branch
already builds a fresh temp (non-mutating); the asymmetry is the hazard.

- [ ] Fix: mirror the QuadExpr branch:

```julia
else
    temp_expr = JuMP.AffExpr()
    JuMP.add_to_expression!(temp_expr, v.variant_terms)
    return JuMP.add_to_expression!(temp_expr, v.invariant_terms)
end
```

  (One extra allocation per solve step — negligible; remove the stale "will mutate the variant
  terms" comment.)
- [ ] Test: call `get_objective_expression(obj_fn)` twice; assert equal results.

### Task 2.18: `find_timestamp_index` — BoundsError on length-1 input, silent floor of off-grid timestamps

**File:** `src/utils/datetime_utils.jl` (~lines 12–21)

- [ ] Guard the stride computation with `length(dates) >= 2` so a length-1 vector reaches the descriptive error path instead of `dates[2]` BoundsError.
- [ ] Validate alignment before returning: compute the index, then check `first(dates) + (index - 1) * dates_resolution == date`, else throw the descriptive error (dataset state reads currently floor silently to the previous row).
- [ ] Tests: length-1 vector + non-matching date; off-grid date errors; exact date returns the right index.

---

## Phase 3: [DECISION] semantic alignments (change model results)

Implement the recommended default; flag each in the PR description for maintainer sign-off. All
are small diffs; the cost is deciding, not coding.

### Task 3.1 [DECISION]: Quadratic FuelCurve applies `objective_function_multiplier`, contradicting its own comment and the linear path

**File:** `src/objective_function/quadratic_curve.jl` (~lines 226, 246–256)

Directly above `_add_fuel_quadratic_variable_cost!` sits the comment "Multiplier is not necessary
here. There is no negative cost for fuel curves." — yet the call passes
`multiplier * proportional_term_per_unit, multiplier * quadratic_term_per_unit`. The linear
fuel path (`linear_curve.jl` ~60–64) follows the comment and never applies the multiplier. Any
formulation with multiplier −1 flips quadratic fuel cost signs but not linear ones.

- [ ] **Recommended:** drop `multiplier *` from both arguments (align with the linear path and the comment). Grep POM for `objective_function_multiplier` overrides returning −1 used with quadratic fuel curves; note in PR.

### Task 3.2 [DECISION]: Proportional (no-load/fixed) costs are not scaled by `dt`

**File:** `src/objective_function/proportional.jl` (~line 28 and the same pattern in `add_proportional_cost_maybe_time_variant!` ~line 72)

`rate = cost_term * multiplier` multiplies the variable at every timestep with no resolution
scaling, while the sibling helper for variable/VOM costs applies
`dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR` (`cost_term_helpers.jl`
~220–222). At 15-min resolution, proportional costs are over-weighted 4× relative to energy costs
in the same objective. This mirrors legacy PSI behavior, so changing it alters results vs. PSI.

- [ ] **Recommended:** apply the same `dt` factor in both functions for internal consistency, and add a CHANGELOG/PR note ("objective values change for sub-hourly resolutions; was inconsistent with energy-cost scaling"). If maintainers veto, instead document loudly on `proportional_cost` that it must return $/timestep and add a comment at both sites.

### Task 3.3 [DECISION]: `is_nontrivial_offer` returns `false` for time-series-backed curves

**File:** `src/objective_function/value_curve_cost.jl` (~line 179)

`is_nontrivial_offer(::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}) = false` inverts the
documented meaning ("is this carrying meaningful data, as opposed to the ZERO_OFFER_CURVE
placeholder") — a TS-backed curve always carries real data. Any load-formulation gate
`is_nontrivial_offer(curve) || skip` silently drops time-varying decremental offer costs.

- [ ] **Recommended:** return `true`; check POM's load-formulation call sites to confirm the TS path is handled downstream of the gate. If the intent was "TS curves gated elsewhere", rename the predicate instead — the docstring and method must agree.

### Task 3.4 [DECISION]: Breakpoint offer-curve parameters classified as `TimeSeriesParameter` (VariableRef storage) while every reader requires Float64

**File:** `src/objective_function/offer_curve_types.jl` (~lines 25, 38)

`AbstractPiecewiseLinearSlopeParameter <: ObjectiveFunctionParameter` but
`AbstractPiecewiseLinearBreakpointParameter <: TimeSeriesParameter`. The `TimeSeriesParameter`
`add_param_container!` dispatch uses `get_param_eltype(container)` → `JuMP.VariableRef` in
recurrent-solve builds, but all IOM readers (`_fill_pwl_data_from_arrays!`,
`_get_raw_pwl_data` in `value_curve_cost.jl`) are typed `DenseAxisArray{Float64}` → a
time-varying MarketBidCost in a standard simulation build fails at objective construction.

- [ ] **Recommended:** reparent `AbstractPiecewiseLinearBreakpointParameter <: ObjectiveFunctionParameter` (Float64 storage, matching its slope twin and the rebuild-variant-terms-each-step architecture). **Coordinate with POM** — POM implements `add_parameters!` for these types; verify which `add_param_container!` signature POM calls.
- [ ] Related (document, do not fix blind): the TS delta-PWL block-width constraints are built from parameter *values* at build time as anonymous constraints (`objective_function_pwl_delta.jl` ~145–147 via `value_curve_cost.jl` ~292–311) — nothing can update them across simulation steps, so δ-block widths go stale after step 1 unless the model is rebuilt. Add a tracking issue + a loud comment; the proper fix (store width constraints in a sparse container indexed `(name, segment, t)` and `JuMP.set_normalized_rhs` them from the parameter-update path) needs the POM update hook and is out of scope for a mechanical pass.

---

## Phase 4: Performance

### Task 4.1: O(N²) quadratic objective accumulation

**File:** `src/core/optimization_container.jl` (~lines 1189–1197)

`add_to_objective_invariant_expression!(container, cost_expr::JuMP.GenericQuadExpr)` uses
`invariant_terms += cost_expr`, copying the entire accumulated expression per call — called once
per device per timestep by `quadratic_curve.jl`. Total build work is O(N²) in quadratic terms.

- [ ] Fix with a one-time promotion + in-place adds:

```julia
function add_to_objective_invariant_expression!(
    container::OptimizationContainer,
    cost_expr::JuMP.GenericQuadExpr,
)
    obj = container.objective_function
    invariant = obj.invariant_terms
    if invariant isa JuMP.GenericAffExpr   # one-time promotion of the union-typed field
        promoted = JuMP.GenericQuadExpr{Float64, JuMP.VariableRef}()
        JuMP.add_to_expression!(promoted, invariant)
        obj.invariant_terms = promoted
        invariant = promoted
    end
    JuMP.add_to_expression!(invariant, cost_expr)
    return
end
```

  (The `isa` guards a 2-member concrete-union field promotion — compile-time union split, the
  accepted exception per the 2026-04-06 plan precedent. Remove the now-stale comment at ~1189.)
- [ ] Benchmark sanity check: build time for a model with ~100 quadratic-cost devices × 24 steps should drop measurably; at minimum assert tests pass.

### Task 4.2: Non-`const` exported enum aliases + untyped `ENUM_MAPPINGS`

**File:** `src/core/definitions.jl` (~lines 156–174)

`ModelBuildStatus = ...`, `SimulationBuildStatus = ...`, `RunStatus = ...` are non-const globals
read in every solve-status check (`execute_optimizer!`, `solve_model!`, `is_built`).

- [ ] Add `const` to all three. Type the mappings: `const ENUM_MAPPINGS = Dict{DataType, Dict{String, Any}}()` (or per-enum value types).

### Task 4.3: `jump_value` MOI round-trips per element in per-step store writes

**Files:** `src/utils/jump_utils.jl` (~lines 21–29), `src/operation/store_common.jl` (broadcast sites ~86, 152, 184, 222)

`jump_value(::VariableRef)` calls `JuMP.is_fixed` and `JuMP.has_values(input.model)` (a
solver-attribute query) per element, N×T times per step.

- [ ] Add a model-level fast path: in the `write_model_*_outputs!` functions, query `has_vals = JuMP.has_values(jump_model)` once per call; add `jump_value(v::JuMP.VariableRef, has_vals::Bool)` (or broadcast a prefetched closure-free two-arg form) keeping the per-element `is_fixed` branch. With `!has_vals`, `fill(NaN, ...)` directly. Keep the 1-arg method for general use; also change `input.model` to `JuMP.owner_model(input)`.

### Task 4.4: `add_variables!` / `add_service_variables!` loop hygiene

**File:** `src/common_models/add_variable.jl` (with Task 2.16)

The most-executed builder in the package re-evaluates `get_variable_upper_bound/lower_bound/warm_start_value`
per timestep (they depend only on the device), re-reads `get_warm_start(settings)` N×T times,
re-fetches `get_jump_model(container)` inside the macro per iteration, and re-does the keyed
`variable[name, t]` lookup for each `set_*` call.

- [ ] Restructure both functions: device-outer loop; hoist `jump_model`, `name`, `ub`, `lb`, `init`(gated), service name string; bind `v = variable[name, t] = JuMP.@variable(jump_model, ...)` and call `set_*(v, ...)`.

### Task 4.5: Range-constraint inner loops

**File:** `src/common_models/range_constraint.jl`

- [ ] Device-invariant hoisting at all five `for device in devices, t in time_steps` sites (~103–107, ~222–227, ~270–276, ~465–468, ~495–498): move `IS.get_name(device)`, `get_min_max_limits(...)`/`get_max_active_power(...)` to a device-outer loop.
- [ ] ~270–277: split the `invert_binary` ternary out of the loop — two type-stable loops or `Val`/singleton dispatch matching the file's `BoundDirection` idiom (currently `bin::Union{AffExpr, VariableRef}` forces per-element dynamic dispatch).
- [ ] ~409 + ~525–530: the TimeSeriesParameter path filters devices by `IS.has_time_series` to build `names`, then `_bound_range_with_parameter!` re-checks `has_time_series` per device. Keep the filtered device list and pass it down; drop the recheck (the code itself marks this "PERF: compilation hotspot").

### Task 4.6: `get_parameter_column_refs` dead recompute + TS explode-loop hoisting

**File:** `src/core/parameter_container.jl` (after Tasks 1.1/2.5)

- [ ] ~lines 171–178: line computes `expand_ixs((_get_ts_uuid(...),), param_array)` and **discards it**, then recomputes it on the next line. Delete the dead statement; hoist into a local. Consider returning a `view` instead of the splatted-getindex copy (check callers tolerate a view).
- [ ] ~lines 210–223 (`get_parameter_values(::TimeSeriesAttributes, ...)`): hoist `axes(param_array)[2:end]` and `axes(multiplier_array)[2:end]` out of the per-name loop.

### Task 4.7: Duration-constraint loops

**File:** `src/common_models/duration_constraints.jl` (after Task 2.14)

- [ ] All four builders slice `initial_duration[:, 1]` / `[:, 2]` **inside** the per-timestep loop (2 fresh Vectors per t). Hoist `ics_up = view(initial_duration, :, 1); ics_dn = view(initial_duration, :, 2)` above the t-loop.
- [ ] Hoist `jump_model = get_jump_model(container)` out of all `@constraint` call sites; hoist `get_component_name(ic)`/`get_value(ic)`/`duration_data[ix]` per device where the nesting allows.

### Task 4.8: Rate-of-change loops

**File:** `src/common_models/rateofchange_constraints.jl` (after Task 1.4)

- [ ] `_get_ramp_slack_vars` is called per (device, t) and does two keyed container lookups + returns `Union{NamedTuple{VariableRef}, NamedTuple{Float64}}`. Fetch both slack arrays (or `nothing`) once before the device loop; pass arrays down; index inside.
- [ ] ~104, ~152, ~310: `name ∉ device_name_set` is an O(N) Vector scan per IC → `Set(device_name_set)` for membership (keep the Vector for container axes).
- [ ] ~317: `hasmethod(get_must_run, Tuple{V})` per device — hoist above the loop (V is fixed); preferred: a `supports_must_run(::Type)` trait with POM override.
- [ ] `_get_minutes_per_period` (~2–11): type-unstable `Union{Int, Float64}`, spurious warning at exactly 1 minute, and `Dates.Minute(resolution)` throws `InexactError` for non-whole-minute resolutions > 1 min (e.g. 90 s). Fix: `minutes_per_period = Dates.value(Dates.Millisecond(resolution)) / 60_000.0` (always Float64); warn only when `resolution < Dates.Minute(1)`; delete the stale "NOTE: not included currently." comment.
- [ ] ~156: simplify the assert to `@assert !parameters || ic_power isa JuMP.VariableRef`.

### Task 4.9: Abstract / UnionAll struct fields on hot paths

**Files:** `src/core/dataset.jl`, `src/core/model_internal.jl`, `src/core/device_model.jl`, `src/core/service_model.jl`, `src/core/optimization_container_metadata.jl`, `src/core/outputs_by_time.jl` (after 2.3)

- [ ] `dataset.jl` ~23–33: `InMemoryDataset{N}.values::DenseAxisArray{Float64, N}` is abstract (axes/lookup params free) — every per-step state read/write dispatches dynamically. Parameterize: `mutable struct InMemoryDataset{N, A <: DenseAxisArray{Float64, N}} ... values::A`. Update constructors; run tests (this is the highest-value field fix — it's the emulation state store).
- [ ] `model_internal.jl` ~12: `time_series_cache::Dict{TimeSeriesCacheKey, <:TimeSeriesCache}` is a UnionAll field — declare `Dict{TimeSeriesCacheKey, TimeSeriesCache}` (what the constructor already builds). ~16: `store_params::Union{Nothing, AbstractModelStoreParams}` → `Union{Nothing, ModelStoreParams}` (only concrete subtype; feeds `get_interval` per solve).
- [ ] `device_model.jl` ~61 / `service_model.jl` ~36: `feedforwards::Vector{<:AbstractAffectFeedforward}` → `Vector{AbstractAffectFeedforward}` (heterogeneous by design), converting in the constructor.
- [ ] `optimization_container_metadata.jl` ~3–5: `container_key_lookup::Dict{String, <:OptimizationContainerKey}` → `Dict{String, OptimizationContainerKey}`; make `add_container_key!` end with bare `return`.
- [ ] `outputs_by_time.jl`: store `resolution::Dates.Millisecond` (everything upstream converts to Millisecond already).
- [ ] `parameter_container.jl` (after 1.1/2.5 — same file): `VariableValueAttributes.affected_keys::Set` is `Set{Any}` (constructor passes bare `Set()`) → `Set{OptimizationContainerKey}`. `CostFunctionAttributes.variable_types::Tuple{Vararg{Type}}` is non-concrete — leave unless profiling shows it (build-time struct; LOW).
- [ ] Defer (note in PR, do not implement): parameterizing `DecisionModel`/`EmulationModel`/`ModelInternal{T}` model structs — invasive, POM-facing.

### Task 4.10: Emulation per-step write path

**File:** `src/operation/emulation_model_store.jl` (after Task 2.10)

- [ ] `write_output!` (~109–120, ~132–139): hoist `dataset = container[key]` once (currently 3 hash lookups); avoid the `array[:, 1]` slice copy (use the 2-D single-column `set_value!` from Task 1.7 or a view).
- [ ] `initialize_storage!` (~86–95): add the missing `!should_write_resulting_value(key) && continue` filter (decision store has it; without it, parameters/expressions allocate `n_cols × num_executions` permanently-NaN arrays and reads return silent NaN instead of failing fast).

### Task 4.11: λ-PWL and cost-helper loop hoisting

**Files:** `src/objective_function/piecewise_linear.jl`, `src/objective_function/value_curve_cost.jl`, `src/objective_function/start_up_shut_down.jl`, `src/objective_function/proportional.jl`, `src/objective_function/cost_term_helpers.jl` — parallel-safe

- [ ] `piecewise_linear.jl` ~128–146: the per-t call to the cost-function-typed `get_pwl_cost_expression_lambda` re-runs the entire unit-system normalization (fresh points vector + `PiecewiseLinearData` per t). `add_pwl_term_lambda!` already computed the normalized `data` — hoist `dt`/`multiplier` and call the data-typed overload directly: `get_pwl_cost_expression_lambda(container, T, name, t, data, dt * multiplier)`.
- [ ] `piecewise_linear.jl` ~219–225: hoist `get_parameter_array(container, FuelCostParameter, V)` / multiplier array / `get_fuel_cost(component)` out of the t-loop. ~109: replace `all(iszero.((point -> point.y).(IS.get_points(data))))` with `all(p -> iszero(p.y), IS.get_points(data))`. ~139–140: hoist the `PiecewiseLinearCostVariable` container fetch out of the t-loop.
- [ ] `value_curve_cost.jl` ~324–347 (`_fill_pwl_data_from_arrays!` + `_get_raw_pwl_data`): compute `x_ratio`/`y_ratio` once per device before the time loop and apply in place (`breakpoints .*= x_ratio; slopes .*= y_ratio`) instead of allocating converted vectors + `copyto!` per (device, t); hoist the four parameter-container fetches out of the per-call path.
- [ ] `start_up_shut_down.jl` (~56, 65, 118, 124–128), `proportional.jl` (~36), `cost_term_helpers.jl` (~224): hoist `get_variable(container, U, T)` (and t-invariant `start_up_cost(...)` + its `iszero` test) out of the time loops.
- [ ] `objective_function_pwl_lambda.jl` ~240–248, ~330–338, ~372–380: three copy-pasted blocks construct `ConstraintKey` directly (forbidden by `.claude/claude.md`) with abstract-eltype `Union{Nothing, JuMP.ConstraintRef}` dicts. Replace with `add_constraints_container!(...; sparse = true)`/`lazy_container_addition!` as `add_pwl_constraint_delta!` does.

### Task 4.12: Approximation loop hoists

**Files:** `src/quadratic_approximations/pwmcc_cuts.jl` (after 1.3), `sawtooth.jl` (after 2.15)

- [ ] `pwmcc_cuts.jl` ~150–163: 7 breakpoint vectors depend only on `bounds[idx]` but are rebuilt per (name, **t**). Split into name-outer / t-inner loops.
- [ ] `sawtooth.jl` ~217–221: `saw_coeffs` likewise — hoist per name or inline the scalar `delta^2 * 2.0^(-2j)`.

### Task 4.13: Misc per-step / utility costs

**Files:** various — parallel-safe

- [ ] `optimization_container.jl` `is_milp` (~204–216): `JuMP.all_constraints(model, VariableRef, ZeroOne)` materializes every binary-constraint index per solve (also per `add_dual_container!`). Use `JuMP.num_constraints(model, JuMP.VariableRef, MOI.ZeroOne) > 0`.
- [ ] `optimization_model_interface.jl` ~47–50: `get_optimizer_stats(model)` deepcopies a 21-field mutable struct per step — write a field-by-field `Base.copy(::OptimizerStats)` and use it.
- [ ] `time_series_interface.jl` ~28–29, ~76–77: `haskey` + `getindex` double lookup per parameter per step → single `get(cache, key, nothing)`.
- [ ] `jump_utils.jl` `container_spec(Float64, ...)` (~458–462): `cont.data .= fill(NaN, size(...))` allocates a full temp → `fill!(cont.data, NaN)`.
- [ ] `jump_utils.jl` `sparse_container_spec(T <: AbstractJuMPScalar, ...)` (~467–474): `Dict(indexes .=> zero(T))` aliases ONE mutable zero across all entries (mutation-corruption trap) → `Dict(k => zero(T) for k in indexes)`.
- [ ] `jump_utils.jl` `remove_undef!(::AbstractArray)` (~500–510): body requires `.data`; restrict signature to `::DenseAxisArray` and add a plain-`Array` method using `eachindex`/`isassigned` (function is exported).
- [ ] `jump_utils.jl` `jump_fixed_value(::AffExpr)` (~46–48): drop the temporary array — `sum(...; init = 0.0)` over the generator.
- [ ] `settings.jl` `log_values` (~97–103): `type <: Base.RefValue` runtime branch → `_unwrap(x::Base.RefValue) = x[]; _unwrap(x) = x`.
- [ ] `optimization_problem_outputs_export.jl` `_check_fields` (~107–113): `typeof(fields) <: Set` → dispatch (`_check_fields(::Set)` / generic).
- [ ] `external_evaluation.jl` (~36, 53–59): `evaluators::Dict{DataType, Any}` → `Dict{DataType, AbstractEvaluator}`; make `add_evaluator!`/`add_evaluation_data!` return `nothing`.
- [ ] `common_models/get_time_series.jl` ~67–69 and `common_models/set_expression.jl` ~11–12: collapse `haskey` + `getindex` double lookups into single `get`-style lookups.
- [ ] `core/optimization_container.jl` `check_parameter_multiplier_values` (~360–366): `!all(isnan.(x))` allocates → `!all(isnan, x)` / `!all(isnan, values(x.data))`.
- [ ] `core/optimization_container.jl` `write_initial_conditions_data!` (~1139–1147): use the loop's `field_container`/pair value instead of re-indexing `ic_container_dict[key]`.
- [ ] `utils/indexing.jl` `fix_expand` (~57–58): non-scalar `getindex` on the DenseAxisArray copies the slice, and the broadcast materializes a discarded `Array{Nothing}`; it also mutates JuMP state without a `!` name. Use `@views`/an explicit loop and rename to `fix_expand!` (no in-repo callers — grep POM for callers and note the rename in the PR).

### Task 4.14: `OptimizerStats` deserialization typing

**File:** `src/core/optimizer_stats.jl` (~91–110)

- [ ] `Vector(undef, n)` (= `Vector{Any}`) + 21-arg splat into the constructor → construct field-by-field or use a typed `Vector{Union{Float64, Missing}}`. Also `_summary_to_dict!` (`jump_utils.jl` ~566–586): collapse the redundant `ismissing` if/else into one `setfield!`, hoist the `fields` vector to a `const` tuple.

---

## Phase 5: Cleanup, dead code, conventions, docs

### Task 5.1: Delete verified-dead/broken code

- [ ] `jump_utils.jl` `_calc_dimensions` (both methods, ~514–557): zero callers, wrong debug messages, `Dict{String,Any}` returns.
- [ ] `file_utils.jl` `find_variable_length` (~29–31): zero callers.
- [ ] `jump_utils.jl` `write_data(::Float64, ...)` (~16–19): double-encodes JSON, zero callers.
- [ ] `model_internal.jl` (~61–62 vs ~83–84): duplicate `set_store_params!` — keep the typed one.
- [ ] `network_model.jl` ~245–265: unreachable final `else` branch in the validation chain; also the `\\` inside the string literal at ~253 puts a literal backslash + newline into the user-facing error — use a single-line string or proper continuation. Same `\\`-in-string issue in `optimization_container.jl` `add_dual_container!` warn (~755–757) and `init_optimization_container!` (~335–336).
- [ ] `range_constraint.jl` ~10–16 + `constraint_helpers.jl` ~97–107: docstring promises an epsilon/equality degenerate case that is not implemented and `add_range_equality_constraint!` has zero callers — fix the docstring; delete or wire up the orphan helper (ask maintainer if unsure: prefer deleting).
- [ ] `src/InfrastructureOptimizationModels.jl:142`: `const POM = InfrastructureOptimizationModels` — `POM` means PowerOperationsModels everywhere else in Sienna; the alias is unused in `src/` (verified). Delete it (grep `test/` first; update any use to `IOM`-style alias if needed).

### Task 5.2: `isa`/`<:` → dispatch conversions (convention)

- [ ] `objective_function/linear_curve.jl` ~61: `if fuel_cost isa Float64` → split `_add_fuel_linear_variable_cost!(..., ::Float64)` / `(..., ::IS.TimeSeriesKey)` mirroring `quadratic_curve.jl`.
- [ ] `quadratic_approximations/incremental.jl` ~168–172, ~203: `R <: DCVoltage` branching → `_incr_x_var` / `_incr_x_name` dispatch helpers.
- [ ] `operation/decision_model.jl` ~53, `operation/emulation_model.jl` ~34: `name isa String` ladders → `_model_name(::Nothing, ::Type{M})` / `(::Symbol, _)` / `(::String, _)` helpers.
- [ ] `model_numerical_analysis_utils.jl` ~113/~138, `optimization_debugging.jl` ~66/~95: `isa(x, SparseAxisArray)` → two-method dispatch (`_scan(arr::SparseAxisArray)` / `(::DenseAxisArray)`). The TODO at ~112 already asks for this.

### Task 5.3: Docstring/math-comment corrections (code is right, docs are wrong)

- [ ] `duration_constraints.jl`: LaTeX multipliers swapped (up-time doc says `d_min^down · x^stop`, code correctly uses `duration_data[ix].up * varstop`; mirror for down-time) at ~114/126 and ~213/225; argument lists document nonexistent `cons_name::Symbol`, `var_keys`, `initial_duration_on/off::Vector` params — align with actual signatures.
- [ ] `rateofchange_constraints.jl` ~56–72 and ~263–271: doc equations don't match the implemented `bound_decrease=false` form, omit the `minutes_per_period` factor, and claim `rate_data[2][ix].max*varstart` where code uses `power_limits.min` — rewrite to match code.
- [ ] `constraint_helpers.jl` ~138 vs ~146: contradictory statements about the `bound_decrease` default (actual default `true`); fix line ~146.
- [ ] `add_variable.jl` ~16: `lb \ge x \le ub` → `lb \le x \le ub`.
- [ ] `objective_function_pwl_delta.jl` ~16–17/36–37: comment claims width bounds "naturally enforce ordering… non-convex curves remain LP-feasible" — false (only the convex case is exact in an LP; correctness rests on `curvity_check`). Rewrite the comment; also fix the `add_pwl_variables_delta!` docstring ("n_points (= number of delta variables)" — callers pass n_segments).
- [ ] `solver_sos2.jl` ~86 / `manual_sos2.jl` ~82: "`pwmcc_segments` … default 4" → default 0.
- [ ] `nmdt_common.jl` ~279: `_residual_product!` docstring signature omits the `depth` positional arg.
- [ ] `device_model.jl` ~30–41 / `service_model.jl` ~26–29: kwarg names (`feedforward` → `feedforwards`; nonexistent `use_service_name`) and keywords-shown-as-positional.
- [ ] `optimization_problem_outputs.jl` ~111–115: `load_system` docstring says "return it"; function returns `nothing`.
- [ ] `optimization_container_keys.jl` ~52–56: comment says AuxVarKey is exempt from the abstract-component check but the exemption is implemented for `ConstraintType` — resolve which is intended (likely fix the comment; confirm no caller constructs abstract-component AuxVarKeys).

### Task 5.4: `@debug` with side effect

**File:** `src/core/optimization_container.jl` (~277–279)

`@debug begin JuMP.set_string_names_on_creation(JuMPmodel, true) end` silently overrides the
user's `store_variable_names` setting whenever debug logging is enabled.

- [ ] Replace with explicit, logged behavior:

```julia
if Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
    @debug "Debug logging active: forcing string names on JuMP model"
    JuMP.set_string_names_on_creation(JuMPmodel, true)
end
```

  or simply delete the override (preferred — ask maintainer; default to delete + note).

### Task 5.5: Misc small correctness hygiene

- [ ] `interfaces.jl` `get_min_max_limits` fallback (~222–226) returns `nothing` which every caller immediately dereferences (`limits.min`) → make the stub `error("get_min_max_limits not implemented for $(...)")` like `get_variable_binary` does.
- [ ] `print_pt_v3.jl` ~145: `first(encode_key_as_string.(get_affected_values(v)))` throws on empty and drops extra values → `join(encode_key_as_string.(get_affected_values(v)), ", ")` with empty-safe handling.
- [ ] `dataframes_utils.jl` ~6–8: loosen `to_matrix(df_row::DataFrameRow{DataFrame, DataFrames.Index})` to `::DataFrameRow`.
- [ ] `optimization_model_interface.jl` ~161–163: infeasible-model dump uses a fixed filename — successive allowed failures overwrite; embed the (sanitized) timestamp like the export path does.
- [ ] `decision_model_store.jl` `initialize_storage!` (~45–56): pre-filled per-execution NaN arrays are discarded by every `write_output!` (rebinds the slot) — prefill with empty dicts; keep column names available for `get_column_names` via store params or a side table.
- [ ] `core/optimization_container.jl` `calculate_dual_variables!` (~1317–1329): dead `status =` assignment, bare `return` discards it — return the status or drop the variable.
- [ ] `core/optimization_container.jl` ~119: constructor builds `OrderedDict{InitialConditionKey, Vector{InitialCondition}}()` for a field declared `OrderedDict{InitialConditionKey, Vector{<:InitialCondition}}` — make them match (use the field's exact type in the constructor).

### Task 5.6: Update `.claude/claude.md`

- [ ] The repo-structure tree is stale: `objective_function/import_export.jl` is listed but does not exist; `objective_function_pwl_lambda.jl` / `objective_function_pwl_delta.jl` exist but are not listed; `core/` is missing `optimization_container_utils.jl`, `external_evaluation.jl`, `dual_processing.jl`; `initial_conditions/` now contains only `calculate_initial_condition.jl`. Refresh the tree (the file's own maintenance note requires this).

### Task 5.7: `Settings.ext` / model `ext` Dict{String, Any} fields

- [ ] Leave as-is (extension escape hatches by design) — explicitly out of scope; do not "fix".

### Task 5.8: Formatter sweep

- [ ] `julia scripts/formatter/formatter_code.jl` — commit any residue separately as `style: formatter`.

---

## Phase 6: Tests and verification

### Task 6.1: Regression tests for every Phase 1–2 fix

Each Phase 1/2 task lists its test; this task is the checkpoint that none were skipped.
- [ ] Confirm a test exists (new or pre-existing) for: 1.1 multiplier, 1.2 NMDT domains, 1.3 PWMCC validation, 1.4 t=1 ramp, 1.5 sparse export labels, 1.7 dataset write guard, 1.8 integer-var duals, 2.1 emulation store keys, 2.2 conversion matrix (1d/2d/3d × timestamps/nothing), 2.3 OutputsByTime, 2.4/2.5 constructors, 2.6 dual validation error, 2.10 empty!/isempty, 2.11 meta validation, 2.14 duration IC asymmetry, 2.15 container completeness, 2.17 idempotent objective, 2.18 timestamp index.

### Task 6.2: Re-enable the emulation test suite

- [ ] `test/test_model_emulation.jl` is disabled ("FIXME not working"). After Tasks 1.7, 1.9, 2.1, 2.10, 2.12, 2.13, 4.10, re-include it in the test discovery, fix residual failures (they are the bugs this plan fixes — investigate any remainder rather than re-disabling), and delete the FIXME.

### Task 6.3: Non-[0,1]-domain coverage for approximations

- [ ] Parametrize existing quadratic/bilinear approximation tests over domains `[-1, 1]`, `[0, 2]`, `[5, 10]` for every config exposing `epigraph_depth`/`pwmcc_segments` (NMDT, DNMDT, SolverSOS2, ManualSOS2, sawtooth, epigraph). The review found two HIGH math bugs that hid behind exclusively-[0,1] tests.

### Task 6.4: Full suite + Aqua

- [ ] `julia --project=test test/runtests.jl` green, including Aqua (watch for new ambiguities from added dispatch methods — particularly Tasks 2.6, 5.2).

### Task 6.5: Final report

- [ ] PR description must list: the four **[DECISION]** items (3.1–3.4) with their behavior changes; the exported-API semantic changes (1.1 `get_parameter_values`, 2.8 rename, 3.3); the deferred items (delta-PWL width-constraint staleness from 3.4; model-struct parameterization from 4.9; lb/ub asymmetry note from 2.16).

---

## Appendix: verified-finding index (for traceability)

| # | File | Issue | Sev | Phase |
|---|------|-------|-----|-------|
| 1 | parameter_container.jl | multiplier applied twice (generic attrs) | HIGH | 1.1 |
| 2 | nmdt_common.jl | epigraph tightening frame mix (default-on) | HIGH | 1.2 |
| 3 | solver_sos2.jl, manual_sos2.jl | PWMCC alignment check too weak | HIGH | 1.3 |
| 4 | rateofchange_constraints.jl | t=1 big-M omits initial status | HIGH | 1.4 |
| 5 | jump_utils.jl | sparse to_dataframe label misalignment; _to_matrix row/key handling | HIGH | 1.5 |
| 6 | model_numerical_analysis_utils.jl | signed/abs min-max corruption | MED | 1.6 |
| 7 | dataset.jl | 2-D set_value! slices input by store index | MED | 1.7 |
| 8 | dual_processing.jl | UndefVar `v`; integers → binary on restore; redundant sparse double-write | HIGH | 1.8 |
| 9 | problem_outputs.jl / optimization_problem_outputs.jl | emulation timestamps collapse | MED | 1.9 |
| 10 | abstract_model_store.jl | generated accessors bypass get_data_field | HIGH | 2.1 |
| 11 | jump_utils.jl | to_outputs_dataframe: trailing `()`, enum-not-Val shadow method, 1-D method, 3-D method gap, broken keyless sparse to_dataframe, axis-value timestamp indexing | HIGH | 2.2 |
| 12 | outputs_by_time.jl | dead validation/self-recursion; self-referential kwarg; Matrix→DataFrame ctor | HIGH | 2.3 |
| 13 | initial_conditions.jl | key-based ctor TypeError | HIGH | 2.4 |
| 14 | parameter_container.jl + add_param_container.jl | EventParametersAttributes unconstructible | HIGH | 2.5 |
| 15 | add_constraint_dual.jl | _validate_keys out-of-scope interpolation | MED | 2.6 |
| 16 | optimization_problem_outputs.jl | set_source_data!/_read_outputs/JSON export/read_expressions kwargs | MED | 2.7 |
| 17 | optimization_model_interface.jl | mutating 2-arg get_simulation_info | HIGH | 2.8 |
| 18 | jump_utils.jl | sparse check_conflict_status never dispatches; iterates values | HIGH | 2.9 |
| 19 | dataset_container.jl, emulation_model_store.jl | empty!/isempty broken/dead branches | MED | 2.10 |
| 20 | optimization_container_keys.jl | check_meta_chars shadowed; undefined decode_symbol | MED | 2.11 |
| 21 | store_common.jl | range(::Int; step::Millisecond) export crash | MED | 2.12 |
| 22 | emulation_model.jl, problem_template.jl | template aliasing; non-exception-safe deepcopy dance | MED | 2.13 |
| 23 | duration_constraints.jl | up-column axis vs down-column writes | MED | 2.14 |
| 24 | manual_sos2/sawtooth/epigraph/incremental | container off-by-one/overwrites; unbounded δ; anonymous ordering constraints | MED | 2.15 |
| 25 | add_variable.jl | service warm start ignores settings | MED | 2.16 |
| 26 | optimization_container.jl | non-idempotent get_objective_expression | MED | 2.17 |
| 27 | datetime_utils.jl | find_timestamp_index bounds/floor | MED | 2.18 |
| 28 | quadratic_curve.jl | fuel multiplier vs comment/linear path | MED | 3.1 |
| 29 | proportional.jl | no dt scaling | MED | 3.2 |
| 30 | value_curve_cost.jl | is_nontrivial_offer TS polarity | MED | 3.3 |
| 31 | offer_curve_types.jl | breakpoint param storage class | MED | 3.4 |
| 32 | optimization_container.jl | O(N²) QuadExpr `+=` | HIGH | 4.1 |
| 33 | definitions.jl | non-const enum aliases; untyped ENUM_MAPPINGS | MED | 4.2 |
| 34 | jump_utils.jl + store_common.jl | per-element MOI queries in jump_value | MED | 4.3 |
| 35–47 | (see Phase 4 tasks) | loop hoisting, abstract fields, double lookups, allocations | MED/LOW | 4.4–4.14 |
| 48+ | (see Phase 5 tasks) | dead code, isa→dispatch, docs, claude.md | LOW | 5.x |
