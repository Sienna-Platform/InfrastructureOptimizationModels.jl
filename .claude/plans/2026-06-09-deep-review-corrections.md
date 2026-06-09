# Deep Review Corrections — Implementation Plan

> **For the implementing agent:** This plan was produced by an in-depth correctness/performance
> review of the entire `src/` tree (reviewed at commit `37dbcf6`, branch `claude/dazzling-mayer-0x7p70`).
> Every finding below was verified against the source at the cited lines. Line numbers may drift
> as tasks land — **always re-read the cited code and match on the quoted snippet, not the line
> number, before editing.** Work task-by-task, one commit per task, checkbox (`- [ ]`) tracking.

**Goal:** Fix verified correctness bugs (broken APIs, wrong math, wrong outputs) and remove
performance defects (O(N²) accumulation, hot-loop allocations, abstract struct fields, non-const
globals) found in the Fable deep review of 2026-06-09.

**Read first:** `.claude/Sienna.md` (conventions: no `isa`/`<:` branching — use dispatch; concrete
struct fields; no untyped containers; `add_*!` returns `nothing`; run the formatter after every
task) and `.claude/claude.md` (architecture, container-key rules, MarketBidCost data flow).

**Testing protocol (every task):**
1. `julia --project=test test/runtests.jl` (first run: `julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`).
   If the environment has no package-registry access (cloud sandboxes may 403), commit, push, and
   verify via the GitHub Actions CI run instead — do not skip verification.
2. Add the regression test specified in the task **before or with** the fix; it must fail on the
   old code where feasible.
3. `julia scripts/formatter/formatter_code.jl` before each commit.

**Commit style:** `fix:`/`perf:`/`docs:` prefix, one task per commit, mention file and behavior.

---

## Global caveats

- **POM coordination (behavior-visible fixes).** IOM is consumed by PowerOperationsModels (POM).
  Tasks marked **[POM-CHECK]** change behavior POM may rely on. Implement the recommended fix,
  but list each in the PR description so POM maintainers can confirm: T13 (parameter-value
  semantics), T14 (t=1 ramp big-M), T25 (objective semantics block), T8 (setter rename).
- **Severity legend:** HIGH = crashes, wrong optimization results, or wrong written outputs;
  MEDIUM = conditional bug or measurable hot-path cost; LOW = hygiene/dead code/docs.
- The emulation-model test file `test/test_model_emulation.jl` is currently disabled
  ("FIXME not working"). Task T21f re-enables it; until then, emulation fixes get targeted unit
  tests against the store/dataset APIs directly.

## Parallelism map

Tasks within a phase touching different files can be dispatched to parallel agents. Same-file
tasks are merged into one task below precisely to avoid conflicts. Suggested waves:

```
Wave 1 (independent files): T1, T2, T3, T4, T5+T6 (one agent, same file), T7, T8, T9, T11, T12
Wave 2 (independent files): T13, T14, T15, T16, T17, T18+T19 (verify file overlap), T20, T22,
                            T23, T24, T26, T27, T28
Wave 3 (sequential where same-file): T10, T21 (touches 4 files — single agent), T25
Wave 4 (perf): T29, T30, T31, T32, T33a-e, T34a-c, T35, T36, T37, T38, T39
Wave 5: T40, T41, T42, then full verification
```

---

# Phase 1 — Correctness: deterministic crashes / broken exported APIs

### Task 1: `InitialCondition` key-based constructor instantiates an impossible type

- **File:** `src/core/initial_conditions.jl:20-30` — HIGH
- **Problem:** The struct is `InitialCondition{T <: InitialConditionType, U <: Union{JuMP.VariableRef, Float64, Nothing}}`,
  but the key-based convenience constructor returns `InitialCondition{T, U}(component, value)` where
  `U` is the **component** type from `InitialConditionKey{T, U}` — violates the struct's own bound,
  so every call throws `TypeError`. Exported API (`InitialCondition` is exported).
- **Fix:** `return InitialCondition{T, V}(component, value)`.
- **Test:** in `test/test_initial_conditions.jl` (or nearest IC test file): construct via
  `InitialCondition(InitialConditionKey(SomeICType, MockThermalGen), mock_component, 1.0)` and
  assert type/value. Use existing mocks.

### Task 2: `EventParametersAttributes` is unconstructible as used

- **Files:** `src/core/parameter_container.jl:83-94`, `src/common_models/add_param_container.jl:85-103` — HIGH
- **Problem:** `struct EventParametersAttributes{T <: IS.InfrastructureSystemsComponent, U <: ParameterType} <: ParameterAttributes`
  has neither type parameter in any field (`affected_devices::Vector{<:IS.InfrastructureSystemsComponent}` —
  also a UnionAll field), so Julia generates no 1-arg constructor; the only call site
  (`add_param_container.jl:99`) calls `EventParametersAttributes(V)` with a component **type** →
  `MethodError` on every `add_param_container!` for `T <: EventParameter` (exported; consumed by
  `range_constraint.jl:479-503`). `get_param_type` reads `U`, which is never bound.
- **Fix (recommended):** redesign minimally:
  ```julia
  struct EventParametersAttributes{T <: IS.InfrastructureSystemsComponent, U <: ParameterType} <: ParameterAttributes
      affected_devices::Vector{T}
  end
  EventParametersAttributes(::Type{T}, ::Type{U}) where {T <: IS.InfrastructureSystemsComponent, U <: ParameterType} =
      EventParametersAttributes{T, U}(T[])
  ```
  and change the call site to `EventParametersAttributes(V, T)` (the `T <: EventParameter` of the
  wrapper). Keep `get_param_type` working.
- **Test:** call `add_param_container!` with a mock `EventParameter` subtype + mock component, then
  `get_parameter`/`get_param_type` round-trip. Note `get_parameter_values(::EventParametersAttributes, ...)`
  (parameter_container.jl:202-208) must still dispatch.

### Task 3: `OutputsByTime` — dead validation, self-referential kwarg default, broken Matrix dataframe

- **File:** `src/core/outputs_by_time.jl` — HIGH ×3
- **Problems:**
  1. Lines 8-16: the outer "validating" constructor (untyped `column_names`) is **shadowed** by the
     implicit constructor whenever `column_names isa NTuple{N, Vector{String}}` (every normal call),
     so `_check_column_consistency` never runs (all 4 methods at lines 18-68 are dead); any
     non-NTuple `column_names` makes the outer method call itself → `StackOverflowError`.
  2. Line 149: `make_dataframes(outputs; table_format::TableFormat = table_format)` — the default
     references itself, resolving to a nonexistent global → `UndefVarError` whenever the kwarg is
     omitted.
  3. Line 132: `DataFrames.DataFrame(array, outputs.column_names)` passes an `NTuple` where the
     DataFrames constructor needs an `AbstractVector` of names → `MethodError` for every
     `make_dataframe(::OutputsByTime{Matrix{Float64}}, ...)` call.
- **Fixes:**
  1. Convert to an inner constructor (mirrors PSI's `ResultsByTime`):
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
  2. `table_format::TableFormat = TableFormat.LONG`.
  3. `DataFrames.DataFrame(array, outputs.column_names[1])` (Matrix-backed outputs are `N == 1`).
- **Test:** construct `OutputsByTime` with mismatched column names → expect `error`; call
  `make_dataframes(outputs)` without the kwarg; call `make_dataframe` on a `Matrix{Float64}`-backed
  instance. See `test/test_outputs_by_time.jl` if present, else add one.

### Task 4: Generated store accessors bypass `get_data_field` — every `list_*_keys` / outputs API broken for `EmulationModel`

- **File:** `src/core/abstract_model_store.jl:75-99` — HIGH
- **Problem:** `list_fields`, `list_keys`, `get_value` are `@generated` over `getfield(store, $field)`,
  but `EmulationModelStore` keeps containers inside `data_container` and overrides `get_data_field`
  (`src/operation/emulation_model_store.jl:11-17`). Raw `getfield(store, :variables)` →
  `ErrorException: type EmulationModelStore has no field variables`. Breaks
  `list_variable_keys`/`list_all_keys` (`optimization_model_interface.jl:271,339-354`) and
  `OptimizationProblemOutputs(::EmulationModel)` (`problem_outputs.jl:61-66`). Hidden because the
  emulation test file is disabled.
- **Fix:** route through the overridable accessor in all three generated bodies:
  `:(return collect(keys(get_data_field(store, Val($field)))))` (and the `get_value` analog
  `:(return get_data_field(store, Val($field))[$K(T, U)])`). `DecisionModelStore` behavior is
  unchanged (generic `get_data_field` is `getfield`).
- **Test:** unit test on an `EmulationModelStore` populated via `initialize_storage!` with the mock
  container: `list_keys(store, VariableType)` returns the expected keys; same for `get_value`.

### Task 5: `to_outputs_dataframe` / sparse `to_dataframe` conversion layer — four hard failures and a silent mislabeling

- **File:** `src/utils/jump_utils.jl` — HIGH ×4, MEDIUM ×3 (single task: same file)
- **Problems (all verified):**
  1. Lines 252-254: generic entry `to_outputs_dataframe(array, timestamps)` ends in
     `Val(TableFormat.LONG))()` — the trailing `()` calls the returned `DataFrame` → always throws.
     **Fix:** delete the trailing `()`.
  2. Lines 431-441: the 3d `(Vector{String}, Vector{String}, UnitRange)` + `::Nothing` method
     shadows the working parametric method at 395-429 and forwards the **raw enum**
     (`TableFormat.LONG`) instead of `Val(...)` → `MethodError` on every
     `read_variable`/`read_dual`/`read_expression` of a 3d output
     (`_read_outputs`, `operation/optimization_model_interface.jl:285-288`). **Fix:** delete the
     method (395 covers it). Do **not** "fix" it with `Val(...)` — that recurses into itself.
  3. Lines 256-266: 1d LONG variant hardcodes `:DateTime => [1]`, ignores `timestamps`, and the
     1-element vector does not broadcast → constructor error for >1 name; column name/type also
     break the `:time_index` joins in `optimization_problem_outputs.jl:413,421,443`.
     **Fix:** mirror the 2d variants — with `timestamps === nothing` emit
     `:time_index => fill(1, length(axes(array, 1)))`; with timestamps emit
     `:DateTime => fill(only(timestamps), length(axes(array, 1)))`.
  4. Lines 233-235 + 95-101 + 175-183: `to_dataframe(::SparseAxisArray, key)` orders data columns by
     `sort!`ed **raw key tuples** (`to_matrix`, line 99) but labels by `sort!`ed **encoded strings**
     (line 180) — orders diverge for mixed-type tuples or names containing the delimiter →
     **silently mislabeled CSV exports** (`store_common.jl:17`). The repo documents this exact
     hazard at `decision_model_store.jl:128-146`. **Fix:** derive the tuple ordering once and encode
     it: `tuple_columns = sort!(unique!([k[1:(N-1)] for k in keys(array.data)]));
     DataFrame(_to_matrix(array, tuple_columns), Symbol.(encode_tuple_to_column.(tuple_columns)))`.
  5. (MEDIUM) Lines 87-91: `_to_matrix(::SparseAxisArray)` uses the raw time value as the row index
     (`data[t, ix]`) — `BoundsError` for any time key set ≠ `1:T` — and `array.data[(col..., t)]`
     `KeyError`s on genuinely sparse key sets. **Fix:** `row = Dict(t => i for (i, t) in
     enumerate(sort!(collect(time_steps))))`; `data[row[t], ix] = get(array.data, (col..., t), NaN)`.
  6. (MEDIUM) Lines 443-446: the no-key `to_dataframe(::SparseAxisArray)` passes a tuple of encoded
     strings into `_to_matrix` (expects raw tuples) and `DataFrame(::Matrix, ::Tuple)` is not a
     valid constructor — triple-broken, no callers. **Fix:** reimplement using the same
     tuple-ordering approach as item 4, or delete.
  7. (MEDIUM) Lines 287-292 and 375-385: LONG converters index `timestamps_arr[time_index]` by axis
     **values**; any non-`1:T` integer axis misindexes. The 3d loop also does an axis-keyed
     `array[name, name2, time_index]` per element. **Fix:** `for (j, time_index) in
     enumerate(axes(array, 2))` → `timestamps_arr[j]`; in the 3d method read from `array.data`
     directly. Also widen the with-timestamps 3d signature (line 350-357) to the same
     `T <: Union{Vector{String}, UnitRange{Int}}` second-axis parameterization as the `::Nothing`
     variant (line 395-403) — `DecisionModelStore` stores `(String, Int, Int)`-axis 3d arrays and
     `make_dataframe(::OutputsByTime{DenseAxisArray{Float64,3}})` passes real timestamps → method gap.
- **Tests:** extend `test/test_jump_utils.jl`: 1d single+multi name; 3d `(String,String,Int)` and
  `(String,Int,Int)` with `nothing` and with real timestamps; sparse with mixed tuples like
  `("gen", 2)`/`("gen", 10)` asserting value-label alignment; sparse with time window `13:24`.

### Task 6: `check_conflict_status` sparse method can never match and mis-iterates

- **File:** `src/utils/jump_utils.jl:654-687` — HIGH
- **Problem:** Sparse constraint containers are built with eltype `Union{Nothing, ConstraintRef}`
  (`sparse_container_spec`, lines 485-492); the method signature
  `SparseAxisArray{JuMP.ConstraintRef}` never matches (invariance) → `MethodError`, swallowed by the
  `try/catch` in `compute_conflict!` (`optimization_container.jl:488-495`) and misreported as
  "optimizer doesn't support IIS" for any model with sparse constraints. The body also destructures
  `(index, constraint)` while `SparseAxisArray` iterates **values**. Both methods also accumulate
  into untyped `Vector()`.
- **Fix:** signature `constraint_container::SparseAxisArray`; iterate `pairs(constraint_container.data)`;
  skip `nothing` values; drop `isassigned`; type the accumulators (e.g. `Vector{eltype(keys(constraint_container.data))}()`
  for sparse, `Vector{NTuple{N, Any}}` or `Vector{Tuple}` for dense, line 659).
- **Test:** build a sparse constraint container via `add_constraints_container!(...; sparse = true)`,
  assign one constraint, and assert `check_conflict_status` dispatches and returns without error
  (mock `MOI.get` is not needed — assert no `MethodError` by calling with a model where the
  attribute errors and catching the *right* error, or factor the MOI query for testability).

### Task 7: `process_duals` — undefined variable crash and binary/integer corruption

- **File:** `src/core/dual_processing.jl` — HIGH ×2 + MEDIUM
- **Problems:**
  1. Lines 61-63: `cache[key][:fixed_int_value] = jump_value.(v)` — `v` is the iteration variable
    of a *previous* loop and is not in scope → `UndefVarError` for any fixed integer variable.
    **Fix:** `jump_value.(variable)` (or `var_cache[key]`).
  2. Lines 102-121: restore path runs `JuMP.unfix.(variable); JuMP.set_binary.(variable)`
    unconditionally — general-integer variables are re-declared **binary** after dual processing
    (the correct logic sits in the commented block at 107-121). Bounds are also never restored.
    **Fix:** use the cached flags:
    ```julia
    JuMP.unfix.(variable)
    if cache[key][:integer]
        JuMP.set_integer.(variable)
    else
        JuMP.set_binary.(variable)
    end
    haskey(cache[key], :lb) && JuMP.set_lower_bound.(variable, cache[key][:lb])
    haskey(cache[key], :ub) && JuMP.set_upper_bound.(variable, cache[key][:ub])
    haskey(cache[key], :fixed_int_value) && JuMP.fix.(variable, cache[key][:fixed_int_value]; force = true)
    ```
    (delete the commented block once implemented).
  3. (MEDIUM, same file) Lines 23-31: the sparse branch broadcasts `jump_value.(v)` then re-writes
    every element in a loop — delete the `isa` branch; the broadcast alone is correct for both
    container types.
- **Test:** model with one binary and one integer variable; run `process_duals` with a mock/LP
  optimizer; assert post-call `is_integer`/`is_binary` flags and bounds are restored.

### Task 8: Mutating two-arg `get_simulation_info`; `set_simulation_info!` does not exist [POM-CHECK]

- **File:** `src/operation/optimization_model_interface.jl:190` — HIGH
- **Problem:** `get_simulation_info(model::AbstractOptimizationModel, val) = model.simulation_info = val`
  is a setter misnamed as a getter during the PSI port; grep confirms no `set_simulation_info!`
  anywhere. Downstream code calling the PSI name gets `MethodError`; anyone calling this "getter"
  silently overwrites state.
- **Fix:** replace with `function set_simulation_info!(model::AbstractOptimizationModel, val); model.simulation_info = val; return; end`
  and export it if the sibling accessors are exported. Keep a deprecation stub only if POM currently
  calls the 2-arg form (grep POM before deleting).

### Task 9: `_validate_keys` references out-of-scope variables on its own error path

- **File:** `src/common_models/add_constraint_dual.jl:76-81` — MEDIUM
- **Problem:** the error message interpolates `$constraint_type` and `$D`, neither in scope →
  `UndefVarError` instead of the intended `IS.InvalidValue` whenever a dual is requested for an
  un-built constraint (call sites: lines 95, 115).
- **Fix:** `_validate_keys(keys, ::Type{T}, ::Type{D}) where {T, D}` interpolating `T`/`D`; update
  both call sites to pass `constraint_type, D`.
- **Test:** request a dual for a constraint type that was never added; assert `IS.InvalidValue` with
  the formatted message.

### Task 10: `optimization_problem_outputs.jl` — four broken error/IO paths

- **File:** `src/core/optimization_problem_outputs.jl` — MEDIUM ×4 (single task: same file)
- **Problems / fixes:**
  1. Lines 340-346 (`set_source_data!`): message interpolates undefined `$sys_uuid` and
     `$(res.source_uuid)` (field is `source_data_uuid`; local is `source_uuid`) → `UndefVarError`
     on the UUID-mismatch path. **Fix:** `"System mismatch. $source_uuid does not match the stored value of $(res.source_data_uuid)"`.
  2. Lines 443-447 (`_read_outputs`): the "dropped rows" diagnostic interpolates `$(outputs[key])`
     **before** `outputs[key]` is assigned → `KeyError` masks the diagnostic. **Fix:** interpolate `$(df)`.
  3. Lines 970-976 (`read_expressions`): drops `kwargs...` that every sibling
     (`read_variables`/`read_duals`/`read_parameters`/`read_aux_variables`) forwards. **Fix:**
     `read_expression(res, x; kwargs...)`.
  4. Lines 1063-1070 (`export_optimizer_stats`): `JSON.write`/`JSON.json` — the package imports only
     `JSON3` → `UndefVarError` for the documented `format = "json"`. **Fix:**
     `open(joinpath(directory, "optimizer_stats.json"), "w") do io; JSON3.write(io, to_dict(data)); end`.
- **Tests:** unit tests per path (the JSON one round-trips through `JSON3.read`).

### Task 11: Container-key validation shadowed; `Base.convert` methods call a nonexistent function

- **File:** `src/core/optimization_container_keys.jl` — MEDIUM
- **Problems:**
  1. Lines 81-100: the `meta::String` constructor method (92-100) is more specific than the
     validating `meta::Any` method (81-90) and omits `check_meta_chars` — since meta is always a
     `String` in practice, the `__`-in-meta guard never runs, silently corrupting the
     encode/decode round-trip for stored outputs. `make_key` (102-113) also skips it.
     **Fix:** delete the redundant 92-100 method (the default-arg method covers it), or add
     `check_meta_chars(meta)` to it and to `make_key`.
  2. Lines 161-162: `Base.convert(::Type{ExpressionKey}, name::Symbol) = ExpressionKey(decode_symbol(name)...)`
     (and the `ConstraintKey` twin) — `decode_symbol` is defined nowhere in the package →
     `UndefVarError` if ever invoked. **Fix:** delete both methods (grep first for implicit
     conversion reliance), or implement `decode_symbol` against `encode_symbol`'s format.
- **Test:** `@test_throws IS.InvalidValue VariableKey(SomeVar, MockThermalGen, "bad__meta")`.

### Task 12: `Base.empty!(::DatasetContainer)` calls a method that doesn't exist

- **Files:** `src/core/dataset_container.jl:19-28`, `src/core/dataset.jl` — MEDIUM
- **Problem:** `empty!(val)` is called on each dataset value (`InMemoryDataset`/`HDF5Dataset`), but
  no `Base.empty!` method exists for any dataset type → `MethodError` on first non-empty call.
- **Fix (pick per intent):** `empty!(field_dict)` (matches `Base.empty!(::EmulationModelStore)`
  semantics), or define `Base.empty!` for the dataset types. Recommended: empty the dicts.
- **Test:** populate a `DatasetContainer{InMemoryDataset}` and call `empty!`.

---

# Phase 2 — Correctness: wrong results (model math, parameters, outputs)

### Task 13: Parameter values double-multiplied on every per-step store write [POM-CHECK]

- **File:** `src/core/parameter_container.jl:140-147` + `194-200` — HIGH (hottest correctness bug)
- **Problem:** for generic attributes (`NoAttributes`, `VariableValueAttributes`,
  `CostFunctionAttributes`), `get_parameter_values` already multiplies
  (`(.*).(jump_value.(param_array), multiplier_array)`, line 199) and `calculate_parameter_values`
  multiplies **again** (line 145-146) → written values are `param .* mult.^2`. With the common
  `-1.0` multiplier this silently flips signs. Hit every simulation step via
  `write_model_parameter_outputs!` (`store_common.jl:114`), by `lookup_value`
  (`optimization_container.jl:1469-1473`), and by `read_parameters`
  (`optimization_container.jl:992-1005`). The `TimeSeriesAttributes` / `EventParametersAttributes`
  overloads and the SparseAxisArray `calculate_parameter_values` (149-157) all single-multiply —
  the generic dense path is the outlier.
- **Fix:** make the generic `get_parameter_values(::ParameterAttributes, ::DenseAxisArray, ::DenseAxisArray)`
  return **raw** values: `return jump_value.(param_array)`. Semantics after fix (document in
  docstrings): `get_parameter_values` = raw parameter values; `calculate_parameter_values` =
  values × multiplier. Verify `read_parameters`' `_calculate_parameter_values` stays single-multiplied
  (it does: multiplies for `ParameterType`, returns raw for `ObjectiveFunctionParameter`).
- **[POM-CHECK]:** grep POM for direct `get_parameter_values(::ParameterContainer)` calls that may
  assume multiplied values.
- **Test:** `ParameterContainer` with `NoAttributes`, params = 2.0, multipliers = -1.0:
  `calculate_parameter_values` must return -2.0 (old code returns +2.0).

### Task 14: t=1 ramp big-M omits initial status — startups in period 1 can be infeasible [POM-CHECK]

- **File:** `src/common_models/rateofchange_constraints.jl:228-237` (OnStatusParameter path) — HIGH
- **Problem:** at t ≥ 2 the relaxation is `big_m = power_limits.max * (2 - yprev - ycur)` (line 244)
  — relaxed for both start and stop transitions. At t = 1 it is
  `big_m = power_limits.max * (1 - ycur)` (line 232): a unit the upstream UC **starts at t = 1**
  (`ycur = 1`, prior status off, `ic_power ≈ 0`) gets `big_m = 0`, leaving
  `variable[1] - 0 ≤ r_up·dt (+ slack)` fully binding → infeasible (or slack-priced) whenever
  `Pmin > r_up·dt`, which is common for thermal units. Shutdown at t=1 *is* relaxed — the asymmetry
  vs. t ≥ 2 is the bug.
- **Fix:** mirror the t ≥ 2 formula using the initial status:
  - `ic_power isa Float64` (non-recurrent builds): `y_init = ic_power > ABSOLUTE_MIN ? 1.0 : 0.0`
    (use the package's existing tolerance constant; add one if absent) and
    `big_m = power_limits.max * (2 - y_init - ycur)`.
  - `ic_power isa JuMP.VariableRef` (recurrent solves — value unknown at build): use the affine
    relaxation `big_m = power_limits.max * (1 - ycur) + (power_limits.max - ic_power)`. This is
    exact for a start from off (`ic_power = 0` → adds `Pmax`), tight for high-output on-units, and
    only moderately loose for low-output on-units; it never cuts feasible points. Add a comment
    stating the rigorous fix is threading a status initial condition (POM follow-up — flag in PR).
  Implement via dispatch on the IC value type (`_t1_ramp_big_m(ic_power::Float64, ycur, pmax)` /
  `(::JuMP.VariableRef, ...)`), not `isa`.
- **Test:** mock device with `Pmin = 50, Pmax = 100, ramp_up·dt = 10`; OnStatusParameter = 0 before /
  1 at t=1; assert the t=1 ramp-up constraint admits `variable[1] = 50` (build the model and check
  constraint feasibility at that point, or solve with the mock optimizer).

### Task 15: NMDT/DNMDT epigraph tightening mixes normalized and unnormalized frames — wrong models **by default**

- **File:** `src/quadratic_approximations/nmdt_common.jl:244-276` (driven from
  `nmdt.jl:38-46, 115-122` — `epigraph_depth = 3 * depth` defaults it ON) — HIGH
- **Problem:** `_tighten_lower_bounds!` builds `epi_expr` over `x_disc.norm_expr` with bounds
  `(0,1)` — it lower-bounds the **normalized** `xh² ∈ [0,1]` — then constrains
  `result_expr[name, t] >= epi_expr[name, t]` where `result_expr` is in **original units**
  (assembled at nmdt_common.jl:400-410: `lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min`).
  Valid only for domains exactly `[0,1]` — which is the only domain the test suite uses. For
  domain `[-1,1]` at `x = 0` the cut renders the model **infeasible**; for `[0,2]` it collapses the
  lower envelope (silent massive underestimation of x²), compounded because `tighten = true`
  simultaneously drops the McCormick lower bounds (`nmdt.jl:193/291`, `nmdt_common.jl:217/335`).
- **Fix:** unnormalize the cut. Pass the per-name bounds into `_tighten_lower_bounds!` and emit:
  ```julia
  lx = b.max - b.min
  epi_cons[name, t] = JuMP.@constraint(jump_model,
      result_expr[name, t] >=
      b.min^2 + 2.0 * b.min * lx * x_disc.norm_expr[name, t] + lx^2 * epi_expr[name, t])
  ```
  (this is exactly how `sawtooth.jl:192-197` does its epigraph tightening, in original units).
  Check the bilinear NMDT path for the same pattern before closing.
- **Test:** extend `test/test_nmdt_approximations.jl` with non-[0,1] domains: `(min=-1.0, max=1.0)`
  must remain feasible at `x = 0` with tightening on; `(min=0.0, max=2.0)` approximation of `x = 1`
  must be ≈ 1 within the documented tolerance.

### Task 16: PWMCC validation admits misaligned partitions that cut off feasible points

- **Files:** `src/quadratic_approximations/solver_sos2.jl:42-51`, `src/quadratic_approximations/manual_sos2.jl:37-48`
  (cut at `pwmcc_cuts.jl:205-212`) — HIGH
- **Problem:** the chord cut `q ≤ (brk_k + brk_{k+1})·v − brk_k·brk_{k+1}` is valid for the SOS2
  PWL value only when every PWMCC boundary coincides with a PWL breakpoint — for uniform grids,
  `depth % pwmcc_segments == 0`. The constructors only check `pwmcc_segments ≤ depth`.
  Verified counter-example: `depth = 3, pwmcc_segments = 2` on `[0,1]` — PWL forces
  `q = 5/18 ≈ 0.2778` at `v = 1/2`, both chords cap `q ≤ 0.25` → the band `v ∈ (4/9, 5/9)`
  becomes MIP-infeasible (~11% of the domain silently excluded).
- **Fix:** in both constructors require
  `pwmcc_segments == 0 || depth % pwmcc_segments == 0` with an actionable error message.
- **Test:** `@test_throws ArgumentError SolverSOS2QuadConfig(depth = 3, pwmcc_segments = 2)`;
  `SolverSOS2QuadConfig(depth = 4, pwmcc_segments = 2)` constructs.

### Task 17: Duration retrospective constraints — container axis built from the wrong IC column

- **File:** `src/common_models/duration_constraints.jl:50-53` vs `87-99` — MEDIUM
- **Problem:** `device_name_sets` filters non-`nothing` ICs from **column 1** (up) only, but the
  down loop iterates **column 2** and writes `con_down[name, t]`. Devices with up-IC `nothing` but
  down-IC set → `KeyError`; up-IC set but down-IC `nothing` → `#undef` rows in `con_down`
  (`UndefRefError` on any later full container read). The other three duration functions use the
  unfiltered column-1 names (only the undef-row half of the hazard).
- **Fix:** build one filtered name set per column and pass each to its own
  `add_constraints_container!`; keep loop filters consistent with the axis used.
- **Test:** ICs with `(up = 5.0, down = nothing)` for one device and `(nothing, 5.0)` for another;
  both containers must have exactly their own device axis and no `#undef` entries.

### Task 18: Service variables ignore the warm-start setting

- **File:** `src/common_models/add_variable.jl:122-123` (device version gates at 70-73) — MEDIUM
- **Problem:** `add_service_variables!` calls `JuMP.set_start_value` unconditionally; the paired
  device method wraps it in `if get_warm_start(settings)`. Copy-paste asymmetry.
- **Fix:** hoist `settings = get_settings(container)` and gate identically. While here, note the
  ub/lb asymmetry (line 117 vs 120: lb has a `!binary` guard, ub doesn't) — confirm intent and
  align or comment.
- **Test:** `Settings(...; warm_start = false)` → service variables have no start values.

### Task 19: `_get_minutes_per_period` — type-unstable, misclassifies exactly-1-minute, `InexactError` on non-whole minutes

- **File:** `src/common_models/rateofchange_constraints.jl:1-11` — MEDIUM
- **Problem:** returns `Int` or `Float64` per branch (feeds `dt` into all four ramp builders);
  `resolution > Dates.Minute(1)` sends exactly `Minute(1)` into the "under 1-minute" branch with a
  spurious warning; and `Dates.Minute(Second(90))` (any >1-min non-whole-minute resolution) throws
  `InexactError`. The line-1 comment "NOTE: not included currently." is stale.
- **Fix:**
  ```julia
  function _get_minutes_per_period(container::OptimizationContainer)
      resolution = get_resolution(container)
      resolution < Dates.Minute(1) &&
          @warn("Not all formulations support under 1-minute resolutions. Exercise caution.")
      return Dates.value(Dates.Millisecond(resolution)) / 60_000.0
  end
  ```
- **Test:** `Minute(1)` → 1.0 with no warning; `Second(90)` → 1.5; `Hour(1)` → 60.0.

### Task 20: `set_value!(::InMemoryDataset{2}, ::DenseAxisArray{Float64,2}, index)` slices the incoming array by the store row index

- **File:** `src/core/dataset.jl:177-180` (reached from `emulation_model_store.jl:109-120`) — MEDIUM
- **Problem:** `s.values[:, index] = vals[:, index]` — `index` is the execution counter, so at
  execution 3 it writes the 3rd **time column** of the incoming solve output, and `BoundsError`
  once `index > horizon`. The comment above the method states incoming vals carry a single step.
  PSI guarded this path with an assert; the port replaced the guard with silently wrong indexing.
- **Fix:** `IS.@assert_op size(vals, 2) == 1` (matching the documented contract) and write
  `s.values[:, index] = vals[:, 1]`. If multi-column emulation writes are intended later, that is a
  separate feature — fail fast now.
- **Test:** dataset with 4 columns × 10 executions; write a `(names × 1)` array at index 3; assert
  column 3 holds it; a `(names × 24)` array must throw.

### Task 21: Emulation-model cluster — phantom datasets, template aliasing, collapsed timestamps, broken exports, dead `empty!` branches

- **Files:** `src/operation/emulation_model_store.jl`, `src/operation/emulation_model.jl`,
  `src/operation/problem_outputs.jl`, `src/core/optimization_problem_outputs.jl`,
  `src/operation/store_common.jl` — MEDIUM ×5 (single agent; interlocking)
- **Problems / fixes:**
  a. `emulation_model_store.jl:86-95` (`initialize_storage!`): missing
     `!should_write_resulting_value(key) && continue` (the decision store has it at
     `decision_model_store.jl:41`) → never-written all-NaN datasets allocated for every
     parameter/expression key, memory ∝ run length, silent NaN reads. **Fix:** add the same skip.
  b. `emulation_model.jl:37`: `finalize_template!(template, sys)` mutates and **aliases** the
     caller's template (DecisionModel deepcopies via `_deepcopy_template`, `decision_model.jl:62-63`).
     **Fix:** `template_ = _deepcopy_template(template); finalize_template!(template_, sys)` and
     store `template_`.
  c. `problem_outputs.jl:74` + `optimization_problem_outputs.jl:489-493`: emulation outputs store a
     **one-element** `StepRange(initial_time, resolution, initial_time)`, and `_process_timestamps`
     papers over it with `repeat(get_timestamps(res), def_len)` → every row labeled `initial_time`;
     `start_time > initial_time` → `collect(nothing:def_len)` `MethodError`. **Fix:** store
     `range(initial_time; step = get_resolution(model), length = <executions written>)` (take the
     length from the store's last recorded row) and delete the `repeat` special case; handle
     `findfirst === nothing` with an actionable error.
  d. `store_common.jl:17-20` (`_export_container_output!`): `range(index; length, step::Millisecond)`
     throws for emulation `index::Int`; the column would be wrong even if it didn't. **Fix:**
     compute the timestamp column from the model — pass `initial_time + (index - 1) * resolution`
     (or the `update_timestamp`) for `EmulationModelIndexType`, dispatching on index type.
  e. `emulation_model_store.jl:42-50, 67-72`: `Base.empty!`/`Base.isempty` carry dead
     `:values`/`:timestamps`/`:update_timestamp` branches (`DatasetContainer` has no such fields;
     `store.update_timestamp` doesn't exist and would throw; `isempty`'s `iszero(val) && return false`
     is inverted). **Fix:** iterate `fieldnames(DatasetContainer)`, `empty!` each dict, plus
     `empty!(store.optimizer_stats)`; `isempty` = all dicts empty && stats empty.
  f. **Re-enable `test/test_model_emulation.jl`** (header says "FIXME not working and not included")
     once a-e and T4 land; fix residual failures it surfaces (this is the verification vehicle for
     the emulation path). If full re-enablement stalls, carve out the passing subset and leave a
     tracked TODO with the specific failures.
- **Tests:** beyond (f): store-level unit tests for a (phantom-dataset absence), c (timestamps
  vector), d (export of an emulation index writes a DateTime column).

### Task 22: `update_numerical_bounds` mixes signed values with `abs` comparisons and skips max via `elseif`

- **File:** `src/operation/model_numerical_analysis_utils.jl:64-75` — MEDIUM
- **Problem:** compares `v.min > abs(value)` / `v.max < abs(value)` but stores the **signed**
  `value`; one negative stored min poisons all subsequent comparisons and the
  `max - min > 1e9` conditioning checks (`optimization_model_interface.jl:210-229`) report garbage.
  The `elseif` also means the first sample never initializes `max`.
- **Fix:**
  ```julia
  a = abs(value)
  if v.min > a; set_min!(v, a); set_min_index!(v, idx); end
  if v.max < a; set_max!(v, a); set_max_index!(v, idx); end
  ```
- **Test:** feed coefficients `[-2.0, 0.5]` → min = 0.5, max = 2.0.

### Task 23: Constraint-container bookkeeping in approximations (model is right, containers are wrong)

- **Files:** `src/quadratic_approximations/manual_sos2.jl:249-257`,
  `src/quadratic_approximations/sawtooth.jl:165-174 + 252-267`,
  `src/quadratic_approximations/epigraph.jl:116-124 + 185-193` — MEDIUM ×3
- **Problems / fixes:**
  1. manual_sos2: adjacency loop stores at `adj_cons[name, i + 1, t]` for `i in 2:(n_points-1)` —
     slot 2 left `#undef` (UndefRefError on container iteration/serialization), slot `n_points`
     double-written. **Fix:** store at `adj_cons[name, i, t]`.
  2. sawtooth: `mip_cons` axes are `(names, 1:4, time_steps)` but writes happen inside
     `for j in alpha_levels` — each level overwrites the previous; only level L's 4 refs survive.
     **Fix:** add the level axis — `(names, alpha_levels, 1:4, time_steps)` — and index `[name, j, c, t]`.
  3. epigraph: same pattern with `lp_cons` `(names, 1:2, time_steps)` inside `for j in 1:depth`.
     **Fix:** add the level axis.
- **Test:** for each: build with depth ≥ 2 and assert every container index is assigned
  (`isassigned` sweep) and constraint counts match formula (e.g. manual SOS2: `n_points` adjacency
  rows per (name, t)).

### Task 24: Incremental PWL δ variables unbounded by default; ordering constraints not stored

- **File:** `src/quadratic_approximations/incremental.jl:73-100, 230-245` — MEDIUM
- **Problems:**
  1. The formulation requires `δ ∈ [0,1]` (docstring line ~19; ordering `δ_{i+1} ≤ z_i ≤ δ_i`), but
     bounds are only set when the extension hooks `get_variable_upper_bound`/`get_variable_lower_bound`
     return non-`nothing` — IOM's defaults return `nothing` (`common_models/interfaces.jl:64-77`), so
     δ is **unbounded** → silently invalid approximation values.
  2. The two ordering constraints per segment are anonymous — never stored (violates the
     container convention; uninspectable/unexportable).
- **Fixes:** default δ bounds: `JuMP.set_lower_bound(v, something(lb, 0.0))`;
  `JuMP.set_upper_bound(v, something(ub, 1.0))` (hooks remain overrides). Store the ordering
  constraints in a sparse constraints container (axes `(name, segment, 1:2, t)` or two meta-suffixed
  containers).
- **Test:** build with default hooks; assert `has_lower_bound`/`has_upper_bound` with values 0/1 on
  every δ; assert ordering-constraint container populated.

### Task 25: Objective-function semantics block [POM-CHECK — implement recommended fixes, list all in PR]

- **Files:** `src/objective_function/` — MEDIUM ×5; these change objective values or cost routing,
  so they ship together with explicit PR notes for POM maintainers.
- **Items:**
  a. **Quadratic fuel-curve multiplier** (`quadratic_curve.jl:226, 246-256`): code multiplies
     `multiplier * proportional_term_per_unit` and `multiplier * quadratic_term_per_unit` directly
     under the comment "Multiplier is not necessary here. There is no negative cost for fuel
     curves."; the linear fuel path (`linear_curve.jl:60-64`) omits the multiplier per the same
     comment. **Recommended fix:** drop `multiplier *` from both arguments (align with the linear
     path and the comment). Only the static `Float64` fuel-cost branch changes.
  b. **`is_nontrivial_offer` inverted for TS curves** (`value_curve_cost.jl:170-180`): the docstring
     defines it as "is this carrying meaningful data (vs the ZERO_OFFER_CURVE placeholder)"; a
     time-series-backed curve always carries data, yet the TS method returns `false` — load
     formulations gating on it silently drop time-varying decremental offer costs.
     **Recommended fix:** return `true`; check POM's load-formulation gates.
  c. **Proportional (no-load/fixed) costs not scaled by `dt`** (`proportional.jl:28` and `:72`):
     `rate = cost_term * multiplier` multiplies the variable at every timestep with no resolution
     scaling, while the sibling linear-variable-cost helper applies
     `dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR`
     (`cost_term_helpers.jl:220-222`). At 15-min resolution proportional costs are 4× overweighted
     relative to energy costs in the same objective. **Decision required:** if `proportional_cost`
     contract is $/h, apply `dt` in both `add_proportional_cost!` and
     `add_proportional_cost_maybe_time_variant!`; if the contract is $/timestep (legacy PSI
     behavior), add a loud docstring to both and to `proportional_cost` stubs instead. Implement
     the `dt` scaling unless POM maintainers say otherwise; this MUST be called out in the PR.
  d. **Breakpoint parameters classified as `TimeSeriesParameter`** (`offer_curve_types.jl:25` vs
     `:38`): slope params are `ObjectiveFunctionParameter` (always Float64 storage) but breakpoint
     params are `TimeSeriesParameter`, whose `add_param_container!` path uses
     `get_param_eltype(container)` → `JuMP.VariableRef` storage in recurrent-solve builds — which
     every reader requires to be Float64 (`value_curve_cost.jl:214-217, 324-327`;
     `cost_term_helpers.jl:168-176` takes `rate::Float64`; `variant_terms::GAE`).
     **Recommended fix:** `abstract type AbstractPiecewiseLinearBreakpointParameter <: ObjectiveFunctionParameter end`,
     matching its slope twin and the rebuild-variant-terms-each-step architecture. Confirm POM's
     `add_parameters!` implementations for breakpoint params construct Float64 containers.
  e. **TS delta-PWL width constraints are anonymous and value-baked**
     (`objective_function_pwl_delta.jl:145-147` via `value_curve_cost.jl:292-311`): per-segment
     `δ_k ≤ breakpoints[k+1] − breakpoints[k]` bakes build-time Float64 values into anonymous
     constraints; nothing can update them when breakpoint parameters change across simulation
     steps → stale segment capacities after step 1. **Recommended fix:** store them in a sparse
     constraints container indexed `(name, segment, t)` so the parameter-update path can
     `JuMP.set_normalized_rhs` them; wire that update where breakpoint parameters are refreshed
     (extension point — document it for POM). Also correct the file-top comment claiming width
     bounds "naturally enforce ordering" (they don't — convexity/curvity validation is what makes
     the LP correct; the TS path never runs `curvity_check` — add the check or document the
     monotone-slope requirement).
- **Tests:** a: static quadratic fuel cost with multiplier -1 formulation → cost sign unchanged
  after fix; b: TS-backed curve → `is_nontrivial_offer == true`; c: build at 30-min resolution and
  assert objective coefficient halves (if dt applied); d: recurrent-solve container →
  breakpoint param arrays are Float64; e: width constraints present in a container with correct
  RHS, and `set_normalized_rhs` round-trip.

### Task 26: `find_timestamp_index` — BoundsError on length-1 input; silent wrong index for unaligned dates

- **File:** `src/utils/datetime_utils.jl:12-21` — MEDIUM
- **Problem:** `dates[2] - dates[1]` throws for a length-1 vector with a non-matching date (masking
  the intended descriptive error), and an off-grid `date` integer-divides down to the previous
  index and is returned **silently** — dataset reads/writes (`dataset.jl:118,126,167`) at an
  unaligned timestamp hit the wrong row.
- **Fix:** guard `length(dates) >= 2` before the stride computation; validate alignment:
  `(date - first(dates)) % dates_resolution == Dates.Millisecond(0) || error(...)`, and keep
  uniform-stride as a documented precondition.
- **Test:** length-1 vector + mismatched date → descriptive error; `date` between grid points →
  error, not previous index.

### Task 27: `OptimizationContainer` objective-expression hygiene

- **File:** `src/core/optimization_container.jl` — MEDIUM + LOW (single task: same file)
- **Items:**
  1. (MEDIUM) Lines 32-47: `get_objective_expression(v::ObjectiveFunction)` — the AffExpr branch
     (line 44) **mutates** `variant_terms` by `add_to_expression!(v.variant_terms, v.invariant_terms)`.
     Non-idempotent exported getter: a second call before `reset_variant_terms` double-counts the
     invariant terms in the objective (the QuadExpr branch at 39-41 is correctly non-mutating).
     **Fix:** mirror the QuadExpr branch — build a temp `AffExpr`, add both, return it (one
     allocation per solve step, negligible).
  2. (LOW) Lines 277-279: `@debug begin JuMP.set_string_names_on_creation(JuMPmodel, true) end` —
     a side effect inside a logging macro: enabling debug logging silently overrides the
     user's `store_variable_names` setting. **Fix:** replace with an explicit, documented check
     (or delete); never mutate state inside `@debug`.
  3. (LOW) Lines 1317-1329: `calculate_dual_variables!` assigns `status =` then `return`s nothing —
     dead variable, inconsistent with its callees returning `RunStatus`. **Fix:** `return status`
     or drop the assignments (pick one; callers ignore the value today).
  4. (LOW) Line 119: constructor builds `OrderedDict{InitialConditionKey, Vector{InitialCondition}}()`
     for a field declared `OrderedDict{InitialConditionKey, Vector{<:InitialCondition}}` — silent
     convert. **Fix:** construct the declared type directly.
  5. (LOW) Lines 1139-1147 (`write_initial_conditions_data!`): uses `ic_container_dict[key]` inside
     `for (key, field_container) in ic_container_dict` — use `field_container`.
- **Test:** for item 1: call `get_objective_expression(obj_fn)` twice with nonzero variant terms;
  assert equal results and `variant_terms` unchanged.

### Task 28: `sparse_container_spec` aliases one mutable zero across all entries; `remove_undef!` errors on plain `Array`s

- **File:** `src/utils/jump_utils.jl:467-474, 500-512` — MEDIUM ×2
- **Problems / fixes:**
  1. `Dict{eltype(indexes), T}(indexes .=> zero(T))` for `T <: JuMP.AbstractJuMPScalar` evaluates
     `zero(T)` **once** — every entry shares one mutable `AffExpr`; the first in-place
     `add_to_expression!` corrupts all entries. **Fix:**
     `Dict{eltype(indexes), T}(k => zero(T) for k in indexes)`.
  2. `remove_undef!(expression_array::AbstractArray)` requires `.data` — plain `Vector`/`Matrix`
     (e.g. the package's own `JuMPAffineExpressionArray = Matrix{GAE}`) throw. It is exported.
     **Fix:** restrict the existing method to `::DenseAxisArray`; add
     `remove_undef!(a::Array)` operating via `eachindex`/`isassigned` (dispatch, no `isa`).
- **Tests:** sparse AffExpr container: mutate one entry, assert others unchanged;
  `remove_undef!(Matrix{JuMP.AffExpr}(undef, 2, 2))` fills zeros.

---

# Phase 3 — Performance

### Task 29: O(N²) objective accumulation for quadratic cost terms

- **File:** `src/core/optimization_container.jl:1189-1197` (hot from `objective_function/quadratic_curve.jl` per device×timestep) — HIGH
- **Problem:** `container.objective_function.invariant_terms += cost_expr` builds a brand-new
  `QuadExpr` copying every accumulated term on each call → total O(N²) in the number of quadratic
  terms (1000 units × 48 steps ≈ 10⁹ dict-entry copies at build). The AffExpr overload (1199-1205)
  already uses in-place `add_to_expression!`.
- **Fix:** promote once, then always add in place:
  ```julia
  function add_to_objective_invariant_expression!(container::OptimizationContainer, cost_expr::JuMP.GenericQuadExpr)
      obj = container.objective_function
      if obj.invariant_terms isa JuMP.GenericAffExpr   # one-time promotion guard on a 2-member union
          promoted = JuMP.GenericQuadExpr{Float64, JuMP.VariableRef}()
          JuMP.add_to_expression!(promoted, obj.invariant_terms)
          obj.invariant_terms = promoted
      end
      JuMP.add_to_expression!(obj.invariant_terms, cost_expr)
      return
  end
  ```
  (The `isa` here guards JuMP type promotion on the concrete 2-member `JuMPScalarExpr` union —
  compile-time union split, the sanctioned exception per the 2026-04-06 plan.)
- **Test:** correctness: accumulate 3 quad + 2 aff terms, compare against `sum`. Perf (optional):
  `@allocated` growth linear, not quadratic, for 1000 adds.

### Task 30: Non-`const` exported enum aliases defeat inference in the solve loop

- **File:** `src/core/definitions.jl:156-159` — MEDIUM
- **Problem:** `ModelBuildStatus = ISOPT.ModelBuildStatus`, `SimulationBuildStatus = ...`,
  `RunStatus = ...` are untyped non-const globals (explicitly banned), read on every solve:
  `execute_optimizer!` (`optimization_container.jl:425,434,450`), `solve_model!`
  (`optimization_model_interface.jl:159`), `is_built`/`isempty` checks, `dual_processing.jl:85,124`.
- **Fix:** add `const` to all three. Also type `ENUM_MAPPINGS` (line 167): at least
  `const ENUM_MAPPINGS = Dict{DataType, Dict{String, Any}}()` (LOW).

### Task 31: `jump_value(::VariableRef)` does two MOI queries per element in every store write

- **File:** `src/utils/jump_utils.jl:21-29` (hot from `store_common.jl:86,152,184,222` per key per step) — MEDIUM
- **Problem:** broadcast over every container each step; `JuMP.is_fixed` (a `MOI.is_valid` probe)
  and `JuMP.has_values(input.model)` (a `primal_status` solver round-trip) run **per element** for
  answers constant per model/solve. Also uses `input.model` instead of `JuMP.owner_model`.
- **Fix:** hoist the model-level check: in the `write_model_*_outputs!` functions query
  `has_vals = JuMP.has_values(jump_model)` once and pass down (add
  `jump_value(v::JuMP.VariableRef, has_vals::Bool)` or a container-level
  `jump_values(container_array, has_vals)` helper); when `!has_vals`, `fill` NaN without
  broadcasting; keep only the per-element `is_fixed` branch. Preserve the exported 1-arg
  `jump_value` for compatibility.
- **Test:** existing store-write tests must pass; spot-check fixed-variable handling.

### Task 32: `is_milp` materializes the full binary-constraint list per solve

- **File:** `src/core/optimization_container.jl:204-216` (called per solve at line 448 and per
  `add_dual_container!` at 754) — MEDIUM
- **Fix:** `return JuMP.num_constraints(get_jump_model(container), JuMP.VariableRef, MOI.ZeroOne) > 0`
  (keep the `supports_milp` early return).

### Task 33: Concrete-field batch (per-step dynamic dispatch from struct fields)

Five independent sub-tasks; verify no constructor breaks; run full tests after each.

- **a.** `src/core/dataset.jl:23-33`: `InMemoryDataset{N}.values::DenseAxisArray{Float64, N}` is
  abstract (4-param type, 2 bound). **Fix:** `mutable struct InMemoryDataset{N, A <: DenseAxisArray{Float64, N}} ... values::A ...`;
  update constructors/aliases (`DatasetContainer{InMemoryDataset}` usages keep working —
  `InMemoryDataset` remains the UnionAll supertype; check `EmulationModelStore()` constructor).
- **b.** `src/core/model_internal.jl:12,16`: `time_series_cache::Dict{TimeSeriesCacheKey, <:TimeSeriesCache}`
  is a UnionAll field — declare `Dict{TimeSeriesCacheKey, TimeSeriesCache}` (what the constructor
  already builds); `store_params::Union{Nothing, AbstractModelStoreParams}` → `Union{Nothing, ModelStoreParams}`
  (only in-repo subtype; grep POM for other subtypes first — if any exist, parameterize instead).
- **c.** `src/operation/decision_model.jl:21-29`, `src/operation/emulation_model.jl:1-9`: bare
  `internal::ModelInternal` (parametric) and `store::EmulationModelStore` are UnionAll fields hit
  by every `get_optimization_container`/`get_store` call per step. **Fix:** annotate the concrete
  configurations constructed in-repo: `internal::ModelInternal{OptimizationContainer}`,
  `store::EmulationModelStore{InMemoryDataset}` — confirm with greps; if POM constructs others,
  parameterize the model structs instead.
- **d.** `src/core/optimization_container_metadata.jl:3-5`: `container_key_lookup::Dict{String, <:OptimizationContainerKey}`
  → `Dict{String, OptimizationContainerKey}`; make `add_container_key!` return `nothing` (line 43-44).
- **e.** `src/core/device_model.jl:61`, `src/core/service_model.jl:36`:
  `feedforwards::Vector{<:AbstractAffectFeedforward}` → `Vector{AbstractAffectFeedforward}`
  (heterogeneous by design), converting in constructors. Also `src/core/outputs_by_time.jl:4`:
  `resolution::Dates.Period` (abstract) → `Dates.Millisecond` (everything upstream converts already)
  — coordinate with Task 3's inner constructor. And `src/core/parameter_container.jl:62-65`:
  `affected_keys::Set` → `Set{OptimizationContainerKey}` (constructor passes `Set()` today).

### Task 34: Device×time build-loop hoisting (hottest build paths)

Three sub-tasks, one file each:

- **a.** `src/common_models/add_variable.jl:57-74, 108-124`: per-(t,d) iteration recomputes
  `get_variable_upper_bound`/`lower_bound`/`warm_start` (device-invariant), re-reads
  `get_warm_start(settings)`, calls `get_jump_model(container)` inside the macro, and re-does the
  keyed `variable[name, t]` lookup for every `set_*` call. **Fix:** device-outer loop; compute
  `name/ub/lb/init` once per device; `jm = get_jump_model(container)` once per function;
  `v = variable[name, t] = JuMP.@variable(jm, ...)` then use `v`. In the service version also hoist
  `IS.get_name(service)` out of the `base_name` interpolation.
- **b.** `src/common_models/duration_constraints.jl` (4 functions): hoist
  `ics_up = view(initial_duration, :, 1); ics_dn = view(initial_duration, :, 2)` out of the t-loop
  (currently `initial_duration[:, 1]` **copies a column twice per timestep** — lines 71-72/87,
  164-165/182, 275-276/305, 399-400/421); hoist `jm = get_jump_model(container)` out of all
  `@constraint` calls; consider device-outer/time-inner so `get_component_name(ic)`/`get_value(ic)`/
  `duration_data[ix]` are computed once per device.
- **c.** `src/common_models/rateofchange_constraints.jl` + `src/common_models/range_constraint.jl`:
  - `_get_ramp_slack_vars` (37-50) does two keyed container lookups per (device, t) and returns a
    Union forcing dynamic dispatch at every call (109, 117, 158, 165, 231, 243, 320). **Fix:** fetch
    both slack arrays (or `nothing`) once before the device loop and pass them down; index inside.
  - O(N²) membership: `name ∉ device_name_set` over a `Vector{String}` (104, 152, 310). **Fix:**
    `Set(device_name_set)` for membership (keep the Vector for container axes).
  - `hasmethod(get_must_run, Tuple{V})` per device (317). **Fix:** hoist above the loop (or a
    `supports_must_run` trait with POM override).
  - range_constraint.jl: device-invariant `IS.get_name(device)`/`get_min_max_limits(device, T, W)`
    recomputed per t (103-107, 222-227, 270-276, 465-468, 495-498) — device-outer loops; the
    `invert_binary` ternary (273) makes `bin::Union{AffExpr, VariableRef}` per element — split into
    two typed loops or dispatch on a singleton.
  - range_constraint.jl:409 + 525-530: `IS.has_time_series` evaluated twice per device (the file
    itself marks it "PERF: compilation hotspot") — keep the filtered device list from the names
    comprehension and iterate it in `_bound_range_with_parameter!` instead of re-checking.

### Task 35: Objective-function build-loop hoisting

- **Files:** `src/objective_function/piecewise_linear.jl`, `value_curve_cost.jl`, `proportional.jl`,
  `start_up_shut_down.jl`, `cost_term_helpers.jl` — MEDIUM
- **Items:**
  1. `piecewise_linear.jl:128-146`: per-t call of the cost_function-typed
     `get_pwl_cost_expression_lambda` re-runs curve normalization (`get_value_curve`, `power_units`,
     base powers, `get_piecewise_pointcurve_per_system_unit` — fresh vectors), `get_resolution`,
     and `_get_pwl_cost_multiplier`, although `add_pwl_term_lambda!` already computed the identical
     normalized `data`. **Fix:** hoist `dt`/`multiplier` and call the data-typed overload
     `get_pwl_cost_expression_lambda(container, T, name, t, data, dt * multiplier)`.
  2. `piecewise_linear.jl:219-225`: `get_parameter_array`/`get_parameter_multiplier_array`/
     `get_fuel_cost(component)` re-fetched every t — hoist.
  3. `value_curve_cost.jl:324-347` (`_fill_pwl_data_from_arrays!`): for non-SystemBase units,
     `get_piecewise_curve_per_system_unit` allocates two fresh vectors per (device, t) then
     `copyto!`s them back. **Fix:** compute `x_ratio`/`y_ratio` once per device before the time
     loop and apply in place (`breakpoints .*= x_ratio; slopes .*= y_ratio`). Same file 200-217
     (`_get_raw_pwl_data`): hoist the four parameter-container fetches out of the per-t path.
  4. `proportional.jl:36`, `start_up_shut_down.jl:56,65,118,128`, `cost_term_helpers.jl:224`:
     `get_variable(container, U, T)` keyed lookup inside the time loop — hoist the array.
     `start_up_shut_down.jl:124-127`: t-invariant `start_up_cost(...)`/`iszero` per t — hoist.
  5. `piecewise_linear.jl:109`: `all(iszero.((point -> point.y).(IS.get_points(data))))` →
     `all(p -> iszero(p.y), IS.get_points(data))` (two temp arrays per device).

### Task 36: Approximation build-loop hoisting

- **Files:** `src/quadratic_approximations/pwmcc_cuts.jl:150-163`, `sawtooth.jl:217-221`,
  `incremental.jl:77-95, 168-172` — MEDIUM/LOW
- **Items:** pwmcc: 7 breakpoint vectors depend only on `bounds[idx]` but are rebuilt per (name, t)
  — split into name-outer/time-inner loops. sawtooth: `saw_coeffs` same — hoist per name or inline
  the scalar. incremental: delete the dead `pwlvars` buffer; hoist `ub`/`lb` hook calls out of the
  segment loop; keep the fresh `VariableRef` in a local instead of re-lookup; replace the
  `R <: DCVoltage` branches (168-172, 203) with dispatch helpers (convention violation).

### Task 37: Emulation store write path — per-key copies and triple lookups

- **File:** `src/operation/emulation_model_store.jl:109-139` — MEDIUM
- **Problem:** `array[:, 1]` allocates a copy per key per step just to re-dispatch; the 1-D method
  then does three separate `container[key]` hash lookups.
- **Fix:** hoist `dataset = container[key]` once; pass a `view(array, :, 1)`-compatible path
  (add/adjust the `set_value!` method accordingly — coordinate with Task 20's assert).

### Task 38: Parameter-container micro-fixes on the per-step path

- **File:** `src/core/parameter_container.jl:171-178, 210-223` — MEDIUM
- **Items:** delete the dead duplicated `expand_ixs((_get_ts_uuid(...)), param_array)` at line 176
  (computed and discarded, then recomputed at 177; per-column per-step); in the TS
  `get_parameter_values` explode loop hoist `axes(param_array)[2:end]` and
  `axes(multiplier_array)[2:end]` out of the per-name loop.

### Task 39: Small-perf batch (LOW, mechanical)

- `src/utils/jump_utils.jl:458-462`: `cont.data .= fill(NaN, size(cont.data))` → `fill!(cont.data, NaN)`.
- `src/utils/jump_utils.jl:46-48` (`jump_fixed_value(::AffExpr)`): array comprehension inside `sum`
  → generator with `init = 0.0`.
- `src/utils/jump_utils.jl:566-586` (`_summary_to_dict!`): collapse the redundant `ismissing`
  branch (both branches identical); hoist the `fields` vector to a `const` tuple.
- `src/core/optimization_container.jl:360-366`: `!all(isnan.(x))` → `!all(isnan, x)` (two sites).
- `src/operation/time_series_interface.jl:28-29, 76-77`: `haskey` + `getindex` → single
  `get(cache, key, nothing)` pattern.
- `src/operation/optimization_model_interface.jl:47-50`: `deepcopy(::OptimizerStats)` per call →
  hand-written field-by-field `copy(::OptimizerStats)` (all fields isbits/Missing);
  `src/operation/decision_model_store.jl:166`: `deepcopy(data[index])` → `copy` of the
  DenseAxisArray (Float64 payload; verify callers only mutate top level).
- `src/common_models/set_expression.jl:11-12`: `has_container_key` + `get_expression` builds/hashes
  the key twice per call in the objective path — single lookup (`get`-style helper on the container).
- `src/common_models/get_time_series.jl:67-69`: `haskey(ts_names, T)` + `ts_names[T]` → one `get`.
- `src/core/optimizer_stats.jl:91-110`: `Vector(undef, n)` (`Vector{Any}`) + 21-arg splat →
  field-by-field construction.
- `src/core/external_evaluation.jl:36, 53-59`: `evaluators::Dict{DataType, Any}` →
  `Dict{DataType, AbstractEvaluator}`; make both `add_*!` return `nothing`.

---

# Phase 4 — Dead code, conventions, documentation

### Task 40: Delete verified-dead / can-never-work code

- `src/utils/jump_utils.jl:514-557`: both `_calc_dimensions` methods (no callers; debug messages
  log the wrong axis; `Dict{String, Any}` return). Grep `test/` too before deleting.
- `src/utils/file_utils.jl:29-31`: `find_variable_length` (no callers, untyped, crashes on empty).
- `src/utils/jump_utils.jl:16-19`: `write_data(::Float64, ...)` double-encodes JSON (no callers;
  live `write_data` methods are in `optimization_problem_outputs.jl`).
- `src/core/model_internal.jl:61-62` vs `83-84`: duplicate `set_store_params!` — keep the typed one.
- `src/common_models/constraint_helpers.jl:97-107`: `add_range_equality_constraint!` has zero
  callers and its documented use case (min≈max epsilon) is unimplemented in
  `add_range_constraints!` — either wire it in or delete it and fix the docstring at
  `range_constraint.jl:10-16` (see Task 42).
- `src/core/network_model.jl:245-265`: unreachable final `else` (first two conditions exhaust);
  also the string at line 253 ends in `\\` (escaped backslash inside a regular string — the error
  message contains a literal `\` + newline). Drop the dead branch; fix the continuation.

### Task 41: Convention violations (`isa`/`<:` → dispatch; key-construction; alias)

- `src/objective_function/objective_function_pwl_lambda.jl:240-248, 330-338, 372-380`: three
  copy-pasted blocks construct `ConstraintKey` directly and `_assign_container!` with
  `Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}` (abstract `ConstraintRef` eltype;
  violates the "keys only via `add_*_container!`" rule in `.claude/claude.md`). **Fix:** use
  `add_constraints_container!(...; sparse = true)` / `lazy_container_addition!` as
  `add_pwl_constraint_delta!` does.
- `src/objective_function/linear_curve.jl:61`: `if fuel_cost isa Float64` → split into
  `_add_fuel_linear_variable_cost!(..., ::Float64)` / `(..., ::IS.TimeSeriesKey)` mirroring
  `quadratic_curve.jl:167-196`.
- `src/core/settings.jl:97-103`: `if type <: Base.RefValue` → `_unwrap(x::Base.RefValue) = x[]; _unwrap(x) = x`.
- `src/core/optimization_problem_outputs_export.jl:107-113`: `typeof(fields) <: Set` →
  `_check_fields(fields::Set) = fields; _check_fields(fields) = Set(fields)`.
- `src/operation/decision_model.jl:53` / `src/operation/emulation_model.jl:34`: `name isa String`
  → `_model_name(::Nothing, ::Type{M})` / `(n::Symbol, _)` / `(n::String, _)` helpers.
- `src/operation/model_numerical_analysis_utils.jl:113,138` and
  `src/operation/optimization_debugging.jl:66,95`: `isa(..., SparseAxisArray)` →
  dispatch `_scan!(..., ::SparseAxisArray)` / `(..., ::DenseAxisArray)` (the TODO at 112 already
  asks for this). While in `optimization_debugging.jl`: add the missing `isassigned` guard at
  77-78 (its non-detailed twin has it at `model_numerical_analysis_utils.jl:122`).
- `src/InfrastructureOptimizationModels.jl:142`: `const POM = InfrastructureOptimizationModels` —
  in Sienna vocabulary POM means PowerOperationsModels (the downstream package); aliasing *this*
  package as `POM` is a landmine. The alias is unused in-repo (`POM.` appears only in comments).
  **Fix:** rename to `const IOM = InfrastructureOptimizationModels` (grep POM/downstream for
  `IOM.POM` usage first; if unsure, add `const IOM` and deprecate `POM`).
- `src/utils/indexing.jl:57-58` (`fix_expand`): non-scalar `getindex` copies the slice and the
  broadcast materializes a discarded array; also mutates JuMP state without `!`. **Fix:**
  `@views`/explicit loop; rename `fix_expand!` (no in-repo callers; check POM).
- `src/operation/problem_template.jl:146-150` (`_deepcopy_template`): nulls `PTDF_matrix`/`MODF_matrix`
  on the **input** template and restores only on success — wrap in `try ... finally`.
- `src/operation/optimization_model_interface.jl:161-163`: failed-solve dump uses a fixed filename;
  consecutive allowed failures overwrite — embed the timestamp like the export at line 148-149.

### Task 42: Documentation corrections (docs contradict code — fix docs, not code)

- `src/common_models/duration_constraints.jl`: LaTeX multipliers swapped vs code at 114/126 and
  213/225 (code is correct: `varstop × duration.up`, `varstart × duration.down`); argument lists
  document nonexistent `cons_name::Symbol`, `var_keys`, `initial_duration_on/off` params
  (35, 139, 236-242, 360-364) — the real argument is `initial_duration::Matrix`.
- `src/common_models/rateofchange_constraints.jl:56-72, 263-271`: docstrings disagree with the
  generated constraints (missing `· minutes_per_period`, wrong sign in the LaTeX down-ramp, and
  the start/stop relaxation documents `rate_data[2].max` while the code uses `power_limits.min` —
  code matches the standard SU≈Pmin formulation). Also `constraint_helpers.jl:138` vs `:146`:
  line 146 says "`bound_decrease = false` (default)" but the signature default is `true`.
- `src/common_models/add_variable.jl:16`: `lb \ge x ... \le ub` → `lb \le x \le ub`.
- `src/common_models/interfaces.jl:222-226`: make the `get_min_max_limits` fallback `error(...)`
  with an actionable message instead of returning `nothing` that callers dereference (`limits.min`).
- `src/quadratic_approximations/solver_sos2.jl:86`, `manual_sos2.jl:82`: "default 4" → "default 0"
  for `pwmcc_segments` (constructors default 0). `nmdt_common.jl:279`: docstring signature omits
  the `depth` positional arg.
- `src/core/optimization_container_keys.jl:52-56`: comment says AuxVarKey is the
  abstract-component exception; the dispatch exempts `ConstraintType` — resolve explicitly
  (decide which is intended; align comment or dispatch).
- `src/core/device_model.jl:30-37` / `src/core/service_model.jl:26-29`: kwargs documented as
  `feedforward`/`use_service_name` don't exist (`feedforwards`; no such kwarg).
- `src/utils/print_pt_v3.jl:145`: `first(...)` on possibly-empty affected values + silently drops
  the rest → `join(encode_key_as_string.(get_affected_values(v)), ", ")` guarded for empty.
- `src/objective_function/objective_function_pwl_delta.jl:16-17, 36-37`: comments claim width
  bounds enforce fill order (they don't) and `n_points` = number of delta variables (callers pass
  `n_segments`) — covered functionally in Task 25e; fix the comments with it.
- **`.claude/claude.md` refresh** (per its own maintenance note): the tree lists
  `objective_function/import_export.jl` (doesn't exist), omits `objective_function_pwl_lambda.jl`,
  `objective_function_pwl_delta.jl`, `core/optimization_container_utils.jl`,
  `core/external_evaluation.jl`, `core/dual_processing.jl`; `initial_conditions/` now contains only
  `calculate_initial_condition.jl`; `quadratic_approximations/` gained `nmdt.jl`/`nmdt_common.jl`/
  `pwmcc_cuts.jl`/`no_approx.jl` (listed) but check the rest of the tree against `ls -R src` and
  update.

---

# Final verification

- [ ] Full suite: `julia --project=test test/runtests.jl` — green, including the re-enabled
      emulation tests (T21f) and all new regression tests.
- [ ] Formatter: `julia scripts/formatter/formatter_code.jl` — clean tree.
- [ ] Optional: run `test/performance/performance_test.jl` before/after Phase 3 and record the
      build/solve timings in the PR description.
- [ ] PR description lists every **[POM-CHECK]** item (T8, T13, T14, T25a-e, T33b/c, T41 alias)
      with one line each on the behavior change, for downstream maintainers.
