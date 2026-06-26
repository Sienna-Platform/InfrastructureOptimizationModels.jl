# IOM port plan — PSI → InfrastructureOptimizationModels.jl

Scope: bring IOM (`InfrastructureOptimizationModels.jl`, the **generic optimization core** that POM
depends on) up to date with PSI (`PowerSimulations.jl`). IOM owns what POM does not: the
optimization container, dual processing, settings, serialization, parameter machinery, build/solve
lifecycle, objective-function machinery, and model-export.

Companion file: `pom_port_plan.md` (formulation changes + in-progress G-1 → PowerOperationsModels).

## Topology / ground rules
- A PSI change belongs in IOM iff it touches generic optimization infrastructure (NOT formulation-,
  device-, network-, or simulation-specific). Formulation parts go to POM; many PSI PRs are **SPLIT**.
- IOM independently tracks PSI and has its **own** PSI-port history (commits cite PSI #1559, 1568,
  1569, 1579, 1637). IOM sometimes **improves on** PSI (e.g. stores the in-flight serialization
  `Task` on `OptimizationContainer.serialization_task` rather than PSI's `ext`-dict hack).
- Verify by **symbol**, not by PR-number reference. Fork baseline ≈ PSI #1503; latest PSI PR = #1640.

---

## Confirmed NEEDS-PORT (symbol-verified ABSENT in IOM)

### 1. PR #1585 — dual-cache MIP-tolerance rounding (PRIORITY)
`core/dual_processing.jl` still uses the old `Dict{Symbol,Any}` cache and is missing:
- `struct VarRestoreInfo`
- `_round_cache_values!` (rounds cached values to dodge MIP-tolerance infeasibility on integer restore)
- the integer-variable restore fix.

→ `InfrastructureOptimizationModels.jl/src/core/dual_processing.jl`.
**⚠ Coordinate:** IOM appears to be independently reworking `dual_processing.jl`. Confirm with the
IOM maintainers that the rounding fix lands in the rework rather than blind-porting onto a moving file.

### 2. PR #1559 + #1561 (IOM portion of DLR) — Dynamic Line Ratings
The DLR feature is absent from both clones. The formulation/network parts go to POM (see pom plan);
the **IOM portion** is the generic parameter-type registration / container plumbing for the new
`DynamicBranchRatingTimeSeriesParameter` (time-series parameter type + `add_param_container!` wiring),
if it is to live in IOM's `core/time_series_parameter_types.jl` / `core/parameter_container.jl`
rather than POM. **Confirm the intended home** before porting — DLR may be implemented entirely in
POM using IOM's existing generic param machinery, in which case there is no IOM work here.

---

## Optional / minor residuals (decide if completeness matters)

### PR #1539 (and the #1505/#1506 slack fix it consolidates)
IOM is mostly already ported (`assign_maybe_broadcast!`/`expand_ixs`/`fix_maybe_broadcast!` in
`src/utils/indexing.jl` are generalized; `decision_model_store.jl` has 3D `write_result!`). Residual
gaps if you want full parity:
- 1-D integer-indexed `write_result!(... DenseAxisArray{T,1,<:Tuple{UnitRange}})` (the #1506 slack
  fix) — absent from `src/operation/decision_model_store.jl`.
- `emulation_model_store.jl` shows no 3D `write_result!` branch.
- `_update_parameter_values!` Service+EmulationModel method has no target (param-update path is
  restructured in IOM — likely intentional, verify).

These are low-value unless a concrete failure points at them.

---

## Already ported to IOM — do NOT redo (symbol-verified PRESENT)
For reviewer confidence; these PSI changes are confirmed present in IOM:
- **#1568** — perf consolidation (`dual_processing.jl`, optimization-container/settings trims, lifecycle).
- **#1563 / #1564** — production-cost expression refactor: `ConstituentCostExpression`,
  `FuelCostExpression`, etc. + `optimization_container` generalization (device cost terms live in POM).
- **#1591** — decision-model `interval` / `get_interval` support.
- **#1609** — parameter-broadcast fast path: `get_parameter_array_data`, `_set_parameter_at!`,
  `_set_multiplier_at!` (`core/parameter_container.jl`).
- **#1625 / #1626** — `OptimizationModelExportFormat` enum (`core/definitions.jl`),
  `Settings.export_optimization_model` + `_validate_export_optimization_model` (`core/settings.jl`),
  backgrounded serialization (`serialize_optimization_model`, `wait_for_serialization!`,
  `_copy_jump_model_for_export`/`_write_export_model`) — refined beyond PSI (task on
  `OptimizationContainer.serialization_task`). The `branches_modeled` trait is the POM half.
- **#1629** — time-varying ORDC objective params: generic `add_param_container!` for
  `ObjectiveFunctionParameter` (`optimization_container.jl:1055`) + `*PiecewiseLinearSlopeParameter`
  types (`src/objective_function/`).
- **#1633 / #1637** — SC slacks + pf/print fixes (IOM uses `column_labels` directly; the PrettyTables
  v2/v3 shim PSI added is unnecessary in IOM).
- **#1569** — serialize-System-into-HDF lifecycle (its surface `store_system_in_results` lives in POM
  operation models; the HDF write is simulation-layer = PSI-only).

---

## PSI-only (never IOM): docs, CI, version bumps, results-IO (`to_results_dataframe`,
`table_format` cache reads), simulation_state/partitions/store, PowerFlows-naming bumps.

---

## Suggested execution order
1. **#1585** — coordinate with the in-flight `dual_processing.jl` rework, then land the rounding fix.
2. **DLR IOM portion** — only after confirming DLR's param-type home is IOM (else it's all POM).
3. **#1539 residuals** — optional; do only if a failure implicates the 1-D/3D `write_result!` paths.

---

## Execution log (2026-06-26, verified against local PSI clone)

### 1. PR #1585 — **PORTED** (branch `ac/port-psi-1585-dual-mip-rounding`)
Did NOT blind-port PSI's `VarRestoreInfo` struct refactor. IOM's `dual_processing.jl` has
intentionally diverged and is *ahead* of PSI: it adds the `axes(dual) == axes(constraint)`
assertion in `_copy_dual_values!`, a sparse binary/integer relaxation warning, a detailed
`error` instead of `@assert !isempty(cache)`, and re-fixes fixed *binary* vars (not just
integer) via `_refix!`. Porting the struct would have undone those.

The actual bug #1585 fixes is **MIP-tolerance rounding** (a solver returns `0.9999997`
instead of `1.0`; fixing an integer var to that makes the relaxation infeasible). Landed the
functional fix only:
- Added `_round_cache_values!(cache::DenseAxisArray)` — only the dense method, because IOM
  `continue`s past `SparseAxisArray` containers before any fixing, so a sparse method would
  be dead code.
- Call it on `var_cache[key]` immediately before `JuMP.fix.(...)`. IOM's restore path already
  reuses `var_cache[key]` through `_refix!`, so rounding once also covers the integer-restore
  path PSI patched separately.
- Added a regression test (`_round_cache_values! snaps MIP-tolerance values to integers`).

Full suite green: 1330/1330 unit + 10/10 Aqua.

### 2. DLR IOM portion — **NO IOM WORK** (home confirmed = POM)
The DLR time-series rating parameter types live entirely in POM, not IOM:
`AbstractBranchRatingTimeSeriesParameter`, `BranchRatingTimeSeriesParameter`,
`PostContingencyBranchRatingTimeSeriesParameter` are defined in POM
`src/core/parameters.jl` and exported from `PowerOperationsModels.jl` (≈ lines 918–919).
They are referenced only by POM branch/network formulations
(`template_validation.jl`, `instantiate_network_model.jl`, `add_parameters.jl`, the AC-branch
device models) and build on IOM's existing generic `TimeSeriesParameter` + `add_param_container!`
machinery. IOM's `core/time_series_parameter_types.jl` deliberately keeps only the small
domain-agnostic set (`ActivePower*`, `ReactivePower*`, `Requirement*`). So the "confirm the
intended home" question resolves to POM — nothing to add in IOM.

### 3. PR #1539 residuals — **1-D `UnitRange` `write_output!` PORTED**; 3-D already present
IOM renamed PSI's `write_result!` to `write_output!` in `src/operation/decision_model_store.jl`.
The genuine #1539 broadcast generalizations are present and verified: `expand_ixs`,
`assign_maybe_broadcast!`, `fix_maybe_broadcast!` in `src/utils/indexing.jl`.

Re-checking the store dispatch by symbol (not by name) found a real gap, not a stale ref:
`initialize_storage!` allocates a time-only result (empty `column_names`) as a 1-D `UnitRange`
`DenseAxisArray`, but `write_output!` had no `DenseAxisArray{T, 1, <:Tuple{UnitRange}}` method —
only the 1-D `Vector{String}` one — so writing such a result would `MethodError`. This is exactly
PSI's #1506 system-slack method consolidated by #1539. **Ported** the 4-line method (mirrors the
existing 1-D `Vector{String}` method) plus a `DecisionModelStore` write/read round-trip test in
`test/test_model_store.jl`. The 3-D paths were already covered (IOM carries two 3-D methods, one
more than PSI). The EmulationModel store already handled 1-D generically
(`DenseAxisArray{Float64, 1}`), so only the decision store needed the fix. "No failures" reflected
that no current IOM/POM formulation builds a time-only result, not robustness.
