# InfrastructureOptimizationModels.jl — Claude Guide

Platform-wide Sienna conventions (performance, type stability, formatter, environments, code style) live in `.claude/Sienna.md` — read it too. This file is repo-specific and does not restate them.

## Purpose & Role

IOM is the **optimization-model utility layer** of Sienna — the abstraction beneath
PowerSimulations.jl (PSI) and PowerOperationsModels.jl (POM). It defines the generic,
domain-agnostic machinery for *building* optimization models: the optimization container,
variable/constraint/expression/parameter abstractions and their containers, model wrappers
(`DeviceModel`/`NetworkModel`/`ServiceModel`), the two operation-model types
(`DecisionModel`, `EmulationModel`), output stores, and objective-function formulations.

It deliberately does **not** know about power-system domain concepts. It builds on
InfrastructureSystems.jl (IS) types and exposes extension points that downstream packages
implement with concrete PSY types. Julia compat: `^1.10`. IS compat: `3, 4` (currently
sourced from the `IS4` branch).

**Costs vs objective functions:** IOM defines *objective functions* (delta PWL, lambda PWL,
proportional, quadratic formulations translating IS `ValueCurve` types into JuMP terms), not
*costs*. "Production cost", "fuel cost", "start-up cost" are domain concepts owned by POM.
IOM functions may name PSY cost types in signatures (for dispatch) but their job is building
JuMP objective expressions.

## Optimization Model Construction Conventions

### `add_*!()` methods must not return collections
Methods that create variables, constraints, or expressions (`add_variables!`,
`add_constraints!`, `add_expressions!`, etc.) must always end with a bare `return` (i.e.,
return `nothing`). They must never return dicts or collections of JuMP objects. Instead,
instantiate the appropriate container via `add_*_container!` and store all created objects
there.

### Inline expressions when possible
Expression construction should be inlined at the point of use. Only store an expression in a
container when it is intended to be reused across multiple constraints or objective terms.
Avoid creating expression containers solely as intermediate computation steps.

## Architecture & src/ Layout

- **`src/core/`** — foundational types and containers. `OptimizationContainer` (central
  JuMP-backed container), its keys/types/metadata/utils, `DeviceModel`/`NetworkModel`/
  `ServiceModel` parametric wrappers, `Settings`, `ParameterContainer`, dataset
  (`dataset.jl`/`dataset_container.jl`), abstract store interface, initial conditions,
  optimizer stats, time-series parameter types, network reductions, dual processing,
  external evaluation, outputs access layers.
- **`src/common_models/`** — reusable variable/constraint/expression/parameter builders that
  concrete formulations call into (`add_variable.jl`, `add_jump_expressions.jl`,
  `range_constraint.jl`, `duration_constraints.jl`, `rateofchange_constraints.jl`,
  `add_param_container.jl`, `constraint_helpers.jl`, `interfaces.jl`, etc.).
- **`src/objective_function/`** — generic objective-function formulations, one file per value
  curve / term type (`linear_curve.jl`, `quadratic_curve.jl`, `piecewise_linear.jl`,
  `value_curve_cost.jl`, `proportional.jl`, `start_up_shut_down.jl`, lambda/delta PWL
  builders, `offer_curve_types.jl`, `cost_term_helpers.jl`).
- **`src/quadratic_approximations/`** — PWL approximations of x²: incremental, solver-SOS2,
  manual-SOS2, sawtooth, epigraph, DNMDT (`nmdt.jl`/`nmdt_common.jl`), no-op passthrough,
  McCormick cuts, shared breakpoint utils.
- **`src/bilinear_approximations/`** — approximations of bilinear products x·y: McCormick
  envelopes, Bin2, HybS, NMDT, no-op passthrough.
- **`src/operation/`** — `DecisionModel` (single-period) and `EmulationModel` (rolling
  horizon), their output stores, `ProblemTemplate`, serialization, time-series integration,
  numerical-analysis and debugging utils.
- **`src/initial_conditions/`** — initial-condition calculation logic.
- **`src/utils/`** — pure utilities with no domain coupling (JuMP/DataFrames/datetime/file/
  logging/indexing/time-series helpers, formulation validation, pretty-printing).

The repo-wide directory tree is detailed enough to reconstruct from `src/` itself; keep this
section in sync only at the directory/role level when structure changes.

## Main Public API

275 exports in the main module file (`src/InfrastructureOptimizationModels.jl`). Notable
groups:
- **Model types:** `DecisionModel`, `EmulationModel`, `AbstractProblemTemplate`,
  `DeviceModel`, `NetworkModel`, `ServiceModel`, `FixedOutput`.
- **Containers:** `ServicesModelContainer`, `DevicesModelContainer`, `BranchModelContainer`,
  `ParameterContainer`, parameter attribute types (`TimeSeriesAttributes`,
  `VariableValueAttributes`, `CostFunctionAttributes`, etc.).
- **Template setup:** `set_device_model!`, `set_service_model!`, `set_network_model!`,
  `set_hvdc_network_model!`, `get_device_models`, `get_branch_models`, etc.
- **Build/solve hooks:** `init_optimization_container!`, `validate_time_series!`,
  `get_initial_conditions`, `serialize_outputs`, `serialize_optimization_model`.
- **Model-building helpers:** `add_variables!`, `add_to_expression!`, the
  `add_*_to_jump_expression!` family, PWL builders (`add_pwl_variables_delta!`,
  `add_pwl_linking_constraint!`, `add_pwl_sos2_constraint!`, …), cost-term helpers.
- **Output access:** `get_variable_values`, `get_dual_values`, `get_parameter_values`,
  `get_aux_variable_values`, `get_expression_values`, network accessors (`get_PTDF_matrix`,
  `get_duals`, `get_subnetworks`, …).

## Conventions, Invariants, Gotchas

- **Never create container keys directly.** Constraint/variable/aux-variable/parameter keys
  must only be created through `add_constraints_container!`, `add_variables_container!`,
  `add_aux_variables_container!`, `add_parameters_container!`. Do not instantiate keys
  outside these functions. (See also the `add_*!` return-`nothing` rule above.)
- **Prefer IS types over PSY types.** Use IS parent types where possible:
  `PSY.Component` → `IS.InfrastructureSystemsComponent`,
  `PSY.System` → `IS.InfrastructureSystemsContainer`, cost curves via `IS.CostCurve`,
  `IS.LinearCurve`, `IS.UnitSystem`, etc. IOM imports key/formulation abstract types and
  generic accessors from `InfrastructureSystems.Optimization` / `InfrastructureSystems` to
  avoid duplication.
- **Extension-point stubs.** The main module declares empty generic functions
  (`get_base_power`, `get_active_power_limits`, `get_ramp_limits`, `get_start_up`,
  `get_operation_cost`, `has_service`, `validate_time_series!`, …) that downstream packages
  (chiefly POM) add methods to, dispatched on concrete PSY types. Do not add PSY-specific
  methods here.
- **No Project.toml version bump.** Do NOT edit the `version` field in `Project.toml`, even
  for breaking-change work. A local version ahead of the registry breaks cross-repo
  `Pkg.develop`/test resolution for the rest of the stack. Keep IS unbumped (do not push it
  to `4.0.0`), and the same for sibling packages. If a bump reappears in the working tree
  (it has done so spontaneously mid-session), revert it. Release versions are set at publish
  time, not during dev.
- **Plans/specs live in `.claude/`** (e.g. `.claude/plans/`), not in `docs/`. Keep generated
  planning artifacts out of the project source tree. An existing plan lives at
  `.claude/plans/2026-06-09-deep-review-corrections.md`.
- **Vararg specialization caution (durable note).** A prior blanket `args...` →
  `Vararg{Any,N}` conversion was found counterproductive for throw-only fallbacks (compiles
  dead specializations) and for user-facing API boundaries (`read_*` forwarders in
  `optimization_problem_outputs.jl`, the template setters in `problem_template.jl`) where
  plain `args...` gives better arity errors and no runtime benefit. Keep `Vararg{Any,N}` only
  where arity genuinely propagates into container construction (`optimization_container.jl`,
  `jump_utils.jl`).

## Time-Varying MarketBidCost (cross-repo)

Relevant to `src/objective_function/value_curve_cost.jl`, `offer_curve_types.jl`,
`start_up_shut_down.jl`, and parameter-update logic. Summary; see `IS/.claude/claude.md`
→ "Time-Varying Cost Curve Type Hierarchy" for the full IS type tree.

- IS provides parallel static and time-series-backed cost-curve hierarchies sharing abstract
  parents, so `CostCurve`/`FuelCurve` accept either. Forwarding functions
  `IS.is_time_series_backed(cost_curve)` and `IS.get_time_series_key(cost_curve)` work at
  every level (CostCurve → ValueCurve → FunctionData).
- `MarketBidCost` (in PSY) holds each field as either a scalar/struct (static) or a
  `TimeSeriesKey` (time-varying); time-varying data lives in the System's time-series store.
- **IOM orchestrates; POM implements.** `process_market_bid_parameters!()` filters devices,
  checks time-variance, and calls `add_parameters!(...)` — implemented in POM, which
  decomposes each `PiecewiseStepData` into separate slope/breakpoint/at-min parameter arrays
  (`IncrementalPiecewiseLinearSlopeParameter`, `…BreakpointParameter`,
  `IncrementalCostAtMinParameter`, `StartupCostParameter`, `ShutdownCostParameter`;
  decremental variants mirror these).
- `add_pwl_term!()` builds, per time step, the delta/block PWL variables, the linking
  constraint `p = Σ δ_k`, the width bounds, and the cost expression `Σ slope_k·δ_k·dt`,
  routing to **variant** expressions (time-varying, rebuilt each step) or **invariant**
  (static, computed once).
- Direction dispatch via `OfferDirection` (`IncrementalOffer`/`DecrementalOffer`,
  `src/core/definitions.jl`) selects accessors and parameter types.
- **Target refactor (future):** replace bare `TimeSeriesKey` unions on `MarketBidCost` with
  typed `CostCurve{TimeSeriesPiecewiseIncrementalCurve}`, so `is_time_series_backed()` and
  dispatch replace `isa(field, TimeSeriesKey)` runtime branches throughout IOM/POM. IS
  Phases 1–2 (TS function-data and value-curve types) are done; remaining work is PSY field
  retyping, then IOM `value_curve_cost.jl` and POM dispatch simplification.

## Cross-Package Coupling

- **Upstream:** InfrastructureSystems.jl — base types, key/formulation abstract types,
  generic component/time-series accessors, cost-curve types, serialization. Imported, not
  duplicated.
- **Downstream:** PowerSimulations.jl, PowerOperationsModels.jl, PowerSystemsInvestments.jl
  build concrete formulations and supply methods for IOM's extension-point stubs using PSY
  types. Consider downstream impact for any signature/abstract-type change.

## Commands (verified)

All commands use `--project=<env>` per Sienna rules. Run from the repo root.

```sh
# Full test suite (Aqua checks + ReTest-style runner via run_tests())
julia --project=test test/runtests.jl

# Instantiate the test environment if resolution fails
julia --project=test -e 'using Pkg; Pkg.instantiate()'

# Build docs
julia --project=docs docs/make.jl

# Formatter — run after completing each task (not optional)
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

The formatter script self-activates `scripts/formatter` and instantiates JuliaFormatter; it
formats `./src`, `./test`, and `./docs/src` (`.jl` and `.md`). Do not manually revert its
output.

### Testing specifics

- `test/runtests.jl` runs Aqua (`test_all`, `persistent_tasks=false`) then includes
  `InfrastructureOptimizationModelsTests.jl` and calls `run_tests()`.
- **Test files are included by `InfrastructureOptimizationModelsTests.jl`**, which owns the
  `using`/`import`, `const` aliases, and mock infrastructure. Do **not** add `using`,
  `include`, or `const` alias statements at the top of individual `test_*.jl` files.
- **Use mocks over PSY types.** `test/mocks/` provides `MockThermalGen`, `MockRenewableGen`,
  `MockSystem`, mock container/optimizer/services/time-series so tests don't depend on
  PowerSystems concrete types. `test/verify_mocks.jl` validates mocks match real interfaces.
  Shared fixtures live in `test/test_utils/`.
- Run the full suite after changes and report results (do not assume pass). After each Julia
  edit, verify it compiles
  (`julia --project=test -e 'using InfrastructureOptimizationModels'`) before moving on.
