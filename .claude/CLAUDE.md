# InfrastructureOptimizationModels.jl — Claude Guide

This file is repo-specific. The shared Sienna rules load from the import below; the `sienna-dev` skill holds the cross-package
context; the `sienna-dev` and `sienna-reviewer` subagents implement and review.

@Sienna.md

## Purpose & Role

IOM is the **optimization-model utility layer** of the psy6 stack — the domain-neutral
abstraction beneath PowerOperationsModels.jl (POM). PowerSimulations (PSI) is the simulation layer above IOM + POM. IOM defines the generic,
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

## The IOM/POM split (PR #104) — do not re-litigate

PR #104 moved the problem taxonomy out of IOM. IOM **removed**: `OperationModel`,
`DecisionProblem`, `EmulationProblem`, the `Generic*`/`Default*` problem types, and the
`Simulation*` stubs. POM now owns the `AbstractPowerOperationProblem <: IOM.AbstractOptimizationProblem`
chain (`POM/src/core/problem_types.jl`); the old `OperationModel` abstract became
`IOM.AbstractOptimizationModel`; `validate_time_series!`/`validate_template` remain here as
stubs on the neutral abstract. **When a symbol is "missing from IOM", check whether #104
moved it to POM before re-adding anything here.**

Domain-neutrality litmus: if a name only makes sense for power systems, it belongs in POM.

Known item to shrink, not extend: model-lifecycle stubs that silently no-op
instead of erroring.

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

- **`src/core/`** — `OptimizationContainer` (the central JuMP-backed container), its keys,
  the `DeviceModel`/`NetworkModel`/`ServiceModel` wrappers, settings, and parameter containers.
- **`src/common_models/`** — reusable variable/constraint/expression/parameter builders that
  concrete formulations call into.
- **`src/objective_function/`** — generic objective-function formulations, one file per value
  curve or term type.
- **`src/quadratic_approximations/`, `src/bilinear_approximations/`** — PWL approximations of
  x² and x·y (SOS2, sawtooth, McCormick, NMDT, and others).
- **`src/operation/`** — `DecisionModel`, `EmulationModel`, their output stores, and `ProblemTemplate`.

## What IOM is

**IOM is the InfrastructureSystems of optimization**: a domain-neutral utility library that
several modeling packages build on — PowerOperationsModels today, PowerSystemsInvestments and
anything else tomorrow. That shapes how to read this code:

- **A function with no caller inside IOM is not dead code.** It is library surface for
  consumers, present or future. Do not propose deleting it, and do not propose moving it into
  POM (or any other consumer) because "POM is the only caller" — a single consumer today is
  not evidence the utility belongs there.
- **Do not reshuffle IOM's or POM's imports and exports.** The export list and the `_*`
  helper surface are deliberate API, not accidents to tidy. Leave them alone unless the task
  is explicitly about the public interface.
- The direction of travel is IOM → consumers. Never move a utility downstream to shrink IOM,
  and never introduce a dependency on a consumer.

## Main Public API

The export list in the main module file (`src/InfrastructureOptimizationModels.jl`) is authoritative.

## Conventions, Invariants, Gotchas

- **Never create container keys directly.** Create keys only inside the `add_*_container!`
  functions in `src/core/optimization_container.jl` and `src/common_models/add_param_container.jl`
  (for example `add_variable_container!`, `add_constraints_container!`). Do not instantiate keys
  outside these functions. (See also the `add_*!` return-`nothing` rule above.)
- **Prefer IS types over PSY types.** Use IS parent types where possible:
  `PSY.Component` → `IS.InfrastructureSystemsComponent`,
  `PSY.System` → `IS.InfrastructureSystemsContainer`, cost curves via `IS.CostCurve`,
  `IS.LinearCurve`, unit markers via `IS.RelativeUnits` (`SU`/`CU`/`NU`, for cost curves only; getters take `u"SU"` — the legacy
  `IS.UnitSystem` enum is gone from the cost-curve API in IS4). IOM imports key/formulation abstract types and
  generic accessors from `InfrastructureSystems.Optimization` / `InfrastructureSystems` to
  avoid duplication.
- **Extension-point stubs.** The main module declares empty generic functions
  (`get_base_power`, `get_active_power_limits`, `get_ramp_limits`, `get_start_up`,
  `get_operation_cost`, `has_service`, `validate_time_series!`, …) that downstream packages
  (chiefly POM) add methods to, dispatched on concrete PSY types. Do not add PSY-specific
  methods here.
- **No Project.toml version bump.** Do not edit the `version` field in `Project.toml`, even
  for breaking-change work. A local version ahead of the registry breaks cross-repo
  `Pkg.develop`/test resolution for the rest of the stack. Keep IS unbumped (do not push it
  to `4.0.0`), and the same for sibling packages. If a bump reappears in the working tree
  (it has done so spontaneously mid-session), revert it. Release versions are set at publish
  time, not during dev.
- **Plans/specs live in `.claude/`** (e.g. `.claude/plans/`), not in `docs/`. Keep generated
  planning artifacts out of the project source tree.
- **Vararg specialization.** Keep `Vararg{Any,N}` only where arity propagates into container
  construction (`optimization_container.jl`, `jump_utils.jl`). Throw-only fallbacks and user-facing
  forwarders (`read_*`, template setters) use plain `args...`, which gives better arity errors.

## Time-Varying MarketBidCost (cross-repo)

Relevant to `src/objective_function/value_curve_cost.jl`, `offer_curve_types.jl`,
`start_up_shut_down.jl`, and parameter-update logic. Summary; see `InfrastructureSystems.jl/.claude/CLAUDE.md`
→ "Time-Varying Cost Curve Type Hierarchy" for the full IS type tree.

- IS provides parallel static and time-series-backed cost-curve hierarchies sharing abstract
  parents, so `CostCurve`/`FuelCurve` accept either. Forwarding functions
  `IS.is_time_series_backed(cost_curve)` and `IS.get_time_series_key(cost_curve)` work at
  every level (CostCurve → ValueCurve → FunctionData).
- `MarketBidCost` (in PSY) holds each field as either a scalar/struct (static) or a
  `TimeSeriesKey` (time-varying); time-varying data lives in the System's time-series store.
- **POM owns `process_market_bid_parameters!`**: it filters devices, checks time-variance,
  and calls `add_parameters!`, which decomposes each `PiecewiseStepData` into slope,
  breakpoint, and at-min parameter arrays (`IncrementalPiecewiseLinearSlopeParameter`,
  `…BreakpointParameter`, `IncrementalCostAtMinParameter`, `StartupCostParameter`,
  `ShutdownCostParameter`; decremental variants mirror these). IOM holds the stubs and the
  generic objective builders.
- `add_pwl_term_delta!` builds, per time step, the delta PWL variables, the linking
  constraint `p = Σ δ_k`, the width bounds, and the cost expression `Σ slope_k·δ_k·dt`,
  routing to **variant** expressions (time-varying, rebuilt each step) or **invariant**
  (static, computed once). POM supplies its static entry point. `add_pwl_term_lambda!`
  builds the lambda form.
- Direction dispatch via `OfferDirection` (`IncrementalOffer`/`DecrementalOffer`,
  `src/core/definitions.jl`) selects accessors and parameter types.

## Cross-Package Coupling

- **Upstream:** InfrastructureSystems.jl (IS4 branch via `[sources]`) — base types,
  key/formulation abstract types, generic component/time-series accessors, cost-curve types,
  serialization. Imported, not duplicated.
- **Downstream:** PowerOperationsModels.jl (primary) and PowerSystemsInvestments.jl build
  concrete formulations and supply methods for IOM's extension-point stubs using PSY types.
  Consider downstream impact for any signature/abstract-type change; run the POM suite after
  IOM edits.
- **The `IOM._*` surface is load-bearing.** POM calls many non-exported
  `IOM._*` helpers (parameter/multiplier setters, PWL machinery). Renaming or changing any
  `_`-prefixed function here can break a consumer even though it is "private", so treat `_*`
  signature changes as breaking.
- **Reduction seam:** IOM keeps only the abstract `AbstractBranchReductionTracker`; the
  concrete `BranchReductionOptimizationTracker` lives in POM. Evaluators: IOM owns
  `AbstractEvaluator`; POM wraps PowerFlows models in `PowerFlowEvaluator`. Keep IOM
  PF-agnostic.
- **Result keys:** outputs encode as `"VariableType__ComponentType"` (`__` delimiter) through
  `IOM.OptimizationProblemOutputs`; read via `read_variable(res, key; table_format =
  TableFormat.WIDE)`. Statuses: `IOM.ModelBuildStatus.BUILT`,
  `IOM.RunStatus.SUCCESSFULLY_FINALIZED`. Renaming any of these breaks every downstream
  results consumer.

## Commands (verified)

All commands use `--project=<env>` per Sienna rules. Run from the repo root.

```sh
# Full test suite (Aqua checks + ReTest-style runner via run_tests())
julia --project=test test/runtests.jl

# Instantiate the test environment if resolution fails
julia --project=test -e 'using Pkg; Pkg.instantiate()'

# Build docs
julia --project=docs docs/make.jl

# Formatter — run after each task
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

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
- Compile-check after each Julia edit: `julia --project=test -e 'using InfrastructureOptimizationModels'`.
