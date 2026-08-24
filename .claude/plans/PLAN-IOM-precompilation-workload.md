# Precompilation workload for InfrastructureOptimizationModels.jl — plan + validated prototype

Date: 2026-07-24 (overnight autonomous session).
Companion: `PLAN-POM-precompilation-workload.md` — read both; the split of responsibilities
between the two workloads is deliberate.
Suggested landing spot for this document: `IOM/.claude/plans/2026-07-24-precompilation-workload.md`.

Status: **a working prototype is in the `claude/precompilation-workload-pom-iom-ilnpwh`
working tree** (unstaged, nothing committed), and every number below was measured on it in
this container. What remains: review, Phase-2 scope tuning, tests/CI guards, formatter pass.

## Goal

Cut time-to-first-use (TTFX) of the IOM container machinery by shipping a
PrecompileTools.jl `@compile_workload` that exercises the hot construction paths using
**internal mock types** — no PowerSystems, no PowerSystemCaseBuilder, no file/HDF5
deserialization anywhere near the workload. (Deserializing a system during precompilation
would spend more than it saves; PSB isn't even a dependency, and a precompile workload can
only use the package's own deps.)

## Why this shape — measured motivation

Environment for all numbers: fresh depot, Julia 1.12.6, 4-core Linux container. Dev
machines will be faster in absolute terms; the ratios are what matter.

Representative first-use of the container machinery (Settings + `OptimizationContainer` +
variable/expression/constraint containers + JuMP fill + objective assembly; 3 components ×
24 time steps):

| measurement | seconds |
|---|---|
| first call, cold (baseline, no workload) | **5.9–6.3** |
| same call again, same process | 0.001 |
| same call with a *different* set of key types, same process | **0.9** |

Two facts fall out:

1. It is ~100% compile latency (runtime is ~1 ms).
2. **~86% of the compile cost is shared across key types.** The outer layers
   (`add_variable_container!` …) specialize on `VariableType`/component type parameters,
   but the inner layers — JuMP model mutation, `DenseAxisArray{VariableRef,2}` with
   `(Vector{String}, UnitRange)` axes, Settings/container init, objective assembly — are
   the same concrete code for every consumer. Precompiling them against *any* concrete key
   types (internal mocks) removes them from every downstream consumer's first-use bill.

### Prototype results (measured on the in-tree prototype)

| metric | before | after | delta |
|---|---|---|---|
| foreign-key-type first use (the honest downstream number) | 6.3 s | **1.36 s** | **−79%** |
| `using InfrastructureOptimizationModels` | 4.2 s | 4.2 s | none |
| `Pkg.precompile()` of IOM alone | ~17–19 s | ~24 s | **+7 s one-time** |
| pkgimage `.so` size | 5.3 MB | 10.1 MB | +4.8 MB |
| escape hatch `precompile_workload = false` | — | verified on 1.12 | workload skipped |

The residual ~1 s is per-key-type outer-layer specialization that only a downstream
package's own workload can remove — which is exactly what POM's workload does for POM/PSY
types (see companion plan).

## Division of labor with POM — do not over-build IOM's workload

Measured hard fact from the POM side: **POM's own workload alone delivered −91% total TTFX
with a workload-less IOM underneath**, because `@compile_workload` caches non-owned callees
(IOM, JuMP, PSY methods) into the *triggering* package's image. Therefore:

- IOM's workload is **not** needed for POM users' TTFX. It serves IOM-standalone consumers
  (PowerSystemsInvestments, future domains), IOM's own dev/test loop, and it shrinks the
  duplicated shared-layer compile across many consumers.
- Keep it **small and generic**: target the shared ~86%, with a precompile budget of
  ~5–10 s. POM-specific coverage belongs in POM's workload.

## What the prototype changed (3 files, review via `git diff` after `git add -N`)

1. `Project.toml` — added `PrecompileTools` to `[deps]` and `[compat] PrecompileTools = "1"`.
   PrecompileTools is already in the dependency closure via JuMP → zero extra install
   weight. No other entries touched (per repo rule: no version/compat bumps).
2. `src/InfrastructureOptimizationModels.jl` — `import PrecompileTools` next to the other
   imports; `include("precompile_workload.jl")` as the **last** include of the module.
3. `src/precompile_workload.jl` (new, ~75 lines) — internal mock types + workload:

```julia
mutable struct _WorkloadSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
end
get_base_power(s::_WorkloadSystem) = s.base_power
stores_time_series_in_memory(::_WorkloadSystem) = true

struct _WorkloadComponent <: IS.InfrastructureSystemsComponent end
struct _WorkloadVariable <: VariableType end
struct _WorkloadConstraint <: ConstraintType end
struct _WorkloadExpression <: ExpressionType end

function _run_precompile_workload()
    sys = _WorkloadSystem(100.0)
    settings = Settings(sys; horizon = Dates.Hour(24), resolution = Dates.Hour(1),
        time_series_cache_size = 0)
    container = OptimizationContainer(sys, settings, nothing, IS.Deterministic)
    set_time_steps!(container, 1:24)
    jm = get_jump_model(container)
    names = ["c1", "c2", "c3"]
    ts = get_time_steps(container)
    vc = add_variable_container!(container, _WorkloadVariable, _WorkloadComponent, names, ts)
    for n in names, t in ts
        vc[n, t] = JuMP.@variable(jm, base_name = "V_{$(n), $(t)}",
            lower_bound = 0.0, upper_bound = 10.0)
    end
    ec = add_expression_container!(container, _WorkloadExpression, _WorkloadComponent, names, ts)
    for n in names, t in ts
        ec[n, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(ec[n, t], vc[n, t])
    end
    cc = add_constraints_container!(container, _WorkloadConstraint, _WorkloadComponent, names, ts)
    for n in names, t in ts
        cc[n, t] = JuMP.@constraint(jm, ec[n, t] <= 5.0)
    end
    add_to_objective_invariant_expression!(container, 2.0 * vc["c1", 1])
    obj = get_objective_expression(container.objective_function)
    JuMP.@objective(jm, MOI.MIN_SENSE, obj)
    return
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        _run_precompile_workload()
    end
end
```

## Design rules (load-bearing; enforce in review)

- **Workload types live in `src`, `_Workload*`-prefixed, never exported.** Structs cannot
  be defined inside `@setup_workload` (it lowers into a `let`), so they sit at file top
  level. Mock methods extend IOM's own extension stubs (`get_base_power`,
  `stores_time_series_in_memory`) on workload-owned types — no piracy, no PSY vocabulary,
  passes the domain-neutrality litmus.
- **The workload include must be the last statement of the module.** POM's prototype hit a
  real landmine: its main module runs `import IOM: set_status!, …` *after* the include
  block, so a workload placed with the other includes threw `UndefVarError` **only during
  precompile** (bindings resolve at execution; normal load order hides this). IOM's module
  currently ends with includes, but keep the rule explicit so a future bottom-of-file
  import doesn't silently break precompilation.
- **Workload body is a plain function** called from `@compile_workload` — runnable in the
  REPL (`IOM._run_precompile_workload()`), testable, debuggable via
  `PrecompileTools.verbose[] = true; include("src/InfrastructureOptimizationModels.jl")`.
- **`@compile_workload` stays at top level.** Never wrap the macro in a closure
  (`mktempdir() do`, `withenv do`, `with_logger do`) — PrecompileTools documents that this
  defeats it. Closures *inside* the called function are fine.
- **Deterministic, hermetic data.** No `rand()`, no clock reads, no network, no files —
  this workload touches no filesystem at all.
- **Fail loudly.** An error in the workload fails `Pkg.precompile()` for the package.
  That is desired: a silently skipped workload is a silent TTFX regression.

## Phase 2 — coverage tuning (measure every increment before keeping it)

Candidates in descending expected value; keep an increment only if it visibly shrinks the
foreign-type number (protocol below) at ≤ a few seconds of added precompile:

1. **Parameter containers + time-series parameter machinery**
   (`add_parameters_container!`, `ParameterContainer`, `TimeSeriesAttributes`) — on every
   downstream build path.
2. **Objective-function machinery over IS value curves** — proportional, quadratic, PWL
   delta builders (`LinearCurve`, `QuadraticCurve`, `PiecewiseLinearData`), exercising
   `src/objective_function/`.
3. **Generic `add_variables!` path** via a mock device + `AbstractDeviceFormulation` with
   the trait methods (`get_variable_binary`, bounds) — the pattern already exists in
   `test/mocks/mock_components.jl` (`TestDeviceFormulation`).
4. **Dataset/store round trip** (`DecisionModelStore`, dataset containers) — only if
   IOM-standalone consumers need it; POM's solve workload covers it downstream.
5. **Range/duration constraint helpers** from `src/common_models/`.

Out of scope for IOM's workload by design: PSY types, PNM network matrices, and
`DecisionModel`/`EmulationModel` end-to-end builds — IOM alone cannot run them (the problem
taxonomy and `build!` moved to POM in PR #104), and mock-based full builds would re-add a
shadow of what #104 removed.

### Alternative considered: promote `test/mocks/` into `src`

Sharing one mock suite between tests and workload (e.g. `src/testing/`) was considered and
rejected for now: the workload needs 4 one-line types, while the test mock suite is a much
larger surface whose churn would then couple to package precompilation. Revisit only if
Phase-2 item 3 starts duplicating nontrivial trait code.

## Validation already performed on the prototype

- **Full IOM test suite run against the workload-carrying tree: Aqua 10/10, unit tests
  1364/1364 pass** (14m22s wall in the container). The formatter was also run; the tree is
  formatter-clean.
- `Pkg.precompile()` clean; escape hatch verified; measurement tables above reproduced on
  the final formatted tree.

## Guards & CI

1. **Runtime test** `test/test_precompile_workload.jl`:
   `@test isnothing(IOM._run_precompile_workload())` — breakage becomes a red test with a
   stack trace, not just a failed precompile. (Test files are included by
   `InfrastructureOptimizationModelsTests.jl`; no per-file `using`s.)
2. **Aqua** — the new dep has a compat entry; `Aqua.test_all` stays green (workload spawns
   no tasks; `persistent_tasks=false` already set).
3. **Formatter** — run `scripts/formatter` before committing (not yet run on the prototype).
4. **Docs** — nothing exported, no docstring/API-page work.
5. **Dev escape hatch** (verified working on 1.12): while iterating on IOM, drop
   ```toml
   [InfrastructureOptimizationModels]
   precompile_workload = false
   ```
   into `LocalPreferences.toml` beside the active Project.toml (do not commit it). Toggling
   recompiles once. Worth a line in the contributor docs.

## Risks & mitigations

- **Invalidation by downstream method insertion.** POM extends IOM stubs at load time. If
  that invalidated IOM's cached workload code the benefit would vanish. Empirically it does
  not: in the combined configuration (POM sourcing this IOM, both workloads on) downstream
  TTFX stayed at the improved level (see POM plan, "combined configuration"). Root cause:
  the workload's call sites are concretely typed on `_Workload*` types, so new methods on
  other types don't intersect their compiled signatures. Keep the workload free of
  abstractly-typed call sites.
- **Precompile-time creep.** +7 s today; budget ≤ ~15 s. Measure every Phase-2 increment.
- **Workload rot.** Loud failure + the runtime test keep it visible; the measurement
  protocol below catches silent coverage loss (foreign-type number regressing).
- **Locked-in bad specializations** (PrecompileTools #33 class): theoretical; the
  second-call timing in the protocol doubles as the runtime-regression check (~1 ms).

## Measurement protocol (rerun after any workload change)

Two ~40-line scripts used tonight (delivered alongside this plan as `iom_ttfx.jl` and
`iom_shared_compile.jl`; recreate under `test/performance/` if the team wants them
versioned):

- `iom_ttfx.jl` — fresh process: `using` time; first + second call of the container
  exercise.
- `iom_shared_compile.jl` — same exercise across two *different* key-type sets; the
  second-typeset time is the foreign-type number this workload exists to shrink.

Procedure: `Pkg.precompile()` (record wall time) → run each script in a fresh process →
record `LOAD_SECONDS` / `FIRST_*` / `SECOND_*` → compare to the tables above. Image size:
`ls -la ~/.julia/compiled/v<ver>/InfrastructureOptimizationModels/`.

## Rollout

1. Review the 3-file prototype diff + this plan; settle Phase-2 scope.
2. Add the runtime test; formatter; full suite; docs build.
3. Land IOM independently (POM's workload does not depend on it).
4. After POM lands, rerun the combined measurement (POM plan) to reconfirm the
   no-invalidation result.
