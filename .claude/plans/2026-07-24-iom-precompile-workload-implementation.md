# IOM Precompilation Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks are strictly sequential — Task 1 must run on the unmodified tree.

**Goal:** Ship a PrecompileTools.jl `@compile_workload` in IOM that cuts downstream first-use latency of the container machinery by ~80%, using internal mock types only.

**Architecture:** A new `src/precompile_workload.jl` defines four private `_Workload*` mock types and one plain function `_run_precompile_workload()` that exercises the hot path: `Settings` → `OptimizationContainer` → variable/expression/constraint containers → JuMP fill → objective assembly. `@compile_workload` calls that function at precompile time; ~86% of the compile cost is key-type-independent, so caching it against mock types removes it from every downstream consumer's first use.

**Tech Stack:** Julia ^1.10, PrecompileTools.jl 1 (already in the dependency closure via JuMP — zero added install weight), JuMP, InfrastructureSystems (IS4 branch).

**Design rationale, measured motivation, risks, and the POM division of labor live in the companion design doc `PLAN-IOM-precompilation-workload.md` (same directory). The validated prototype of the workload body is `container_workload()` in `.claude/plans/iom_ttfx.jl` (with a typeset-parameterized variant in `.claude/plans/iom_shared_compile.jl`); Task 4's implementation mirrors it. This plan is the executable version; the design doc is background reading, not a task source.**

## Global Constraints

- **NEVER run `git commit` or `git push`.** Leave all changes unstaged; register new files with `git add -N <file>` so they appear in `git diff`.
- **Do NOT change `version` in `Project.toml`** (currently `0.1.0`). If it changes spontaneously, revert it.
- All Julia commands use `julia --project=test` from the repo root (`/Users/jdlara/cache/InfrastructureOptimizationModels.jl`). Never bare `julia` or `--project=.`.
- After each Julia source edit, verify compilation with `julia --project=test -e 'using InfrastructureOptimizationModels'` before moving on.
- Workload types are `_Workload*`-prefixed, defined in `src`, never exported, no PSY vocabulary.
- `include("precompile_workload.jl")` must be the **last** statement before the module's closing `end`.
- `@compile_workload` stays at file top level — never wrapped in a closure (`do`-block); that defeats PrecompileTools.
- The workload is deterministic and hermetic: no `rand()`, no clock reads, no network, no file I/O.
- Style: no ternary operators; no dot field access (use getters); explicit `function … end` with explicit `return` for non-trivial bodies; assignment form only for one-liners; terse or no comments.
- Test files contain no `using`, `include`, or `const` statements — `test/InfrastructureOptimizationModelsTests.jl` owns those (`IOM` and `IS` aliases are already defined there).
- Runs longer than ~2 minutes (full test suite, `Pkg.precompile` on a cold cache) must be executed at the controller level with `Bash run_in_background`, never awaited synchronously inside a subagent.

---

### Task 1: Baseline TTFX measurement (unmodified tree)

Must run BEFORE any code change. Produces the "before" numbers that Task 5 compares against.

**Files:**
- Create: `.claude/plans/iom-precompile-measurements.md`
- Read-only use: `.claude/plans/iom_ttfx.jl`, `.claude/plans/iom_shared_compile.jl` (already present, untracked)

**Interfaces:**
- Produces: a `## Baseline` section in the measurements file with `LOAD_SECONDS`, `FIRST_CALL_SECONDS`, `SECOND_CALL_SECONDS`, `TOTAL_TTFX_SECONDS`, `FIRST_TYPESET_A`, `SECOND_TYPESET_B`, `REPEAT_TYPESET_A`, precompile wall time, and pkgimage size. Task 5 reads this section.

- [ ] **Step 1: Confirm clean tree**

Run: `git -C /Users/jdlara/cache/InfrastructureOptimizationModels.jl status --porcelain -- src Project.toml test`
Expected: no output (the `.claude/plans/` untracked files are excluded by the pathspec). If `src/`, `Project.toml`, or `test/` show modifications, STOP and report — the baseline would be invalid.

- [ ] **Step 2: Instantiate and precompile the test environment (timed)**

Run (background if it exceeds ~2 min):
```sh
cd /Users/jdlara/cache/InfrastructureOptimizationModels.jl
julia --project=test -e 'using Pkg; Pkg.instantiate(); @time Pkg.precompile()'
```
Expected: completes without error. Record the `@time` wall seconds as `PRECOMPILE_SECONDS_BASELINE`.

- [ ] **Step 3: Run the TTFX script in a fresh process**

Run: `julia --project=test .claude/plans/iom_ttfx.jl`
Expected output lines: `LOAD_SECONDS=…`, `FIRST_CALL_SECONDS=…`, `SECOND_CALL_SECONDS=…`, `TOTAL_TTFX_SECONDS=…`. On the reference container the baseline `FIRST_CALL_SECONDS` was ~6; expect the same order of magnitude.

- [ ] **Step 4: Run the shared-compile script in a fresh process**

Run: `julia --project=test .claude/plans/iom_shared_compile.jl`
Expected output lines: `FIRST_TYPESET_A=…`, `SECOND_TYPESET_B=…`, `REPEAT_TYPESET_A=…`.

- [ ] **Step 5: Record pkgimage size**

Run: `ls -la ~/.julia/compiled/v$(julia --version | grep -oE '[0-9]+\.[0-9]+' | head -1)/InfrastructureOptimizationModels/`
Record the `.so` (or `.dylib` on macOS) file sizes.

- [ ] **Step 6: Write the baseline section**

Create `.claude/plans/iom-precompile-measurements.md`:
```markdown
# IOM precompile workload — measurements

Machine: <hostname>, Julia <version>, macOS.

## Baseline (unmodified tree, commit <git rev-parse --short HEAD>)

| metric | value |
|---|---|
| Pkg.precompile wall (s) | <PRECOMPILE_SECONDS_BASELINE> |
| LOAD_SECONDS | <…> |
| FIRST_CALL_SECONDS | <…> |
| SECOND_CALL_SECONDS | <…> |
| TOTAL_TTFX_SECONDS | <…> |
| FIRST_TYPESET_A | <…> |
| SECOND_TYPESET_B | <…> |
| REPEAT_TYPESET_A | <…> |
| pkgimage size | <…> |
```
Fill every `<…>` with the recorded numbers. No placeholders may remain.

---

### Task 2: Add the PrecompileTools dependency

**Files:**
- Modify: `Project.toml` (`[deps]` and `[compat]` sections)
- Modify: `src/InfrastructureOptimizationModels.jl` (imports block, near line 117)

**Interfaces:**
- Produces: `PrecompileTools` importable inside the IOM module. Task 4's workload file calls `PrecompileTools.@setup_workload` / `PrecompileTools.@compile_workload`.

- [ ] **Step 1: Add the dep to `Project.toml`**

In `[deps]`, insert between the `MathOptInterface` and `PrettyTables` lines (alphabetical order):
```toml
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
```

In `[compat]`, insert immediately after the `MathOptInterface = "1"` line (alphabetical order):
```toml
PrecompileTools = "1"
```

Do not touch any other line. In particular `version = "0.1.0"` stays.

- [ ] **Step 2: Add the import to the main module**

In `src/InfrastructureOptimizationModels.jl`, directly after the line `import TimerOutputs` (~line 117), add:
```julia
import PrecompileTools
```

- [ ] **Step 3: Resolve and compile-check**

Run:
```sh
julia --project=test -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); using InfrastructureOptimizationModels'
```
Expected: resolves (PrecompileTools is already in the Manifest closure via JuMP), package loads without error.

- [ ] **Step 4: Verify diff is minimal**

Run: `git diff Project.toml src/InfrastructureOptimizationModels.jl`
Expected: exactly the two `PrecompileTools` lines in `Project.toml` and one `import PrecompileTools` line in the module. If `Manifest.toml` changed, that is acceptable; if `version` changed, revert it.

---

### Task 3: Failing runtime test for the workload function

**Files:**
- Create: `test/test_precompile_workload.jl`
- Modify: `test/InfrastructureOptimizationModelsTests.jl` (inside `run_tests()`, "Lightweight Tests" testset, after the `test_settings.jl` include, ~line 117)

**Interfaces:**
- Consumes: the `IOM` const alias defined at the top of `test/InfrastructureOptimizationModelsTests.jl`.
- Produces: a red test that Task 4 turns green. Test name: `"precompile workload"`. It calls `IOM._run_precompile_workload()` — the exact function name Task 4 must define.

- [ ] **Step 1: Write the test file**

Create `test/test_precompile_workload.jl` with exactly:
```julia
@testset "precompile workload" begin
    @test isnothing(IOM._run_precompile_workload())
end
```
No `using`, no `include`, no `const` — the test module owns those.

- [ ] **Step 2: Register the include**

In `test/InfrastructureOptimizationModelsTests.jl`, inside `run_tests()`, find the line:
```julia
                    include(joinpath(TEST_DIR, "test_settings.jl"))
```
and add directly below it (same indentation):
```julia
                    include(joinpath(TEST_DIR, "test_precompile_workload.jl"))
```

- [ ] **Step 3: Verify the test fails for the right reason**

Run:
```sh
julia --project=test -e 'using InfrastructureOptimizationModels; InfrastructureOptimizationModels._run_precompile_workload()'
```
Expected: `UndefVarError: _run_precompile_workload not defined` (exit nonzero). Any other error means Task 2 broke something — stop and report.

- [ ] **Step 4: Register the new file with git**

Run: `git add -N test/test_precompile_workload.jl`

---

### Task 4: Implement the precompile workload

**Files:**
- Create: `src/precompile_workload.jl`
- Modify: `src/InfrastructureOptimizationModels.jl` (add include as the LAST statement before the closing `end`)

**Interfaces:**
- Consumes (all already defined in IOM/IS; verified against source):
  - `Settings(sys; horizon::Dates.Period, resolution::Dates.Period, time_series_cache_size::Int)` — `src/core/settings.jl:63`
  - `OptimizationContainer(sys, settings, ::Union{Nothing, JuMP.Model}, ::Type{T})` — `src/core/optimization_container.jl:98`
  - `set_time_steps!(container, ::UnitRange{Int64})` — `src/core/optimization_container.jl:206`
  - `get_jump_model(container)`, `get_time_steps(container)`
  - `add_variable_container!(container, T, U, axs...)` — `src/core/optimization_container.jl:708`
  - `add_expression_container!` — `:1079`, `add_constraints_container!` — `:826`
  - `add_to_objective_invariant_expression!` — `:1248`
  - `get_objective_expression(container)` → `ObjectiveFunction` (`:188`); `get_objective_expression(::ObjectiveFunction)` → JuMP expression (`:32`)
  - Extension stub `get_base_power` (declared `function get_base_power end` in the main module); `stores_time_series_in_memory` (imported from IS into the IOM namespace — extending it by bare name on a workload-owned type is legal, not piracy)
  - `IS`, `MOI`, `Dates`, `JuMP` aliases/imports already exist in the module.
- Produces: `IOM._run_precompile_workload()` returning `nothing` (Task 3's test), and the `@compile_workload` block that runs it at precompile time.

- [ ] **Step 1: Create `src/precompile_workload.jl`**

Exact content:
```julia
# Precompilation workload. Exercises the key-type-independent container layers
# (Settings/container init, JuMP fill of DenseAxisArray containers, objective
# assembly) against private mock types so downstream consumers skip that compile.
# Structs sit at top level because @setup_workload lowers into a `let` block.

mutable struct _WorkloadSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
end

get_base_power(sys::_WorkloadSystem) = sys.base_power
stores_time_series_in_memory(::_WorkloadSystem) = true

struct _WorkloadComponent <: IS.InfrastructureSystemsComponent end
struct _WorkloadVariable <: VariableType end
struct _WorkloadConstraint <: ConstraintType end
struct _WorkloadExpression <: ExpressionType end

function _run_precompile_workload()
    sys = _WorkloadSystem(100.0)
    settings = Settings(
        sys;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = OptimizationContainer(sys, settings, nothing, IS.Deterministic)
    set_time_steps!(container, 1:24)
    jump_model = get_jump_model(container)
    names = ["c1", "c2", "c3"]
    time_steps = get_time_steps(container)
    variables = add_variable_container!(
        container,
        _WorkloadVariable,
        _WorkloadComponent,
        names,
        time_steps,
    )
    for name in names, t in time_steps
        variables[name, t] = JuMP.@variable(
            jump_model,
            base_name = "V_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = 10.0
        )
    end
    expressions = add_expression_container!(
        container,
        _WorkloadExpression,
        _WorkloadComponent,
        names,
        time_steps,
    )
    for name in names, t in time_steps
        expressions[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(expressions[name, t], variables[name, t])
    end
    constraints = add_constraints_container!(
        container,
        _WorkloadConstraint,
        _WorkloadComponent,
        names,
        time_steps,
    )
    for name in names, t in time_steps
        constraints[name, t] = JuMP.@constraint(jump_model, expressions[name, t] <= 5.0)
    end
    add_to_objective_invariant_expression!(container, 2.0 * variables["c1", 1])
    objective_function = get_objective_expression(container)
    objective = get_objective_expression(objective_function)
    JuMP.@objective(jump_model, MOI.MIN_SENSE, objective)
    return
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        _run_precompile_workload()
    end
end
```

This mirrors the validated prototype `container_workload()` in `.claude/plans/iom_ttfx.jl`, with three deliberate deviations: `_Workload*` naming, no dot field access (getter chain for the objective), and bare `return` instead of returning the container. If a call errors, compare against the prototype before changing signatures.

Notes locked by design (do not "improve"):
- `_run_precompile_workload` ends with bare `return` (repo rule: builder-style functions return `nothing`).
- No dot field access — objective read via the two-level `get_objective_expression` chain.
- Call sites are concretely typed on `_Workload*` types; keep it that way (prevents invalidation when POM adds methods to IOM stubs).
- Nothing here is exported and nothing gets a docstring.

- [ ] **Step 2: Add the include as the module's last statement**

In `src/InfrastructureOptimizationModels.jl`, find the final include:
```julia
include("utils/datetime_utils.jl")
```
and add directly below it (before the module's closing `end`):
```julia
# Must remain the LAST include: bindings referenced by the workload resolve at
# precompile execution time, so anything defined or imported after this line
# would be invisible to it and fail only during Pkg.precompile.
include("precompile_workload.jl")
```

- [ ] **Step 3: Compile-check (this also runs the workload at precompile time)**

Run: `julia --project=test -e 'using InfrastructureOptimizationModels'`
Expected: loads clean. Precompilation takes several seconds longer than before — that is the workload executing. Any `UndefVarError`/`MethodError` during precompile means a binding is defined after the include or a signature drifted; fix the workload, not the include order rule.

- [ ] **Step 4: Verify the runtime test passes**

Run:
```sh
julia --project=test -e 'using Test; using InfrastructureOptimizationModels; const IOM = InfrastructureOptimizationModels; @test isnothing(IOM._run_precompile_workload())'
```
Expected: `Test Passed`.

- [ ] **Step 5: Register the new file with git**

Run: `git add -N src/precompile_workload.jl`

---

### Task 5: Precompile validation, escape hatch, and after-measurement

**Files:**
- Modify: `.claude/plans/iom-precompile-measurements.md` (append `## After` section)
- Create then DELETE: `test/LocalPreferences.toml` (must not survive the task)

**Interfaces:**
- Consumes: the `## Baseline` section from Task 1.
- Produces: an `## After` section with the same metrics plus the acceptance verdict.

- [ ] **Step 1: Timed precompile**

Run:
```sh
julia --project=test -e 'using Pkg; @time Pkg.precompile()'
```
Expected: clean; wall time up to ~15 s over `PRECOMPILE_SECONDS_BASELINE` is acceptable (prototype measured +7 s).

- [ ] **Step 2: Verify the escape hatch, then remove it**

Create `test/LocalPreferences.toml`:
```toml
[InfrastructureOptimizationModels]
precompile_workload = false
```
Run: `julia --project=test -e 'using Pkg; Pkg.precompile(); using InfrastructureOptimizationModels'`
Expected: precompiles and loads clean (workload skipped; a one-time recompile is normal).
Then delete the file: `rm test/LocalPreferences.toml` and recompile once more:
`julia --project=test -e 'using Pkg; Pkg.precompile()'`
Confirm deletion: `git status --porcelain test/` shows no `LocalPreferences.toml`.

- [ ] **Step 3: Rerun both measurement scripts (fresh processes)**

Run: `julia --project=test .claude/plans/iom_ttfx.jl`
Run: `julia --project=test .claude/plans/iom_shared_compile.jl`

- [ ] **Step 4: Record pkgimage size**

Same command as Task 1 Step 5. Expected roughly double the baseline (prototype: 5.3 → 10.1 MB).

- [ ] **Step 5: Append the After section and the verdict**

Append to `.claude/plans/iom-precompile-measurements.md`:
```markdown
## After (workload in tree)

| metric | baseline | after | delta |
|---|---|---|---|
| Pkg.precompile wall (s) | <…> | <…> | <…> |
| LOAD_SECONDS | <…> | <…> | <…> |
| FIRST_CALL_SECONDS | <…> | <…> | <…> |
| SECOND_CALL_SECONDS | <…> | <…> | <…> |
| FIRST_TYPESET_A | <…> | <…> | <…> |
| SECOND_TYPESET_B | <…> | <…> | <…> |
| pkgimage size | <…> | <…> | <…> |

Acceptance: PASS/FAIL per criteria below.
```

Acceptance criteria (all must hold):
1. `FIRST_CALL_SECONDS` (iom_ttfx) and `FIRST_TYPESET_A` (iom_shared_compile) each drop ≥50% vs baseline. The script types are foreign to the workload's `_Workload*` types, so this is the honest downstream number (prototype: −79%).
2. `SECOND_CALL_SECONDS` stays at ~milliseconds (no runtime regression from locked-in specializations).
3. Precompile wall-time increase ≤15 s.

If criterion 1 fails, the workload compiled but did not cache the shared layers — most likely cause is an abstractly-typed call site in the workload or the include not being last. Report the numbers; do NOT start adding Phase-2 coverage to chase the number.

---

### Task 6: Full suite, formatter, final review

**Files:**
- No new files. Formatter may reflow `src/precompile_workload.jl` and `test/test_precompile_workload.jl`.

- [ ] **Step 1: Run the formatter**

Run: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`
Expected: completes; do not revert its output.

- [ ] **Step 2: Compile-check after formatting**

Run: `julia --project=test -e 'using InfrastructureOptimizationModels'`
Expected: loads clean.

- [ ] **Step 3: Full test suite (controller-level, backgrounded — ~15 min; never await inside a subagent)**

Run: `julia --project=test test/runtests.jl` with `run_in_background` and monitor to completion.
Expected: Aqua checks pass (the new dep has a compat entry, so `stale_deps`/`deps_compat` stay green) and all unit tests pass, including the new `"precompile workload"` testset. Report the exact pass/fail counts — do not summarize as "tests pass" without the numbers.

- [ ] **Step 4: Final diff review**

Run: `git status --porcelain && git diff --stat`
Expected state, exactly:
- Modified, unstaged: `Project.toml`, `src/InfrastructureOptimizationModels.jl`, `test/InfrastructureOptimizationModelsTests.jl` (and possibly `Manifest.toml`)
- New, intent-to-add: `src/precompile_workload.jl`, `test/test_precompile_workload.jl`
- Untracked plan artifacts under `.claude/plans/` only
- `Project.toml` `version` still `0.1.0`
- Nothing staged, nothing committed

---

## Explicitly out of scope (do not implement)

- **Phase-2 coverage increments** (parameter containers, objective-function curves, generic `add_variables!`, dataset/store round trip, range/duration helpers). Each requires a measure-and-judge cycle against the precompile budget — a human decision per increment. The candidate list lives in the design doc.
- **PSY types, PNM matrices, `DecisionModel`/`EmulationModel` end-to-end builds** — IOM cannot run them (PR #104 moved the problem taxonomy to POM) and mock-based full builds would recreate what #104 removed.
- **Promoting `test/mocks/` into `src`** — considered and rejected in the design doc.
- **Versioning the measurement scripts under `test/performance/`** — team decision, not part of this change.
- **Any commit, push, or version bump.**
