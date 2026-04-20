# InfrastructureOptimizationModels.jl

Library for Optimization modeling in Sienna. This is a utility library that defines useful objects and
routines for managing power optimization models. Julia compat: `^1.10`.

> **General Sienna Programming Practices:** For information on performance requirements, code conventions, documentation practices, and contribution workflows that apply across all Sienna packages, see [Sienna.md](Sienna.md). Always
check this file before making plans, changes or running tests. Review in detail the testing proceedures at the beggining of every session.

> **Maintenance note:** This file documents the repository structure and conventions. Update it
whenever files, directories, or architectural patterns change so it stays accurate.

## Repository Structure

```
Project.toml              # Package manifest (dependencies, compat)
src/
  InfrastructureOptimizationModels.jl   # Main module file (exports & includes)
  core/                   # Foundational types and containers
    definitions.jl            # Constants, enums, type aliases
    optimization_container.jl # Central JuMP-backed optimization container
    optimization_container_keys.jl
    optimization_container_types.jl
    optimization_container_metadata.jl
    dataset.jl / dataset_container.jl  # Output dataset storage
    device_model.jl           # DeviceModel parametric wrapper
    network_model.jl          # NetworkModel parametric wrapper
    service_model.jl          # ServiceModel parametric wrapper
    settings.jl               # Solver/model settings
    model_internal.jl         # Internal model bookkeeping
    model_store_params.jl     # Store parameter definitions
    abstract_model_store.jl   # Abstract store interface
    initial_conditions.jl     # Initial condition types
    parameter_container.jl    # Parameter container type
    operation_model_abstract_types.jl  # Abstract supertypes for models
    optimization_problem_outputs.jl    # Outputs access layer
    optimization_problem_outputs_export.jl
    optimizer_stats.jl        # Solve statistics tracking
    outputs_by_time.jl        # Time-indexed outputs cache
    standard_variables_expressions.jl  # Standard variable/expression names
    time_series_parameter_types.jl     # Time series parameter wrappers
    network_reductions.jl     # Network reduction utilities
  common_models/          # Reusable model-building methods
    add_variable.jl           # Variable creation helpers
    add_auxiliary_variable.jl # Auxiliary variable helpers
    add_jump_expressions.jl   # JuMP expression builders
    add_param_container.jl    # Parameter container builders
    add_constraint_dual.jl    # Constraint & dual helpers
    constraint_helpers.jl     # Generic constraint utilities
    range_constraint.jl       # Min/max range constraints
    duration_constraints.jl   # Min up/down time constraints
    rateofchange_constraints.jl # Ramp rate constraints
    set_expression.jl         # Expression assignment helpers
    get_time_series.jl        # Time series retrieval
    interfaces.jl             # Abstract interface definitions
  initial_conditions/     # Initial condition logic
    add_initial_condition.jl
    calculate_initial_condition.jl
    initialization.jl
  objective_function/     # Objective function formulations (see note below)
    common.jl                 # Shared objective function utilities
    cost_term_helpers.jl      # Generic objective term → JuMP expression builders
    import_export.jl          # Import/export objective handling
    linear_curve.jl           # LinearCurve objective formulation
    quadratic_curve.jl        # QuadraticCurve objective formulation
    piecewise_linear.jl       # PiecewiseLinearCurve → lambda PWL formulation
    proportional.jl           # Proportional objective terms
    value_curve_cost.jl       # ValueCurve → delta PWL formulation
    offer_curve_types.jl      # Parameter/variable/constraint types for offer curves
    start_up_shut_down.jl     # Start-up/shut-down objective terms
  operation/              # Operation model types and workflows
    decision_model.jl         # DecisionModel (single-period optimization)
    decision_model_store.jl   # DecisionModel output store
    emulation_model.jl        # EmulationModel (rolling horizon)
    emulation_model_store.jl  # EmulationModel output store
    operation_model_interface.jl        # Shared model interface methods
    operation_model_serialization.jl    # Serialization/deserialization
    problem_template.jl       # ProblemTemplate (model specification)
    problem_outputs.jl        # Output post-processing
    store_common.jl           # Shared store utilities
    time_series_interface.jl  # Time series integration
    initial_conditions_update_in_memory_store.jl
    model_numerical_analysis_utils.jl
    optimization_debugging.jl # Debug/diagnostic tools
  quadratic_approximations/  # PWL approximation of x² (univariate)
    common.jl                 # Abstract config hierarchy and shared normalization
    pwl_utils.jl              # Shared breakpoint generator, _square helper
    incremental.jl            # Incremental PWL formulation (δ/z variables)
    solver_sos2.jl            # Solver-native SOS2 quadratic approximation
    manual_sos2.jl            # Manual SOS2 (binary adjacency) quadratic approximation
    sawtooth.jl               # Sawtooth relaxation approximation
    epigraph.jl               # Epigraph-based approximation
    nmdt.jl                   # DNMDT quadratic approximation config and dispatch
    nmdt_common.jl            # Shared NMDT discretization utilities
    no_approx.jl              # No-op passthrough (exact x² for NLP solvers)
    pwmcc_cuts.jl             # Piecewise McCormick tightening cuts
  bilinear_approximations/  # Approximation of bilinear products x·y
    mccormick.jl              # McCormick envelopes for bilinear terms
    bin2.jl                   # Bin2 separable bilinear approximation
    hybs.jl                   # HybS hybrid separable approximation
    nmdt.jl                   # NMDT bilinear approximation config and dispatch
    no_approx.jl              # No-op passthrough (exact x·y for NLP solvers)
  utils/                  # General-purpose utilities
    jump_utils.jl             # JuMP helper functions
    dataframes_utils.jl       # DataFrame manipulation
    datetime_utils.jl         # Date/time helpers
    file_utils.jl             # File I/O utilities
    logging.jl                # Logging setup
    indexing.jl               # Index/key utilities
    component_utils.jl        # Component filtering and unit-system conversion helpers (IS-only)
    time_series_utils.jl      # Time series helpers
    generate_valid_formulations.jl # Formulation validation
    print_pt_v2.jl / print_pt_v3.jl   # Pretty-printing
test/
  runtests.jl             # Entry point — calls load_tests.jl
  load_tests.jl           # Discovers and includes test files
  InfrastructureOptimizationModelsTests.jl  # Test module (imports, aliases, includes)
  includes.jl             # Includes mocks and test utilities
  verify_mocks.jl         # Validates mock types match real interfaces
  test_*.jl               # Individual test files (one per feature area)
  mocks/                  # Mock types for testing without PowerSystems
    mock_components.jl        # MockThermalGen, MockRenewableGen, etc.
    mock_system.jl            # MockSystem container
    mock_container.jl         # Mock optimization container
    mock_optimizer.jl         # Mock solver
    mock_services.jl          # Mock service components
    mock_time_series.jl       # Mock time series data
    constructors.jl           # Convenience constructors for mocks
  test_utils/             # Shared test helpers and fixtures
    common_operation_model.jl
    mock_operation_models.jl
    operations_problem_templates.jl
    solver_definitions.jl
    objective_function_helpers.jl
    model_checks.jl
    test_types.jl
    add_market_bid_cost.jl
    mbc_system_utils.jl
    iec_simulation_utils.jl
    run_simulation.jl
  performance/            # Performance benchmarks
    performance_test.jl
docs/                     # Documenter.jl documentation
  make.jl                 # Build script
  Project.toml            # Docs dependencies
  src/
    index.md
    tutorials/
    how_to_guides/
    explanation/
    reference/
scripts/formatter/        # Code formatting (JuliaFormatter)
```

### Key architectural notes

- **`src/core/`** defines foundational types (`OptimizationContainer`, `DeviceModel`,
  `NetworkModel`, `ServiceModel`, `Settings`, etc.) that are used throughout the package.
- **`src/common_models/`** provides reusable constraint/variable/expression builders that
  concrete formulations call into.
- **`src/objective_function/`** defines objective function formulations — generic methods
  for translating IS `ValueCurve` types into JuMP objective terms. Each value curve type
  has its own file. **IOM defines objective functions, not costs.** "Costs" (e.g.,
  production cost, fuel cost, start-up cost) are domain concepts that belong to POM.
  IOM provides the mathematical formulations (delta PWL, lambda PWL, proportional, quadratic)
  that POM routes specific cost types into. Functions in this directory may reference
  PSY cost types in their signatures (for dispatch), but their purpose is building
  JuMP objective expressions, not defining what a "cost" means.
- **`src/quadratic_approximations/`** implements PWL approximation methods for x²:
  SOS2 (solver and manual), sawtooth, epigraph, plus the incremental formulation
  and shared breakpoint utilities.
- **`src/bilinear_approximations/`** implements approximation methods for bilinear
  products x·y: Bin2 separable decomposition, HybS hybrid relaxation, and McCormick
  envelopes.
- **`src/operation/`** implements `DecisionModel` and `EmulationModel` — the two main model
  types — plus serialization, output stores, and the problem template.
- **`src/utils/`** is for pure utility functions with no domain coupling.
- **`test/mocks/`** provides lightweight stand-ins so tests don't depend on PowerSystems
  concrete types.

## Type and Function Conventions

**Prefer IS types over PSY types:** When possible, use InfrastructureSystems parent types:
- `PSY.Component` → `IS.InfrastructureSystemsComponent`
- `PSY.System` → `IS.InfrastructureSystemsContainer`
- Cost curves: `IS.CostCurve`, `IS.LinearCurve`, `IS.UnitSystem`, etc.

## Optimization Container Keys

**Never create container keys directly.** Constraint keys, variable keys, aux_variable keys, parameter
keys, and any other optimization container keys must only be created through the corresponding
`add_constraints_container!`, `add_variables_container!`, `add_aux_variables_container!`, or
`add_parameters_container!` functions. Do not instantiate keys outside of these `add_foo_container`
functions.

## Testing

**Test file structure:** Test files are included by `InfrastructureOptimizationModelsTests.jl`, which
handles imports and mock infrastructure. Don't add `using`, `include`, or `const` alias statements
at the top of individual test files.

**Use mocks over PSY types:** Tests should use mock components (`MockThermalGen`, `MockSystem`, etc.)
rather than PowerSystems types when possible.

## Time-Varying MarketBidCost Architecture (Cross-Repo)

This section documents how time-varying offer curves flow through the Sienna stack.
Relevant for any work on `src/objective_function/value_curve_cost.jl`, `offer_curve_types.jl`,
`start_up_shut_down.jl`, or parameter update logic.

### IS Type Hierarchy for Cost Curves

IS provides two parallel type hierarchies — static (scalar data) and time-series-backed
(data referenced by `TimeSeriesKey`). Both share abstract parents so `CostCurve`/`FuelCurve`
accept either with zero code changes. See `IS/.claude/claude.md` → "Time-Varying Cost Curve
Type Hierarchy" for the full tree and design rationale.

**Key types for MarketBidCost:**

```
# Static path (current)
CostCurve{PiecewiseIncrementalCurve}
  where PiecewiseIncrementalCurve = IncrementalCurve{PiecewiseStepData}

# Time-series path (new — enables typed cost curve fields)
CostCurve{TimeSeriesPiecewiseIncrementalCurve}
  where TimeSeriesPiecewiseIncrementalCurve = TimeSeriesIncrementalCurve{TimeSeriesPiecewiseStepData}
```

**Forwarding functions** work at every level of the hierarchy:
- `IS.is_time_series_backed(cost_curve)` → `true`/`false` (delegates through ValueCurve → FunctionData)
- `IS.get_time_series_key(cost_curve)` → the underlying `TimeSeriesKey`

### MarketBidCost Struct — Current State (PowerSystems.jl)

```julia
mutable struct MarketBidCost <: OfferCurveCost
    no_load_cost          ::Union{TimeSeriesKey, Nothing, Float64}
    start_up              ::Union{TimeSeriesKey, StartUpStages}
    shut_down             ::Union{TimeSeriesKey, Float64}
    incremental_offer_curves   ::Union{Nothing, TimeSeriesKey, CostCurve{PiecewiseIncrementalCurve}}
    decremental_offer_curves   ::Union{Nothing, TimeSeriesKey, CostCurve{PiecewiseIncrementalCurve}}
    incremental_initial_input  ::Union{Nothing, TimeSeriesKey}
    decremental_initial_input  ::Union{Nothing, TimeSeriesKey}
    ancillary_service_offers   ::Vector{Service}
end
```

Each field independently holds either a scalar/struct value (static) or a `TimeSeriesKey`
(time-varying). When time-varying, the actual data (`PiecewiseStepData` elements for curves,
`Float64` for initial_input/no_load) lives in the System's time series store.

### MarketBidCost — Target State (Future Refactor)

The new IS types enable replacing bare `TimeSeriesKey` unions with typed cost curves:

```julia
# BEFORE: incremental_offer_curves ::Union{Nothing, TimeSeriesKey, CostCurve{PiecewiseIncrementalCurve}}
# AFTER:  incremental_offer_curves ::Union{Nothing, CostCurve{PiecewiseIncrementalCurve}, CostCurve{TimeSeriesPiecewiseIncrementalCurve}}
```

**Benefits of the target state:**
- Unions stay at 3 elements (within Julia's union-splitting threshold)
- `is_time_series_backed()` replaces `isa(field, TimeSeriesKey)` checks throughout IOM/POM
- `initial_input` and `input_at_zero` move into the `TimeSeriesIncrementalCurve` struct,
  eliminating the separate `incremental_initial_input` field
- Type dispatch replaces runtime `if` branches for static vs time-varying paths

**Migration path:**
1. ~~Phase 1: `TimeSeriesFunctionData` types in IS~~ (done)
2. ~~Phase 2: `TimeSeriesValueCurve` types + cost aliases in IS~~ (done)
3. Phase 3: Update `MarketBidCost` field types in PSY
4. Phase 4: Update IOM `value_curve_cost.jl` to dispatch on `is_time_series_backed` instead
   of `isa(field, TimeSeriesKey)`. Simplify `_get_pwl_data` and `process_market_bid_parameters!`.
5. Phase 5: Update POM dispatches (`proportional_cost`, `add_parameters!`)

### Data Flow: Storage → Retrieval → Reconstruction (PowerSystems.jl)

**Setting** (`cost_function_timeseries.jl`):
1. `set_variable_cost!(sys, component, ts_data::Deterministic, power_units)` validates
   `eltype(ts_data) <: PiecewiseStepData`, calls `add_time_series!(sys, component, ts_data)`,
   stores the resulting `TimeSeriesKey` on `MarketBidCost.incremental_offer_curves`.
2. `set_incremental_initial_input!`, `set_no_load_cost!` follow the same pattern.

**Getting** (`cost_function_timeseries.jl`):
1. `get_variable_cost(device, cost; start_time, len)` fetches three independent time series:
   - `incremental_offer_curves` → `TimeArray{PiecewiseStepData}`
   - `incremental_initial_input` → `TimeArray{Float64}` (or scalar/nothing)
   - `no_load_cost` → `TimeArray{Float64}` (or scalar/nothing)
2. Broadcasts scalars to match TimeArray timestamps.
3. Zips and calls `_make_market_bid_curve(psd; initial_input=ii, input_at_zero=iaz)` per timestep.
4. Returns `TimeArray{CostCurve{PiecewiseIncrementalCurve}}`.

### Parameter Decomposition (this repo — IOM)

`process_market_bid_parameters!()` in `src/objective_function/value_curve_cost.jl`:
1. Filters devices with `_has_market_bid_cost()`.
2. For each parameter type, checks `_has_parameter_time_series()` (calls `is_time_variant()`
   which tests if the field is a `TimeSeriesKey`).
3. Calls `add_parameters!(container, ParamType, ts_devices, model)` — **extension point
   implemented in PowerOperationsModels (POM)**.

POM decomposes each `PiecewiseStepData` into separate parameter arrays:

| IOM Parameter Type | Stores | Array axes |
|---|---|---|
| `IncrementalPiecewiseLinearSlopeParameter` | y_coords (marginal rates) | `[device, segment, time]` |
| `IncrementalPiecewiseLinearBreakpointParameter` | x_coords (power points) | `[device, point, time]` |
| `IncrementalCostAtMinParameter` | initial_input scalar | `[device, time]` |
| `StartupCostParameter` | start-up cost | `[device, time]` |
| `ShutdownCostParameter` | shut-down cost | `[device, time]` |

Decremental variants mirror the incremental ones.

### Objective Construction (this repo — IOM)

`add_pwl_term!(dir, container, component, ...)` in `src/objective_function/value_curve_cost.jl`:

For each time step `t`:
1. `_get_pwl_data(dir, container, component, t)` →
   - **Time-varying**: reads slopes/breakpoints from ParameterContainers.
   - **Static**: reads `PSY.get_x_coords` / `PSY.get_y_coords` directly from the CostCurve.
2. Creates delta/block variables (`PiecewiseLinearBlockIncrementalOffer`) — one per segment.
3. Adds linking constraint `p = Σ δ_k` and bound constraints `δ_k ≤ width_k`.
4. Builds cost expression `Σ (slope_k × δ_k × dt)`.
5. Routes to `add_to_objective_variant_expression!` (time-varying) or
   `add_to_objective_invariant_expression!` (static).

### Direction Dispatch

`src/core/definitions.jl` defines `OfferDirection` with `IncrementalOffer` / `DecrementalOffer`.
`value_curve_cost.jl` maps directions to accessors and parameter types:
- `get_output_offer_curves(cost)` → `incremental_offer_curves`
- `get_input_offer_curves(cost)` → `decremental_offer_curves`
- `_slope_param(::IncrementalOffer)` → `IncrementalPiecewiseLinearSlopeParameter`
- `_breakpoint_param(::IncrementalOffer)` → `IncrementalPiecewiseLinearBreakpointParameter`

### Key Architectural Observations

1. **Three-way split**: Offer curve, initial_input, and no_load_cost are stored as separate
   time series with independent `TimeSeriesKey`s. The new `TimeSeriesIncrementalCurve`
   unifies offer curve + initial_input + input_at_zero into a single typed object.
2. **Two-level decomposition**: PSY stores `PiecewiseStepData` objects; IOM further splits
   into separate slope and breakpoint parameter arrays for JuMP.
3. **Static/dynamic is per-field**: Each MarketBidCost field independently decides whether
   it's a scalar or `TimeSeriesKey`. Mixed cases must be handled.
4. **Extension point pattern**: IOM orchestrates (`process_market_bid_parameters!`,
   `_get_pwl_data`, `add_pwl_term!`); POM implements `add_parameters!` and
   `proportional_cost` dispatches.
5. **Variant vs invariant objective**: Time-varying costs → "variant" expressions rebuilt
   each simulation step; static costs → "invariant" expressions computed once.
6. **`is_time_series_backed` replaces `isa(_, TimeSeriesKey)`**: The new forwarding function
   propagates through `CostCurve` → `ValueCurve` → `FunctionData`, making static-vs-TS
   checks uniform across the stack.
