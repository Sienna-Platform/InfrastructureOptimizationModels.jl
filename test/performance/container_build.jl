"""
Container build-time benchmark.

Measures the cost of allocating the two most common optimization-container shapes:

  * dense  `(names::Vector{String}, time_steps::UnitRange{Int})`
  * sparse `(names::Vector{String}, segments::UnitRange{Int}, time_steps::UnitRange{Int})`

These flow through `container_spec` / `sparse_container_spec` (and the `add_*_container!`
wrappers), which have concretely-typed specializations for exactly these shapes. Run against
`main` (generic `Vararg` baseline) and the PR branch (specialized) by
`.github/workflows/performance_test.yml`; the PR comment shows the delta.

Timing uses plain `@elapsed` / `@allocated` (no BenchmarkTools dependency): each case is
warmed up once, then run over several batches of many iterations, reporting the min ns/call.

Usage:
    julia --project=test test/performance/container_build.jl
"""

using InfrastructureOptimizationModels
using InfrastructureSystems
using JuMP
using Dates
using Printf

const IOM = InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization

# ─── Test mocks (mirror scripts/units_dispatch_profile.jl) ────────────────────
const TEST_DIR = joinpath(@__DIR__, "..")
include(joinpath(TEST_DIR, "mocks/mock_optimizer.jl"))
include(joinpath(TEST_DIR, "mocks/mock_system.jl"))
include(joinpath(TEST_DIR, "mocks/mock_components.jl"))
include(joinpath(TEST_DIR, "mocks/mock_time_series.jl"))
include(joinpath(TEST_DIR, "mocks/mock_container.jl"))
include(joinpath(TEST_DIR, "test_utils/test_types.jl"))

const RULE_WIDTH = 78

# ─── Timing helper ────────────────────────────────────────────────────────────

"""
    bench(f; iters, batches) -> (ns_per_call, bytes_per_call)

Warm up `f` once, then time `batches` batches of `iters` calls each and return the min
per-call time (ns) plus the allocations of a single call (bytes).
"""
function bench(f; iters::Int = 2000, batches::Int = 5)
    f()  # warm up / compile
    best = Inf
    for _ in 1:batches
        t = @elapsed for _ in 1:iters
            f()
        end
        best = min(best, t / iters)
    end
    bytes = @allocated f()
    return best * 1e9, bytes
end

make_names(n::Int) = ["gen_$(i)" for i in 1:n]

# ─── Table 1: leaf container specs ────────────────────────────────────────────

function run_leaf_specs(; sizes::Vector{Int} = [50, 500, 5000])
    time_steps = 1:24
    segments = 1:5
    println("Leaf container specs  (time_steps = 1:24, segments = 1:5)")
    println("-"^RULE_WIDTH)
    @printf("%-28s %8s %14s %14s\n", "shape", "N_names", "ns/call", "bytes/call")
    println("-"^RULE_WIDTH)
    for n in sizes
        names = make_names(n)
        cases = [
            (
                "dense VariableRef",
                () -> IOM.container_spec(JuMP.VariableRef, names, time_steps),
            ),
            ("dense Float64", () -> IOM.container_spec(Float64, names, time_steps)),
            (
                "sparse 2-axis",
                () -> IOM.sparse_container_spec(JuMP.VariableRef, names, time_steps),
            ),
            (
                "sparse 3-axis",
                () -> IOM.sparse_container_spec(
                    JuMP.VariableRef,
                    names,
                    segments,
                    time_steps,
                ),
            ),
        ]
        for (label, f) in cases
            ns, bytes = bench(f)
            @printf("%-28s %8d %14.1f %14d\n", label, n, ns, bytes)
        end
    end
    println("-"^RULE_WIDTH)
    println()
end

# ─── Table 2: end-to-end add_*_container! wrappers ────────────────────────────

function make_container()
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(24),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, 1:24)
    return container
end

function run_wrappers(; sizes::Vector{Int} = [50, 500, 5000])
    container = make_container()
    time_steps = IOM.get_time_steps(container)
    segments = 1:5
    println("End-to-end add_*_container!  (time_steps = 1:24, segments = 1:5)")
    println("-"^RULE_WIDTH)
    @printf("%-28s %8s %14s %14s\n", "wrapper", "N_names", "ns/call", "bytes/call")
    println("-"^RULE_WIDTH)
    for n in sizes
        names = make_names(n)

        # Reset the target dict between calls so `_assign_container!` doesn't reject the
        # repeated key; isolates the build cost from JuMP model teardown.
        dense =
            () -> begin
                empty!(IOM.get_variables(container))
                IOM.add_variable_container!(
                    container,
                    TestVariableType,
                    MockThermalGen,
                    names,
                    time_steps,
                )
            end
        sparse =
            () -> begin
                empty!(container.constraints)
                IOM.add_constraints_container!(
                    container,
                    MockConstraint,
                    MockThermalGen,
                    names,
                    segments,
                    time_steps;
                    sparse = true,
                )
            end

        for (label, f) in
            [("add_variable (dense)", dense), ("add_constraints (sparse)", sparse)]
            ns, bytes = bench(f)
            @printf("%-28s %8d %14.1f %14d\n", label, n, ns, bytes)
        end
    end
    println("-"^RULE_WIDTH)
    println()
end

# ─── Entry point ──────────────────────────────────────────────────────────────

function run_benchmark()
    println("="^RULE_WIDTH)
    println("Container build-time benchmark")
    println("="^RULE_WIDTH)
    run_leaf_specs()
    run_wrappers()
    println("="^RULE_WIDTH)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        # Warm up compilation on a tiny workload before the reported run.
        redirect_stdout(devnull) do
            run_leaf_specs(; sizes = [2])
            run_wrappers(; sizes = [2])
        end
        run_benchmark()
    catch e
        @error "Benchmark failed" exception = (e, catch_backtrace())
    finally
        flush(stdout)
        flush(stderr)
    end
end
