"""
Profile the cost-coefficient conversion path under each of the 3 unit systems
to validate (or refute) the claims in IOM issue #90.

Two questions:

1. **Static**: where does Julia's compiler still emit dynamic dispatch on the
   cost-objective path? Run `JET.@report_opt` on the workload — every entry
   it lists is a real failure of inference. Re-run after each rung of the
   ladder ((4) Union field, (6) parameterized component) to measure delta.

2. **Wall-time**: does the entry-point dispatch actually cost noticeable
   time relative to JuMP variable/constraint construction? Run `@btime` on
   the workload — if dispatch is <1% of total, the architectural rungs are
   premature.

Usage:
    # JET and BenchmarkTools are not in test/Project.toml — add them first:
    julia --project=test -e 'using Pkg; Pkg.add(["JET", "BenchmarkTools"])'
    julia --project=test scripts/units_dispatch_profile.jl

The script uses the test mocks (`MockThermalGen`, `MockSystem`, etc.) so it
runs without a PSY system. Workload size is small by default — bump
`N_COMPONENTS` for more realistic measurement.
"""

using InfrastructureOptimizationModels
using InfrastructureSystems
using JuMP
using Dates
using InteractiveUtils
using BenchmarkTools
using JET

const IOM = InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization

# ---------------------------------------------------------------------------
# Bootstrap test mocks
# ---------------------------------------------------------------------------
const TEST_DIR = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIR, "mocks/mock_optimizer.jl"))
include(joinpath(TEST_DIR, "mocks/mock_system.jl"))
include(joinpath(TEST_DIR, "mocks/mock_components.jl"))
include(joinpath(TEST_DIR, "mocks/mock_time_series.jl"))
include(joinpath(TEST_DIR, "mocks/mock_services.jl"))
include(joinpath(TEST_DIR, "mocks/mock_container.jl"))
include(joinpath(TEST_DIR, "mocks/constructors.jl"))
include(joinpath(TEST_DIR, "test_utils/test_types.jl"))

IOM.objective_function_multiplier(::Type{TestCostVariable}, ::Type{TestFormulation}) = 1.0

# ---------------------------------------------------------------------------
# Workload: build container, add a cost term per (component, unit-system) cell
# ---------------------------------------------------------------------------
const N_COMPONENTS = 50
const TIME_STEPS = 1:24

function make_container(devices)
    sys = MockSystem(100.0)
    settings = IOM.Settings(sys; horizon = Dates.Hour(length(TIME_STEPS)),
        resolution = Dates.Hour(1))
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
    IOM.set_time_steps!(container, TIME_STEPS)
    names = [get_name(d) for d in devices]
    var_container = IOM.add_variable_container!(
        container, TestCostVariable, MockThermalGen, names, TIME_STEPS,
    )
    jm = IOM.get_jump_model(container)
    for d in devices, t in TIME_STEPS
        var_container[get_name(d), t] = JuMP.@variable(jm, base_name = "x_$(get_name(d))_$t")
    end
    return container
end

# Build N components, evenly distributed across the 3 unit systems. The cost
# *curves* are concretely typed (CostCurve{LinearCurve, NaturalUnit/SU/DU}) but
# the abstractly-typed `MockThermalGen` storage means upstream callers see a
# UnionAll. This is the realistic shape of consuming code.
function make_workload()
    devices = MockThermalGen[]
    curves = Any[]  # heterogeneous on U, simulates abstract field upstream
    units = (IS.NaturalUnit(), IS.SystemBaseUnit(), IS.DeviceBaseUnit())
    for i in 1:N_COMPONENTS
        u = units[mod1(i, 3)]
        push!(devices, make_mock_thermal("g$i"; base_power = 50.0 + i))
        push!(curves, IS.CostCurve(IS.LinearCurve(30.0 + i / 10), u))
    end
    return devices, curves
end

function add_all_costs!(container, devices, curves)
    for (d, c) in zip(devices, curves)
        IOM.add_variable_cost_to_objective!(
            container, TestCostVariable, d, c, TestFormulation,
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Run profile
# ---------------------------------------------------------------------------

function main()
    devices, curves = make_workload()

    # Warm up the JIT before measuring.
    let c = make_container(devices)
        add_all_costs!(c, devices, curves)
    end

    println("\n=== JET.@report_opt: dynamic dispatch sites in cost-objective path ===")
    let c = make_container(devices)
        # report_opt walks the call graph and lists every site where Julia's
        # compiler couldn't fully infer types — the actionable list for rung 4/6.
        rep = @report_opt add_all_costs!(c, devices, curves)
        show(stdout, MIME"text/plain"(), rep)
        println()
    end

    # Benchmark: build a fresh container + add all costs. The container build is
    # the same work each iteration, so deltas across rungs reflect the cost-path
    # change. (To isolate cost-path more precisely, use @profile on a longer run.)
    println("\n=== @btime build + add_all_costs! (N=$N_COMPONENTS, T=$(length(TIME_STEPS))) ===")
    @btime begin
        c = make_container($devices)
        add_all_costs!(c, $devices, $curves)
    end

    # Single-component @code_warntype at the entry point — useful when iterating
    # on which call sites remain type-unstable. Comment out for batch runs.
    println("\n=== @code_warntype add_variable_cost_to_objective! (single call) ===")
    let c = make_container(devices)
        d, cc = devices[1], curves[1]
        InteractiveUtils.@code_warntype IOM.add_variable_cost_to_objective!(
            c, TestCostVariable, d, cc, TestFormulation,
        )
    end

    return nothing
end

main()
