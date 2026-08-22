# IOM TTFX measurement harness. Run: julia --project=<IOM> iom_ttfx.jl
t_start = time()
using InfrastructureOptimizationModels
const IOM = InfrastructureOptimizationModels
using InfrastructureSystems
const IS = InfrastructureSystems
using JuMP
using Dates
const MOI = JuMP.MOI
t_load = time() - t_start
println("LOAD_SECONDS=", round(t_load; digits = 2))

# --- workload mock types (mirror test/mocks minimal subset) ---
mutable struct WLSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
end
IOM.get_base_power(s::WLSystem) = s.base_power
IOM.stores_time_series_in_memory(::WLSystem) = true

struct WLComponent <: IS.InfrastructureSystemsComponent end
struct WLVar <: IOM.VariableType end
struct WLCon <: IOM.ConstraintType end
struct WLBoundCon <: IOM.ConstraintType end
struct WLExpr <: IOM.ExpressionType end

function container_workload()
    sys = WLSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Hour(24),
        resolution = Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(sys, settings, nothing, IS.Deterministic)
    IOM.set_time_steps!(container, 1:24)
    jm = IOM.get_jump_model(container)
    names = ["c1", "c2", "c3"]
    ts = IOM.get_time_steps(container)

    vc = IOM.add_variable_container!(container, WLVar, WLComponent, names, ts)
    for n in names, t in ts
        vc[n, t] = JuMP.@variable(
            jm,
            base_name = "WLVar_{$(n), $(t)}",
            lower_bound = 0.0,
            upper_bound = 10.0
        )
    end

    ec = IOM.add_expression_container!(container, WLExpr, WLComponent, names, ts)
    for n in names, t in ts
        ec[n, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(ec[n, t], vc[n, t])
    end

    cc = IOM.add_constraints_container!(container, WLCon, WLComponent, names, ts)
    for n in names, t in ts
        cc[n, t] = JuMP.@constraint(jm, ec[n, t] <= 5.0)
    end

    # objective terms through the container API
    IOM.add_to_objective_invariant_expression!(container, 2.0 * vc["c1", 1])
    obj = IOM.get_objective_expression(container.objective_function)
    JuMP.@objective(jm, MOI.MIN_SENSE, obj)
    return container
end

t1 = time()
container_workload()
t_first = time() - t1
println("FIRST_CALL_SECONDS=", round(t_first; digits = 2))

t2 = time()
container_workload()
t_second = time() - t2
println("SECOND_CALL_SECONDS=", round(t_second; digits = 3))
println("TOTAL_TTFX_SECONDS=", round(time() - t_start; digits = 2))
