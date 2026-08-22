# How much first-use compile is shared across DIFFERENT key types?
using InfrastructureOptimizationModels
const IOM = InfrastructureOptimizationModels
using InfrastructureSystems
const IS = InfrastructureSystems
using JuMP, Dates
const MOI = JuMP.MOI

mutable struct WLSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
end
IOM.get_base_power(s::WLSystem) = s.base_power
IOM.stores_time_series_in_memory(::WLSystem) = true
struct WLComponentA <: IS.InfrastructureSystemsComponent end
struct WLVarA <: IOM.VariableType end
struct WLConA <: IOM.ConstraintType end
struct WLExprA <: IOM.ExpressionType end
struct WLComponentB <: IS.InfrastructureSystemsComponent end
struct WLVarB <: IOM.VariableType end
struct WLConB <: IOM.ConstraintType end
struct WLExprB <: IOM.ExpressionType end

function container_workload(::Type{C}, ::Type{V}, ::Type{K}, ::Type{E}) where {C, V, K, E}
    sys = WLSystem(100.0)
    settings = IOM.Settings(sys; horizon = Hour(24), resolution = Hour(1), time_series_cache_size = 0)
    container = IOM.OptimizationContainer(sys, settings, nothing, IS.Deterministic)
    IOM.set_time_steps!(container, 1:24)
    jm = IOM.get_jump_model(container)
    names = ["c1", "c2", "c3"]
    ts = IOM.get_time_steps(container)
    vc = IOM.add_variable_container!(container, V, C, names, ts)
    for n in names, t in ts
        vc[n, t] = JuMP.@variable(jm, base_name = "V_{$(n), $(t)}", lower_bound = 0.0, upper_bound = 10.0)
    end
    ec = IOM.add_expression_container!(container, E, C, names, ts)
    for n in names, t in ts
        ec[n, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(ec[n, t], vc[n, t])
    end
    cc = IOM.add_constraints_container!(container, K, C, names, ts)
    for n in names, t in ts
        cc[n, t] = JuMP.@constraint(jm, ec[n, t] <= 5.0)
    end
    IOM.add_to_objective_invariant_expression!(container, 2.0 * vc["c1", 1])
    obj = IOM.get_objective_expression(container.objective_function)
    JuMP.@objective(jm, MOI.MIN_SENSE, obj)
    return container
end

t1 = time(); container_workload(WLComponentA, WLVarA, WLConA, WLExprA); println("FIRST_TYPESET_A=", round(time() - t1; digits = 2))
t2 = time(); container_workload(WLComponentB, WLVarB, WLConB, WLExprB); println("SECOND_TYPESET_B=", round(time() - t2; digits = 3))
t3 = time(); container_workload(WLComponentA, WLVarA, WLConA, WLExprA); println("REPEAT_TYPESET_A=", round(time() - t3; digits = 3))
