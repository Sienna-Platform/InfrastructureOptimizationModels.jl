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
