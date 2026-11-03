# DenseAxisArray duals broadcast over the backing array. Post-contingency
# duals are SparseAxisArray (Dict-backed), where `.data .= …` is undefined, so
# copy per key instead.
function _copy_dual_values!(dual::DenseAxisArray, constraint::DenseAxisArray)
    # The dual container is built by reusing the constraint's axes (see
    # `assign_dual_variable!` / `_assign_dual_from_existing!`), so a positional
    # copy is correct only when the axes match. Mismatched axes would write each
    # value onto the wrong label, so fail loudly instead.
    IS.@assert_op axes(dual) == axes(constraint)
    dual.data .= jump_value.(constraint).data
    return
end

function _copy_dual_values!(dual::SparseAxisArray, constraint::SparseAxisArray)
    for (k, cref) in constraint.data
        dual.data[k] = jump_value(cref)
    end
    return
end

# Solver tolerances may give us "integer" values that aren't exactly 0 or 1.
rounded_value(v::JuMP.VariableRef) = round(jump_value(v))

struct DiscreteVariableState
    variable::JuMP.VariableRef
    binary::Bool
    value::Float64
    lower_bound::Union{Nothing, Float64}
    upper_bound::Union{Nothing, Float64}
    fixed_value::Union{Nothing, Float64}
end

function DiscreteVariableState(v::JuMP.VariableRef, binary::Bool, var_value::Function)
    return DiscreteVariableState(
        v,
        binary,
        var_value(v),
        JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : nothing,
        JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : nothing,
        JuMP.is_fixed(v) ? JuMP.fix_value(v) : nothing,
    )
end

# Fix every binary and integer variable at `var_value` and drop its integrality, returning a
# function that restores both. Integrality is read from the model's `ZeroOne` and `Integer`
# constraints rather than `JuMP.is_integer`: through a bridge optimizer in direct mode, the
# `Integer` constraint a bridge adds to reformulate a binary is reported by `is_integer`
# although the variable is only binary.
function fix_discrete_variables!(var_value::Function, jump_model::JuMP.Model)
    # Every state is read before the first change invalidates the solution.
    states = DiscreteVariableState[]
    for (S, binary) in ((MOI.ZeroOne, true), (MOI.Integer, false))
        for c in JuMP.all_constraints(jump_model, JuMP.VariableRef, S)
            push!(states, DiscreteVariableState(JuMP.VariableRef(c), binary, var_value))
        end
    end
    for state in states
        if state.binary
            JuMP.unset_binary(state.variable)
        else
            JuMP.unset_integer(state.variable)
        end
        JuMP.fix(state.variable, state.value; force = true)
    end
    return () -> restore_discrete_variables!(states)
end

function restore_discrete_variables!(states::Vector{DiscreteVariableState})
    for state in states
        v = state.variable
        JuMP.unfix(v)
        isnothing(state.fixed_value) || JuMP.fix(v, state.fixed_value)
        isnothing(state.lower_bound) || JuMP.set_lower_bound(v, state.lower_bound)
        isnothing(state.upper_bound) || JuMP.set_upper_bound(v, state.upper_bound)
        if state.binary
            JuMP.set_binary(v)
        else
            JuMP.set_integer(v)
        end
    end
    return
end

function process_duals(container::OptimizationContainer, lp_optimizer)
    var_cache = container.primal_values_cache.variables_cache
    for (k, v) in get_variables(container)
        v1 = first(v)
        if JuMP.is_binary(v1) || JuMP.is_integer(v1)
            var_cache[k] = rounded_value.(v)
        else
            var_cache[k] = jump_value.(v)
        end
    end
    for (k, v) in get_expressions(container)
        container.primal_values_cache.expressions_cache[k] = jump_value.(v)
    end

    jump_model = get_jump_model(container)
    undo_relaxation = fix_discrete_variables!(rounded_value, jump_model)

    if JuMP.mode(jump_model) != JuMP.DIRECT
        JuMP.set_optimizer(jump_model, lp_optimizer)
    else
        @debug("JuMP model set in direct mode during dual calculation")
    end

    JuMP.optimize!(jump_model)

    model_status = JuMP.primal_status(jump_model)
    if model_status ∉ [
        MOI.FEASIBLE_POINT::MOI.ResultStatusCode,
        MOI.NEARLY_FEASIBLE_POINT::MOI.ResultStatusCode,
    ]
        @error "Optimizer returned $model_status during dual calculation"
        undo_relaxation()
        return RunStatus.FAILED
    end

    if !isempty(get_duals(container)) && !JuMP.has_duals(jump_model)
        @error "Optimizer returned no duals during dual calculation" requested =
            collect(keys(get_duals(container)))
        undo_relaxation()
        return RunStatus.FAILED
    end
    for (key, dual) in get_duals(container)
        _copy_dual_values!(dual, get_constraint(container, key))
    end

    undo_relaxation()
    return RunStatus.SUCCESSFULLY_FINALIZED
end
