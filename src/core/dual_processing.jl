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

# MIP solver tolerances leave discrete values like 0.9999997 instead of 1.0;
# fixing a relaxed variable to such a value makes the LP relaxation infeasible.
_relaxed_fix_value(v::JuMP.VariableRef) = round(JuMP.value(v))

# Mirror the rounding applied when fixing back into the cached primal values so
# the reported integer/binary solution is exact.
_round_discrete_cache!(::SparseAxisArray, ::JuMP.Containers.SparseAxisArray) = nothing
function _round_discrete_cache!(values::DenseAxisArray, variable::DenseAxisArray)
    v1 = first(variable)
    if JuMP.is_binary(v1) || JuMP.is_integer(v1)
        values.data .= round.(values.data)
    end
    return
end

function process_duals(container::OptimizationContainer, lp_optimizer)
    var_cache = container.primal_values_cache.variables_cache
    for (k, v) in get_variables(container)
        var_cache[k] = jump_value.(v)
        _round_discrete_cache!(var_cache[k], v)
    end
    for (k, v) in get_expressions(container)
        container.primal_values_cache.expressions_cache[k] = jump_value.(v)
    end

    jump_model = get_jump_model(container)
    undo_relaxation = JuMP.fix_discrete_variables(_relaxed_fix_value, jump_model)

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

    if JuMP.has_duals(jump_model)
        for (key, dual) in get_duals(container)
            _copy_dual_values!(dual, get_constraint(container, key))
        end
    end

    undo_relaxation()
    return RunStatus.SUCCESSFULLY_FINALIZED
end
