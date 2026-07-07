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
rounded_value(v::JuMP.VariableRef) = round(JuMP.value(v))

function process_duals(container::OptimizationContainer, lp_optimizer)
    var_cache = container.primal_values_cache.variables_cache
    for (k, v) in get_variables(container)
        v1 = first(v)
        if JuMP.is_binary(v1) || JuMP.is_integer(v1)
            var_cache[k] = round.(jump_value.(v))
        else
            var_cache[k] = jump_value.(v)
        end
    end
    for (k, v) in get_expressions(container)
        container.primal_values_cache.expressions_cache[k] = jump_value.(v)
    end

    jump_model = get_jump_model(container)
    undo_relaxation = JuMP.fix_discrete_variables(rounded_value, jump_model)

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
