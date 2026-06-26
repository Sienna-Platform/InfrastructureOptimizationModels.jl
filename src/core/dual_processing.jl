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

function process_duals(container::OptimizationContainer, lp_optimizer)
    var_container = get_variables(container)
    for (k, v) in var_container
        container.primal_values_cache.variables_cache[k] = jump_value.(v)
    end

    for (k, v) in get_expressions(container)
        container.primal_values_cache.expressions_cache[k] = jump_value.(v)
    end
    var_cache = container.primal_values_cache.variables_cache
    cache = sizehint!(Dict{VariableKey, Dict{Symbol, Any}}(), length(var_container))
    for (key, variable) in get_variables(container)
        if isa(variable, JuMP.Containers.SparseAxisArray)
            if any(
                v -> v isa JuMP.VariableRef && (JuMP.is_binary(v) || JuMP.is_integer(v)),
                values(variable.data),
            )
                @warn "Sparse variable container $(key) holds binary/integer variables " *
                      "that process_duals does not relax; duals may be unavailable or stale." maxlog =
                    1
            end
            continue
        end
        if JuMP.is_binary(first(variable))
            JuMP.unset_binary.(variable)
            is_integer_flag = false
        elseif JuMP.is_integer(first(variable))
            JuMP.unset_integer.(variable)
            is_integer_flag = true
        else
            continue
        end
        # Bounds and fixed status are per element: bounds come from per-device hooks,
        # so a container can mix bounded and unbounded variables. Cache before fix!
        # (force = true deletes bounds).
        cache[key] = Dict{Symbol, Any}(
            :integer => is_integer_flag,
            :lb => _lower_bound_or_nothing.(variable),
            :ub => _upper_bound_or_nothing.(variable),
            :fixed => JuMP.is_fixed.(variable),
        )
        var_cache[key].data .= round.(var_cache[key].data)
        JuMP.fix.(variable, var_cache[key]; force = true)
    end
    if isempty(cache)
        error(
            "process_duals found no dense binary/integer variable containers to relax; " *
            "cannot compute duals for this MILP.",
        )
    end
    jump_model = get_jump_model(container)

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
        return RunStatus.FAILED
    end

    if JuMP.has_duals(jump_model)
        for (key, dual) in get_duals(container)
            constraint = get_constraint(container, key)
            _copy_dual_values!(dual, constraint)
        end
    end

    for (key, variable) in get_variables(container)
        entry = get(cache, key, nothing)
        entry === nothing && continue
        JuMP.unfix.(variable)
        _restore_lower_bound!.(variable, entry[:lb])
        _restore_upper_bound!.(variable, entry[:ub])
        if entry[:integer]
            JuMP.set_integer.(variable)
        else
            JuMP.set_binary.(variable)
        end
        _refix!.(variable, entry[:fixed], var_cache[key])
    end
    return RunStatus.SUCCESSFULLY_FINALIZED
end

_lower_bound_or_nothing(v::JuMP.VariableRef) =
    JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : nothing
_upper_bound_or_nothing(v::JuMP.VariableRef) =
    JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : nothing
_restore_lower_bound!(::JuMP.VariableRef, ::Nothing) = nothing
_restore_lower_bound!(v::JuMP.VariableRef, lb::Float64) = JuMP.set_lower_bound(v, lb)
_restore_upper_bound!(::JuMP.VariableRef, ::Nothing) = nothing
_restore_upper_bound!(v::JuMP.VariableRef, ub::Float64) = JuMP.set_upper_bound(v, ub)
_refix!(v::JuMP.VariableRef, was_fixed::Bool, val::Float64) =
    was_fixed ? JuMP.fix(v, val; force = true) : nothing
