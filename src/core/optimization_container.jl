struct PrimalValuesCache
    variables_cache::Dict{VariableKey, JuMPArray}
    expressions_cache::Dict{ExpressionKey, JuMPArray}
end

function PrimalValuesCache()
    return PrimalValuesCache(
        Dict{VariableKey, JuMPArray}(),
        Dict{ExpressionKey, JuMPArray}(),
    )
end

function Base.isempty(pvc::PrimalValuesCache)
    return isempty(pvc.variables_cache) && isempty(pvc.expressions_cache)
end

mutable struct ObjectiveFunction
    invariant_terms::JuMPScalarExpr
    variant_terms::GAE
    synchronized::Bool
    sense::MOI.OptimizationSense
    function ObjectiveFunction(invariant_terms::JuMPScalarExpr,
        variant_terms::GAE,
        synchronized::Bool,
        sense::MOI.OptimizationSense = MOI.MIN_SENSE)
        new(invariant_terms, variant_terms, synchronized, sense)
    end
end

get_invariant_terms(v::ObjectiveFunction) = v.invariant_terms
get_variant_terms(v::ObjectiveFunction) = v.variant_terms
function get_objective_expression(v::ObjectiveFunction)
    if iszero(v.variant_terms)
        return v.invariant_terms
    else
        # JuMP doesn't support expression conversion from Affn to QuadExpressions
        if isa(v.invariant_terms, JuMP.GenericQuadExpr)
            # Avoid mutation of invariant term
            temp_expr = JuMP.QuadExpr()
            JuMP.add_to_expression!(temp_expr, v.invariant_terms)
            return JuMP.add_to_expression!(temp_expr, v.variant_terms)
        else
            # This will mutate the variant terms, but these are reseted at each step.
            return JuMP.add_to_expression!(v.variant_terms, v.invariant_terms)
        end
    end
end
get_sense(v::ObjectiveFunction) = v.sense
is_synchronized(v::ObjectiveFunction) = v.synchronized
set_synchronized_status!(v::ObjectiveFunction, value) = v.synchronized = value
reset_variant_terms(v::ObjectiveFunction) = v.variant_terms = zero(JuMP.AffExpr)
has_variant_terms(v::ObjectiveFunction) = !iszero(v.variant_terms)
set_sense!(v::ObjectiveFunction, sense::MOI.OptimizationSense) = v.sense = sense

function ObjectiveFunction()
    return ObjectiveFunction(
        zero(JuMP.GenericAffExpr{Float64, JuMP.VariableRef}),
        zero(JuMP.AffExpr),
        true,
    )
end

mutable struct OptimizationContainer <: AbstractOptimizationContainer
    JuMPmodel::JuMP.Model
    time_steps::UnitRange{Int}
    settings::Settings
    variables::OrderedDict{VariableKey, JuMPArray}
    aux_variables::OrderedDict{AuxVarKey, JuMPArray}
    duals::OrderedDict{ConstraintKey, JuMPArray}
    constraints::OrderedDict{ConstraintKey, JuMPArray}
    objective_function::ObjectiveFunction
    expressions::OrderedDict{ExpressionKey, JuMPArray}
    parameters::OrderedDict{ParameterKey, ParameterContainer}
    primal_values_cache::PrimalValuesCache
    initial_conditions::OrderedDict{InitialConditionKey, Vector{<:InitialCondition}}
    initial_conditions_data::InitialConditionsData
    infeasibility_conflict::Dict{Symbol, Array}
    pm::Union{Nothing, AbstractPowerModel}
    model_base_power::Float64
    optimizer_stats::OptimizerStats
    built_for_recurrent_solves::Bool
    pf_aux_var_keys::Vector{AuxVarKey}
    non_pf_aux_var_keys::Vector{AuxVarKey}
    metadata::OptimizationContainerMetadata
    default_time_series_type::Type
    power_flow_evaluation_data::Vector{<:AbstractPowerFlowEvaluationData}
end

function OptimizationContainer(
    sys,  # Any system implementing get_base_power (duck-typing)
    settings::Settings,
    jump_model::Union{Nothing, JuMP.Model},
    ::Type{T},
) where {T}
    if isabstracttype(T)
        error("Default Time Series Type $T can't be abstract")
    end

    if jump_model !== nothing && get_direct_mode_optimizer(settings)
        throw(
            IS.ConflictingInputsError(
                "Externally provided JuMP models are not compatible with the direct model keyword argument. Use JuMP.direct_model before passing the custom model",
            ),
        )
    end

    return OptimizationContainer(
        jump_model === nothing ? JuMP.Model() : jump_model,
        1:1,
        settings,
        OrderedDict{VariableKey, JuMPArray}(),
        OrderedDict{AuxVarKey, JuMPArray}(),
        OrderedDict{ConstraintKey, JuMPArray}(),
        OrderedDict{ConstraintKey, JuMPArray}(),
        ObjectiveFunction(),
        OrderedDict{ExpressionKey, JuMPArray}(),
        OrderedDict{ParameterKey, ParameterContainer}(),
        PrimalValuesCache(),
        OrderedDict{InitialConditionKey, Vector{InitialCondition}}(),
        InitialConditionsData(),
        Dict{Symbol, Array}(),
        nothing,
        get_base_power(sys),
        OptimizerStats(),
        false,
        AuxVarKey[],
        AuxVarKey[],
        OptimizationContainerMetadata(),
        T,
        AbstractPowerFlowEvaluationData[],
    )
end

built_for_recurrent_solves(container::OptimizationContainer) =
    container.built_for_recurrent_solves

get_aux_variables(container::OptimizationContainer) = container.aux_variables
get_model_base_power(container::OptimizationContainer) = container.model_base_power
get_constraints(container::OptimizationContainer) = container.constraints

function cost_function_unsynch(container::OptimizationContainer)
    obj_func = get_objective_expression(container)
    if has_variant_terms(obj_func) && is_synchronized(container)
        set_synchronized_status!(obj_func, false)
        reset_variant_terms(obj_func)
    end
    return
end

function get_container_keys(container::OptimizationContainer)
    return Iterators.flatten(keys(getfield(container, f)) for f in STORE_CONTAINERS)
end

get_default_time_series_type(container::OptimizationContainer) =
    container.default_time_series_type
get_duals(container::OptimizationContainer) = container.duals
get_expressions(container::OptimizationContainer) = container.expressions
get_infeasibility_conflict(container::OptimizationContainer) =
    container.infeasibility_conflict
get_initial_conditions(container::OptimizationContainer) = container.initial_conditions
get_initial_conditions_data(container::OptimizationContainer) =
    container.initial_conditions_data
get_initial_time(container::OptimizationContainer) = get_initial_time(container.settings)
get_jump_model(container::OptimizationContainer) = container.JuMPmodel
get_metadata(container::OptimizationContainer) = container.metadata
get_optimizer_stats(container::OptimizationContainer) = container.optimizer_stats
get_parameters(container::OptimizationContainer) = container.parameters
get_power_flow_evaluation_data(container::OptimizationContainer) =
    container.power_flow_evaluation_data
get_resolution(container::OptimizationContainer) = get_resolution(container.settings)
get_settings(container::OptimizationContainer) = container.settings
get_time_steps(container::OptimizationContainer) = container.time_steps
get_variables(container::OptimizationContainer) = container.variables

set_initial_conditions_data!(container::OptimizationContainer, data) =
    container.initial_conditions_data = data
get_objective_expression(container::OptimizationContainer) = container.objective_function
is_synchronized(container::OptimizationContainer) =
    container.objective_function.synchronized
set_time_steps!(container::OptimizationContainer, time_steps::UnitRange{Int64}) =
    container.time_steps = time_steps

function reset_power_flow_is_solved!(container::OptimizationContainer)
    for pf_e_data in get_power_flow_evaluation_data(container)
        pf_e_data.is_solved = false
    end
end

@generated function has_container_key(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: OptimizationKeyType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    # without the QuoteNode, we'd get <field_name> when interpolated, not :<field_name>
    # also, QuoteNode(foo) evaluates foo then quotes, whereas :(foo) quotes first.
    field = QuoteNode(field_for_type(T))
    K = key_for_type(T)
    return :(return haskey(getfield(container, $field), $K(T, U, meta)))
end

function is_milp(container::OptimizationContainer)::Bool
    !supports_milp(container) && return false
    if !isempty(
        JuMP.all_constraints(
            get_jump_model(container),
            JuMP.VariableRef,
            JuMP.MOI.ZeroOne,
        ),
    )
        return true
    end
    return false
end

function supports_milp(container::OptimizationContainer)
    jump_model = get_jump_model(container)
    return supports_milp(jump_model)
end

function validate_warm_start_support(JuMPmodel::JuMP.Model, warm_start_enabled::Bool)
    !warm_start_enabled && return warm_start_enabled
    solver_supports_warm_start =
        MOI.supports(JuMP.backend(JuMPmodel), MOI.VariablePrimalStart(), MOI.VariableIndex)
    if !solver_supports_warm_start
        solver_name = JuMP.solver_name(JuMPmodel)
        @warn("$(solver_name) does not support warm start")
    end
    return solver_supports_warm_start
end

function make_empty_jump_model_with_settings(ic_settings::Settings)
    optimizer = get_optimizer(ic_settings)
    JuMPmodel = JuMP.Model(optimizer)
    warm_start_enabled = get_warm_start(ic_settings)
    solver_supports_warm_start = validate_warm_start_support(JuMPmodel, warm_start_enabled)
    set_warm_start!(ic_settings, solver_supports_warm_start)
    if get_optimizer_solve_log_print(ic_settings)
        JuMP.unset_silent(JuMPmodel)
        @debug "optimizer unset to silent" _group = LOG_GROUP_OPTIMIZATION_CONTAINER
    else
        JuMP.set_silent(JuMPmodel)
        @debug "optimizer set to silent" _group = LOG_GROUP_OPTIMIZATION_CONTAINER
    end
    return JuMPmodel
end

function finalize_jump_model!(container::OptimizationContainer, settings::Settings)
    @debug "Instantiating the JuMP model" _group = LOG_GROUP_OPTIMIZATION_CONTAINER
    if built_for_recurrent_solves(container) && get_optimizer(settings) === nothing
        throw(
            IS.ConflictingInputsError(
                "Optimizer can not be nothing when building for recurrent solves",
            ),
        )
    end

    if get_direct_mode_optimizer(settings)
        optimizer = () -> MOI.instantiate(get_optimizer(settings))
        container.JuMPmodel = JuMP.direct_model(optimizer())
    elseif get_optimizer(settings) === nothing
        @debug "The optimization model has no optimizer attached" _group =
            LOG_GROUP_OPTIMIZATION_CONTAINER
    else
        JuMP.set_optimizer(get_jump_model(container), get_optimizer(settings))
    end

    JuMPmodel = get_jump_model(container)
    warm_start_enabled = get_warm_start(settings)
    solver_supports_warm_start = validate_warm_start_support(JuMPmodel, warm_start_enabled)
    set_warm_start!(settings, solver_supports_warm_start)

    JuMP.set_string_names_on_creation(JuMPmodel, get_store_variable_names(settings))

    @debug begin
        JuMP.set_string_names_on_creation(JuMPmodel, true)
    end
    if get_optimizer_solve_log_print(settings)
        JuMP.unset_silent(JuMPmodel)
        @debug "optimizer unset to silent" _group = LOG_GROUP_OPTIMIZATION_CONTAINER
    else
        JuMP.set_silent(JuMPmodel)
        @debug "optimizer set to silent" _group = LOG_GROUP_OPTIMIZATION_CONTAINER
    end
    return
end

# Dispatch helpers so init_optimization_container! works with both PSY.System and mock containers.
temp_set_units_base_system!(sys::PSY.System, base::String) =
    PSY.set_units_base_system!(sys, base)
temp_set_units_base_system!(::IS.InfrastructureSystemsContainer, ::String) = nothing
temp_get_forecast_initial_timestamp(sys::PSY.System) =
    PSY.get_forecast_initial_timestamp(sys)
temp_get_forecast_initial_timestamp(::IS.InfrastructureSystemsContainer) =
    Dates.DateTime(1970)

function init_optimization_container!(
    container::OptimizationContainer,
    network_model::NetworkModel{T},
    sys::IS.InfrastructureSystemsContainer,
) where {T <: AbstractPowerModel}
    # PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    temp_set_units_base_system!(sys, "SYSTEM_BASE")
    # The order of operations matter
    settings = get_settings(container)

    if get_initial_time(settings) == UNSET_INI_TIME
        if get_default_time_series_type(container) <: PSY.AbstractDeterministic
            # set_initial_time!(settings, PSY.get_forecast_initial_timestamp(sys))
            set_initial_time!(settings, temp_get_forecast_initial_timestamp(sys))
        elseif get_default_time_series_type(container) <: PSY.SingleTimeSeries
            ini_time, _ = PSY.check_time_series_consistency(sys, PSY.SingleTimeSeries)
            set_initial_time!(settings, ini_time)
        end
    end

    if get_resolution(settings) == UNSET_RESOLUTION
        error("Resolution not set in the model. Can't continue with the build.")
    end

    horizon_count = (get_horizon(settings) ÷ get_resolution(settings))
    @assert horizon_count > 0
    container.time_steps = 1:horizon_count

    # NOTE: Simplified to avoid referencing concrete network model types (CopperPlatePowerModel, AreaBalancePowerModel)
    # PowerSimulations can implement more specific logic based on concrete types
    total_number_of_devices =
        length(get_available_components(network_model, PSY.Device, sys))
    total_number_of_devices +=
        length(get_available_components(network_model, PSY.ACBranch, sys))

    # The 10e6 limit is based on the sizes of the lp benchmark problems http://plato.asu.edu/ftp/lpcom.html
    # The maximum numbers of constraints and variables in the benchmark problems is 1,918,399 and 1,259,121,
    # respectively. See also https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2013/138847.pdf
    variable_count_estimate = length(container.time_steps) * total_number_of_devices

    if variable_count_estimate > 10e6
        @warn(
            "The lower estimate of total number of variables that will be created in the model is $(variable_count_estimate). \\
            The total number of variables might be larger than 10e6 and could lead to large build or solve times."
        )
    end

    stats = get_optimizer_stats(container)
    stats.detailed_stats = get_detailed_optimizer_stats(settings)

    finalize_jump_model!(container, settings)
    return
end

function reset_optimization_model!(container::OptimizationContainer)
    for field in [:variables, :aux_variables, :constraints, :expressions, :duals]
        empty!(getfield(container, field))
    end
    empty!(container.pf_aux_var_keys)
    empty!(container.non_pf_aux_var_keys)
    container.initial_conditions_data = InitialConditionsData()
    container.objective_function = ObjectiveFunction()
    container.primal_values_cache = PrimalValuesCache()
    JuMP.empty!(get_jump_model(container))
    return
end

function check_parameter_multiplier_values(multiplier_array::DenseAxisArray)
    return !all(isnan.(multiplier_array.data))
end

function check_parameter_multiplier_values(multiplier_array::SparseAxisArray)
    return !all(isnan.(values(multiplier_array.data)))
end

function check_optimization_container(container::OptimizationContainer)
    for (k, param_container) in container.parameters
        valid = check_parameter_multiplier_values(param_container.multiplier_array)
        if !valid
            error("The model container has invalid values in $(encode_key_as_string(k))")
        end
    end
    return
end

function get_problem_size(container::OptimizationContainer)
    model = get_jump_model(container)
    vars = JuMP.num_variables(model)
    cons = 0
    for (exp, c_type) in JuMP.list_of_constraint_types(model)
        cons += JuMP.num_constraints(model, exp, c_type)
    end
    return "The current total number of variables is $(vars) and total number of constraints is $(cons)"
end

function update_objective_function!(container::OptimizationContainer)
    JuMP.@objective(
        get_jump_model(container),
        get_sense(container.objective_function),
        get_objective_expression(container.objective_function)
    )
    return
end

"""
Execute the optimizer on the container's JuMP model, compute aux/dual variables,
and return the run status. Called `solve_impl!(container, system)` in PSI.
"""
function execute_optimizer!(
    container::OptimizationContainer,
    system::IS.InfrastructureSystemsContainer,
)
    optimizer_stats = get_optimizer_stats(container)

    jump_model = get_jump_model(container)

    model_status = MOI.NO_SOLUTION::MOI.ResultStatusCode
    conflict_status = MOI.COMPUTE_CONFLICT_NOT_CALLED

    try_count = 0
    while model_status != MOI.FEASIBLE_POINT::MOI.ResultStatusCode
        _,
        optimizer_stats.timed_solve_time,
        optimizer_stats.solve_bytes_alloc,
        optimizer_stats.sec_in_gc = @timed JuMP.optimize!(jump_model)
        model_status = JuMP.primal_status(jump_model)

        if model_status != MOI.FEASIBLE_POINT::MOI.ResultStatusCode
            if get_calculate_conflict(get_settings(container))
                @warn "Optimizer returned $model_status computing conflict"
                conflict_status = compute_conflict!(container)
                if conflict_status == MOI.CONFLICT_FOUND
                    return RunStatus.FAILED
                end
            else
                @warn "Optimizer returned $model_status trying optimize! again"
            end

            try_count += 1
            if try_count > MAX_OPTIMIZE_TRIES
                @error "Optimizer returned $model_status after $MAX_OPTIMIZE_TRIES optimize! attempts"
                return RunStatus.FAILED
            end
        end
    end

    # Order is important because if a dual is needed then it could move the outputs to the
    # temporary primal container
    _, optimizer_stats.timed_calculate_aux_variables =
        @timed calculate_aux_variables!(container, system)

    # Needs to be called here to avoid issues when getting duals from MILPs
    write_optimizer_stats!(container)

    _, optimizer_stats.timed_calculate_dual_variables =
        @timed calculate_dual_variables!(container, system, is_milp(container))

    return RunStatus.SUCCESSFULLY_FINALIZED
end

function compute_conflict!(container::OptimizationContainer)
    jump_model = get_jump_model(container)
    settings = get_settings(container)
    JuMP.unset_silent(jump_model)
    jump_model.is_model_dirty = false
    conflict = container.infeasibility_conflict
    try
        JuMP.compute_conflict!(jump_model)
        conflict_status = MOI.get(jump_model, MOI.ConflictStatus())
        if conflict_status != MOI.CONFLICT_FOUND
            @error "No conflict could be found for the model. Status: $conflict_status"
            if !get_optimizer_solve_log_print(settings)
                JuMP.set_silent(jump_model)
            end
            return conflict_status
        end

        for (key, field_container) in get_constraints(container)
            conflict_indices = check_conflict_status(jump_model, field_container)
            if isempty(conflict_indices)
                @info "Conflict Index returned empty for $key"
                continue
            else
                conflict[encode_key(key)] = conflict_indices
            end
        end

        msg = IOBuffer()
        for (k, v) in conflict
            PrettyTables.pretty_table(msg, v; header = [k])
        end

        @error "Constraints participating in conflict basis (IIS) \n\n$(String(take!(msg)))"

        return conflict_status
    catch e
        jump_model.is_model_dirty = true
        if isa(e, MethodError)
            @info "Can't compute conflict, check that your optimizer supports conflict refining/IIS"
        else
            @error "Can't compute conflict" exception = (e, catch_backtrace())
        end
    end

    return MOI.NO_CONFLICT_EXISTS
end

function write_optimizer_stats!(container::OptimizationContainer)
    write_optimizer_stats!(get_optimizer_stats(container), get_jump_model(container))
    return
end

"""
Exports the OpModel JuMP object in MathOptFormat
"""
function serialize_optimization_model(container::OptimizationContainer, save_path::String)
    serialize_jump_optimization_model(get_jump_model(container), save_path)
    return
end

function serialize_metadata!(container::OptimizationContainer, output_dir::String)
    for key in Iterators.flatten((
        keys(container.constraints),
        keys(container.duals),
        keys(container.parameters),
        keys(container.variables),
        keys(container.aux_variables),
        keys(container.expressions),
    ))
        encoded_key = encode_key_as_string(key)
        if has_container_key(container.metadata, encoded_key)
            # Constraints and Duals can store the same key.
            IS.@assert_op key ==
                          get_container_key(container.metadata, encoded_key)
        end
        add_container_key!(container.metadata, encoded_key, key)
    end

    filename = _make_metadata_filename(output_dir)
    Serialization.serialize(filename, container.metadata)
    @debug "Serialized container keys to $filename" _group = IS.LOG_GROUP_SERIALIZATION
end

function deserialize_metadata!(
    container::OptimizationContainer,
    output_dir::String,
    model_name,
)
    merge!(
        container.metadata.container_key_lookup,
        deserialize_metadata(
            OptimizationContainerMetadata,
            output_dir,
            model_name,
        ),
    )
    return
end

# PERF: compilation hotspot. from string conversion at the container[key] = value line?
function _assign_container!(container::OrderedDict, key::OptimizationContainerKey, value)
    if haskey(container, key)
        @error "$(encode_key(key)) is already stored" sort!(
            encode_key.(keys(container)),
        )
        throw(IS.InvalidValue("$key is already stored"))
    end
    container[key] = value
    @debug "Added container entry $(typeof(key)) $(encode_key(key))" _group =
        LOG_GROUP_OPTIMIZATION_CONTAINER
    return
end

########################### Generic Container Access ########################################

"""
Generic container entry getter. Dispatches on key type to access the correct field.
Replaces per-type get_variable, get_aux_variable, get_constraint, etc. by key.
"""
@generated function _get_entry(
    container::OptimizationContainer,
    key::OptimizationContainerKey{T, U},
) where {
    T <: OptimizationKeyType,
    U <: InfrastructureSystemsType,
}
    field = QuoteNode(field_for_type(T))
    return quote
        val = get(getfield(container, $field), key, nothing)
        if val === nothing
            throw(
                IS.InvalidValue(
                    "$(encode_key(key)) is not stored. $(encode_key.(collect(keys(getfield(container, $field)))))",
                ),
            )
        end
        return val
    end
end

"""
Generic container entry getter by entry type and component type.
Constructs the appropriate key and delegates to the key-based _get_entry.
"""
@generated function _get_entry(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: OptimizationKeyType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    K = key_for_type(T)
    return :(return _get_entry(container, $K(T, U, meta)))
end

########################### Generic Container Creation ######################################

@generated function _add_container!(
    opt_container::OptimizationContainer,
    key::OptimizationContainerKey{T, U},
    ::Type{E},
    sparse::Bool,
    axs::Vararg{Any, N},
) where {
    T <: OptimizationKeyType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    E,
    N,
}
    field = QuoteNode(field_for_type(T))
    return quote
        if sparse
            value = sparse_container_spec(E, axs...)
        else
            value = container_spec(E, axs...)
        end
        _assign_container!(getfield(opt_container, $field), key, value)
        return value
    end
end

"""
Key-constructing overload: builds the key from (T, U, meta) then delegates.
"""
@generated function _add_container!(
    opt_container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    ::Type{E},
    sparse::Bool,
    axs::Vararg{Any, N};
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: OptimizationKeyType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    E,
    N,
}
    K = key_for_type(T)
    return :(return _add_container!(opt_container, $K(T, U, meta), E, sparse, axs...))
end

####################################### Variable Container #################################
add_variable_container!(
    container::OptimizationContainer, ::Type{T}, ::Type{U}, axs::Vararg{Any, N};
    sparse = false, meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: VariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
} = _add_container!(container, T, U, JuMP.VariableRef, sparse, axs...; meta = meta)

function add_variable_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    meta::String,
    axs::Vararg{Any, N};
    sparse = false,
) where {
    T <: VariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    return _add_container!(container, T, U, JuMP.VariableRef, sparse, axs...; meta = meta)
end

function _get_pwl_variables_container()
    contents = Dict{Tuple{String, Int, Int}, JuMP.VariableRef}()
    return SparseAxisArray(contents)
end

function add_variable_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U};
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: SparseVariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    var_key = VariableKey(T, U, meta)
    _assign_container!(container.variables, var_key, _get_pwl_variables_container())
    return container.variables[var_key]
end

function get_variable_keys(container::OptimizationContainer)
    return collect(keys(container.variables))
end

get_variable(container::OptimizationContainer, key::VariableKey) =
    _get_entry(container, key)

get_variable(
    container::OptimizationContainer, ::Type{T}, ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {
    T <: VariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

##################################### AuxVariable Container ################################
add_aux_variable_container!(
    container::OptimizationContainer, ::Type{T}, ::Type{U}, axs::Vararg{Any, N};
    sparse = false, meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: AuxVariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
} = _add_container!(container, T, U, Float64, sparse, axs...; meta = meta)

function get_aux_variable_keys(container::OptimizationContainer)
    return collect(keys(container.aux_variables))
end

get_aux_variable(container::OptimizationContainer, key::AuxVarKey) =
    _get_entry(container, key)

get_aux_variable(
    container::OptimizationContainer, ::Type{T}, ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {
    T <: AuxVariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

##################################### DualVariable Container ################################
function add_dual_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs::Vararg{Any, N};
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ConstraintType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    if is_milp(container)
        @warn("The model has resulted in a MILP, \\
              dual value retrieval requires solving an additional Linear Program \\
              which increases simulation time and the outputs could be innacurate.")
    end
    const_key = ConstraintKey(T, U, meta)
    if sparse
        dual_container = sparse_container_spec(Float64, axs...)
    else
        dual_container = container_spec(Float64, axs...)
    end
    _assign_container!(container.duals, const_key, dual_container)
    return dual_container
end

function get_dual_keys(container::OptimizationContainer)
    return collect(keys(container.duals))
end

##################################### Constraint Container #################################
add_constraints_container!(
    container::OptimizationContainer, ::Type{T}, ::Type{U}, axs::Vararg{Any, N};
    sparse = false, meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ConstraintType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
} = _add_container!(container, T, U, JuMP.ConstraintRef, sparse, axs...; meta = meta)

function get_constraint_keys(container::OptimizationContainer)
    return collect(keys(container.constraints))
end

get_constraint(container::OptimizationContainer, key::ConstraintKey) =
    _get_entry(container, key)

get_constraint(
    container::OptimizationContainer, ::Type{T}, ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ConstraintType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_constraint(
    container::OptimizationContainer, ::T, ::Type{U},
    meta::String = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ConstraintType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

function read_duals(container::OptimizationContainer)
    return Dict(k => to_dataframe(jump_value.(v), k) for (k, v) in get_duals(container))
end

##################################### Parameter Container ##################################

"""
Determine the element type for parameter arrays: JuMP.VariableRef for recurrent solves,
Float64 otherwise.
"""
function get_param_eltype(container::OptimizationContainer)
    if built_for_recurrent_solves(container) && !get_rebuild_model(get_settings(container))
        return JuMP.VariableRef
    else
        return Float64
    end
end

"""
Allocate a parameter container where param and multiplier arrays share the same axes.
Covers VariableValue, Event, CostFunction parameter types.
"""
function add_param_container_shared_axes!(
    container::OptimizationContainer,
    key::ParameterKey,
    attribute::ParameterAttributes,
    param_type::DataType,
    axs::Vararg{Any, N};
    sparse = false,
) where {N}
    if sparse
        param_array = sparse_container_spec(param_type, axs...)
        multiplier_array = sparse_container_spec(Float64, axs...)
    else
        param_array = DenseAxisArray{param_type}(undef, axs...)
        multiplier_array = fill!(DenseAxisArray{Float64}(undef, axs...), NaN)
    end
    param_container = ParameterContainer(attribute, param_array, multiplier_array)
    _assign_container!(container.parameters, key, param_container)
    return param_container
end

"""
Allocate a parameter container where param and multiplier arrays have different axes.
Used for time series parameters where param_axs and multiplier_axs may differ.
"""
function add_param_container_split_axes!(
    container::OptimizationContainer,
    key::ParameterKey,
    attribute::ParameterAttributes,
    param_type::DataType,
    param_axs,
    multiplier_axs,
    additional_axs,
    time_steps::UnitRange{Int};
    sparse = false,
)
    if sparse
        param_array =
            sparse_container_spec(param_type, param_axs, additional_axs..., time_steps)
        multiplier_array =
            sparse_container_spec(Float64, multiplier_axs, additional_axs..., time_steps)
    else
        param_array =
            DenseAxisArray{param_type}(undef, param_axs, additional_axs..., time_steps)
        multiplier_array =
            fill!(
                DenseAxisArray{Float64}(
                    undef,
                    multiplier_axs,
                    additional_axs...,
                    time_steps,
                ),
                NaN,
            )
    end
    param_container = ParameterContainer(attribute, param_array, multiplier_array)
    _assign_container!(container.parameters, key, param_container)
    return param_container
end

function get_parameter_keys(container::OptimizationContainer)
    return collect(keys(container.parameters))
end

get_parameter(container::OptimizationContainer, key::ParameterKey) =
    _get_entry(container, key)

get_parameter(
    container::OptimizationContainer, ::Type{T}, ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

function get_parameter_array(container::OptimizationContainer, key)
    return get_parameter_array(get_parameter(container, key))
end

function get_parameter_array(
    container::OptimizationContainer,
    key::ParameterKey{T, U},
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_parameter_array(get_parameter(container, key))
end

function get_parameter_multiplier_array(
    container::OptimizationContainer,
    key::ParameterKey{T, U},
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_multiplier_array(get_parameter(container, key))
end

function get_parameter_attributes(
    container::OptimizationContainer,
    key::ParameterKey{T, U},
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_attributes(get_parameter(container, key))
end

function get_parameter_array(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_parameter_array(container, ParameterKey(T, U, meta))
end

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_parameter_array(
    container::OptimizationContainer, ::T, ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = get_parameter_array(container, T, U, meta)

function get_parameter_multiplier_array(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_multiplier_array(get_parameter(container, ParameterKey(T, U, meta)))
end

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_parameter_multiplier_array(
    container::OptimizationContainer, ::T, ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = get_parameter_multiplier_array(container, T, U, meta)

function get_parameter_attributes(
    container::OptimizationContainer,
    ::T,
    ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ParameterType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    return get_attributes(get_parameter(container, ParameterKey(T, U, meta)))
end

# Slow implementation not to be used in hot loops
function read_parameters(container::OptimizationContainer)
    params_dict = Dict{ParameterKey, DenseAxisArray}()
    parameters = get_parameters(container)
    (parameters === nothing || isempty(parameters)) && return params_dict
    for (k, v) in parameters
        # TODO: all functions similar to calculate_parameter_values should be in one
        # place and be consistent in behavior.
        #params_dict[k] = to_dataframe(calculate_parameter_values(v))
        param_array = get_parameter_values(v)
        multiplier_array = get_multiplier_array(v)
        params_dict[k] = _calculate_parameter_values(k, param_array, multiplier_array)
    end
    return params_dict
end

function _calculate_parameter_values(
    ::ParameterKey{<:ParameterType},
    param_array,
    multiplier_array,
)
    return param_array .* multiplier_array
end

function _calculate_parameter_values(
    ::ParameterKey{<:ObjectiveFunctionParameter},
    param_array,
    multiplier_array,
)
    return param_array
end
##################################### Expression Container #################################
function add_expression_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs::Vararg{Any, N};
    expr_type = GAE,
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ExpressionType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    expr_container =
        _add_container!(container, T, U, expr_type, sparse, axs...; meta = meta)
    remove_undef!(expr_container)
    return expr_container
end

# NOTE: add_expression_container! for ProductionCostExpression is in standard_variables_expressions.jl
# because it requires the type to be defined first

function get_expression_keys(container::OptimizationContainer)
    return collect(keys(container.expressions))
end

get_expression(container::OptimizationContainer, key::ExpressionKey) =
    _get_entry(container, key)

get_expression(
    container::OptimizationContainer, ::Type{T}, ::Type{U},
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: ExpressionType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
} = _get_entry(container, T, U, meta)

function read_expressions(container::OptimizationContainer)
    return Dict(
        k => to_dataframe(jump_value.(v), k) for (k, v) in get_expressions(container) if
        !(get_entry_type(k) <: SystemBalanceExpressions)
    )
end

###################################Initial Conditions Containers############################
function _add_initial_condition_container!(
    container::OptimizationContainer,
    ic_key::InitialConditionKey{T, U},
    length_devices::Int,
) where {
    T <: InitialConditionType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    if built_for_recurrent_solves(container) && !get_rebuild_model(get_settings(container))
        ini_conds = Vector{
            Union{InitialCondition{T, JuMP.VariableRef}, InitialCondition{T, Nothing}},
        }(
            undef,
            length_devices,
        )
    else
        ini_conds =
            Vector{Union{InitialCondition{T, Float64}, InitialCondition{T, Nothing}}}(
                undef,
                length_devices,
            )
    end
    _assign_container!(container.initial_conditions, ic_key, ini_conds)
    return ini_conds
end

function add_initial_condition_container!(
    container::OptimizationContainer,
    ::T,
    ::Type{U},
    axs;
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: InitialConditionType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    ic_key = InitialConditionKey(T, U, meta)
    @debug "add_initial_condition_container" ic_key _group = LOG_GROUP_SERVICE_CONSTUCTORS
    return _add_initial_condition_container!(container, ic_key, length(axs))
end

get_initial_condition(
    container::OptimizationContainer, ::Type{T}, ::Type{D},
) where {T <: InitialConditionType, D} = _get_entry(container, T, D)

# TODO: deprecate once POM is migrated to pass types (issue #18)
get_initial_condition(
    container::OptimizationContainer, ::T, ::Type{D},
) where {T <: InitialConditionType, D} = _get_entry(container, T, D)

get_initial_condition(container::OptimizationContainer, key::InitialConditionKey) =
    _get_entry(container, key)

function get_initial_conditions_keys(container::OptimizationContainer)
    return collect(keys(container.initial_conditions))
end

function write_initial_conditions_data!(
    container::OptimizationContainer,
    ic_container::OptimizationContainer,
)
    for field in STORE_CONTAINERS
        ic_container_dict = getfield(ic_container, field)
        if field == STORE_CONTAINER_PARAMETERS
            ic_container_dict = read_parameters(ic_container)
        end
        if field == STORE_CONTAINER_EXPRESSIONS
            continue
        end
        isempty(ic_container_dict) && continue
        ic_data_dict = getfield(get_initial_conditions_data(container), field)
        for (key, field_container) in ic_container_dict
            @debug "Adding $(encode_key_as_string(key)) to InitialConditionsData" _group =
                LOG_GROUP_SERVICE_CONSTUCTORS
            if field == STORE_CONTAINER_PARAMETERS
                ic_data_dict[key] = ic_container_dict[key]
            else
                ic_data_dict[key] = jump_value.(field_container)
            end
        end
    end
    return
end

# Note: These methods aren't passing the potential meta fields in the keys
function get_initial_conditions_variable(
    container::OptimizationContainer,
    type::VariableType,
    ::Type{T},
) where {T <: Union{PSY.Component, PSY.System}}
    return get_initial_conditions_variable(get_initial_conditions_data(container), type, T)
end

function get_initial_conditions_aux_variable(
    container::OptimizationContainer,
    type::AuxVariableType,
    ::Type{T},
) where {T <: Union{PSY.Component, PSY.System}}
    return get_initial_conditions_aux_variable(
        get_initial_conditions_data(container),
        type,
        T,
    )
end

function get_initial_conditions_dual(
    container::OptimizationContainer,
    type::ConstraintType,
    ::Type{T},
) where {T <: Union{PSY.Component, PSY.System}}
    return get_initial_conditions_dual(get_initial_conditions_data(container), type, T)
end

function get_initial_conditions_parameter(
    container::OptimizationContainer,
    type::ParameterType,
    ::Type{T},
) where {T <: Union{PSY.Component, PSY.System}}
    return get_initial_conditions_parameter(get_initial_conditions_data(container), type, T)
end

# When adding a QuadExpr to an AffExpr, += is needed because the AffExpr must be
# promoted to QuadExpr (reallocation). add_to_expression! cannot promote in-place.
function add_to_objective_invariant_expression!(
    container::OptimizationContainer,
    cost_expr::JuMP.GenericQuadExpr,
)
    container.objective_function.invariant_terms += cost_expr
    return
end

function add_to_objective_invariant_expression!(
    container::OptimizationContainer,
    cost_expr::JuMP.AbstractJuMPScalar,
)
    JuMP.add_to_expression!(container.objective_function.invariant_terms, cost_expr)
    return
end

function add_to_objective_invariant_expression!(
    container::OptimizationContainer,
    cost::Float64,
)
    JuMP.add_to_expression!(container.objective_function.invariant_terms, cost)
    return
end

function add_to_objective_variant_expression!(
    container::OptimizationContainer,
    cost_expr::JuMP.AffExpr,
)
    JuMP.add_to_expression!(container.objective_function.variant_terms, cost_expr)
    return
end

function deserialize_key(container::OptimizationContainer, name::AbstractString)
    return deserialize_key(container.metadata, name)
end

function _cache_aux_variable_key_partitions!(container::OptimizationContainer)
    aux_var_keys = keys(get_aux_variables(container))
    pf_keys = filter(is_from_power_flow ∘ get_entry_type, aux_var_keys)
    container.pf_aux_var_keys = collect(pf_keys)
    container.non_pf_aux_var_keys = collect(setdiff(aux_var_keys, pf_keys))
    return
end

function calculate_aux_variables!(
    container::OptimizationContainer,
    system::IS.InfrastructureSystemsContainer,
)
    if isempty(container.pf_aux_var_keys) && isempty(container.non_pf_aux_var_keys)
        _cache_aux_variable_key_partitions!(container)
    end
    pf_aux_var_keys = container.pf_aux_var_keys
    non_pf_aux_var_keys = container.non_pf_aux_var_keys
    # We should only have power flow aux vars if we have power flow evaluators
    @assert isempty(pf_aux_var_keys) || !isempty(get_power_flow_evaluation_data(container))

    TimerOutputs.@timeit RUN_SIMULATION_TIMER "Power Flow Evaluation" begin
        reset_power_flow_is_solved!(container)
        # Power flow-related aux vars get calculated once per power flow
        for (i, pf_e_data) in enumerate(get_power_flow_evaluation_data(container))
            @debug "Processing power flow $i"
            solve_powerflow!(pf_e_data, container)
            for key in pf_aux_var_keys
                calculate_aux_variable_value!(container, key, system)
            end
        end
    end

    # Other aux vars get calculated once at the end
    for key in non_pf_aux_var_keys
        calculate_aux_variable_value!(container, key, system)
    end
    return RunStatus.SUCCESSFULLY_FINALIZED
end

# NOTE: Commented out because it references CopperPlateBalanceConstraint concrete type
# This should be defined in PowerSimulations if needed
# function _calculate_dual_variable_value!(
#     container::OptimizationContainer,
#     key::ConstraintKey{CopperPlateBalanceConstraint, PSY.System},
#     ::PSY.System,
# )
#     constraint_container = get_constraint(container, key)
#     dual_variable_container = get_duals(container)[key]
#
#     for subnet in axes(constraint_container)[1], t in axes(constraint_container)[2]
#         # See https://jump.dev/JuMP.jl/stable/manual/solutions/#Dual-solution-values
#         dual_variable_container[subnet, t] = jump_value(constraint_container[subnet, t])
#     end
#     return
# end

function _calculate_dual_variable_value!(
    container::OptimizationContainer,
    key::ConstraintKey{T, D},
    ::PSY.System,
) where {T <: ConstraintType, D <: Union{PSY.Component, PSY.System}}
    constraint_duals = jump_value.(get_constraint(container, key))
    dual_variable_container = get_duals(container)[key]

    # Needs to loop since the container ordering might not match in the DenseAxisArray
    for index in Iterators.product(axes(constraint_duals)...)
        dual_variable_container[index...] = constraint_duals[index...]
    end

    return
end

function _calculate_dual_variables_continous_model!(
    container::OptimizationContainer,
    system::PSY.System,
)
    duals_vars = get_duals(container)
    for key in keys(duals_vars)
        _calculate_dual_variable_value!(container, key, system)
    end
    return RunStatus.SUCCESSFULLY_FINALIZED
end

function _calculate_dual_variables_discrete_model!(
    container::OptimizationContainer,
    ::PSY.System,
)
    return process_duals(container, container.settings.optimizer)
end

function calculate_dual_variables!(
    container::OptimizationContainer,
    sys::IS.InfrastructureSystemsContainer,
    is_milp::Bool,
)
    isempty(get_duals(container)) && return RunStatus.SUCCESSFULLY_FINALIZED
    if is_milp
        status = _calculate_dual_variables_discrete_model!(container, sys)
    else
        status = _calculate_dual_variables_continous_model!(container, sys)
    end
    return
end

########################### Helper Functions to get keys ###################################
# FIXME this 3-arg version is only called from POM. move it?
@generated function get_optimization_container_key(
    ::Type{T},
    ::Type{U},
    meta::String,
) where {
    T <: OptimizationKeyType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    K = key_for_type(T)
    return :($K(T, U, meta))
end

# note these 3 lazy_container_addition! definitions have different meta handling and adder 
# functions, else we'd collapse into one generated function.
function lazy_container_addition!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs::Vararg{Any, N};
    kwargs...,
) where {
    T <: VariableType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    if !has_container_key(container, T, U)
        var_container = add_variable_container!(container, T, U, axs...; kwargs...)
    else
        var_container = get_variable(container, T, U)
    end
    return var_container
end

function lazy_container_addition!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs::Vararg{Any, N};
    kwargs...,
) where {
    T <: ConstraintType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    meta = get(kwargs, :meta, CONTAINER_KEY_EMPTY_META)
    if !has_container_key(container, T, U, meta)
        cons_container =
            add_constraints_container!(container, T, U, axs...; kwargs...)
    else
        cons_container = get_constraint(container, T, U, meta)
    end
    return cons_container
end

function lazy_container_addition!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs::Vararg{Any, N};
    kwargs...,
) where {
    T <: ExpressionType,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
    N,
}
    meta = get(kwargs, :meta, CONTAINER_KEY_EMPTY_META)
    if !has_container_key(container, T, U, meta)
        expr_container =
            add_expression_container!(container, T, U, axs...; kwargs...)
    else
        expr_container = get_expression(container, T, U, meta)
    end
    return expr_container
end

function get_time_series_initial_values!(
    container::OptimizationContainer,
    ::Type{T},
    component::PSY.Component,
    time_series_name::AbstractString;
    interval::Dates.Millisecond = UNSET_INTERVAL,
    resolution::Dates.Millisecond = UNSET_RESOLUTION,
) where {T <: IS.TimeSeriesData}
    initial_time = get_initial_time(container)
    time_steps = get_time_steps(container)
    forecast = PSY.get_time_series(
        T,
        component,
        time_series_name;
        start_time = initial_time,
        count = 1,
        interval = _to_is_interval(interval),
        resolution = _to_is_resolution(resolution),
    )
    ts_values = IS.get_time_series_values(
        component,
        forecast;
        start_time = initial_time,
        len = length(time_steps),
        ignore_scaling_factors = true,
    )
    return ts_values
end

"""
Get the column names for the specified container in the OptimizationContainer.

# Arguments
- `container::OptimizationContainer`: The optimization container.
- `field::Symbol`: The field for which to retrieve the column names.
- `key::OptimizationContainerKey`: The key for which to retrieve the column names.

# Returns
- `Tuple`: Tuple of Vector{String}.
"""
function get_column_names(
    ::OptimizationContainer,
    field::Symbol,
    subcontainer,
    key::OptimizationContainerKey,
)
    return if field == :parameters
        # Parameters are stored in ParameterContainer.
        get_column_names(key, subcontainer)
    else
        # The others are in DenseAxisArrays.
        get_column_names_from_axis_array(key, subcontainer)
    end
end

lookup_value(
    container::OptimizationContainer,
    key::OptimizationContainerKey{T, U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType} =
    _get_entry(container, key)
# ParameterKey special case: unwrap ParameterContainer via calculate_parameter_values
lookup_value(
    container::OptimizationContainer,
    key::OptimizationContainerKey{T, U},
) where {T <: ParameterType, U <: InfrastructureSystemsType} =
    calculate_parameter_values(get_parameter(container, key))
