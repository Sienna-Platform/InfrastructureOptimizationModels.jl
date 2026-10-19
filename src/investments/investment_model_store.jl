mutable struct InvestmentModelStore <: AbstractModelStore
    # All DenseAxisArrays have axes (column names, row indexes)
    duals::Dict{ConstraintKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}
    variables::Dict{VariableKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}
    aux_variables::Dict{AuxVarKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}
    expressions::Dict{
        ExpressionKey,
        OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}},
    }
    optimizer_stats::OrderedDict{Dates.DateTime, OptimizerStats}
end

function InvestmentModelStore()
    return InvestmentModelStore(
        Dict{ConstraintKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}(),
        Dict{VariableKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}(),
        Dict{AuxVarKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}(),
        Dict{ExpressionKey, OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}}(),
        OrderedDict{Dates.DateTime, OptimizerStats}(),
    )
end

function initialize_storage!(
    store::InvestmentModelStore,
    container::AbstractOptimizationContainer,
    params::ModelStoreParams,
)
    time_mapping = get_time_mapping(container)
    if length(get_time_steps(time_mapping)) < 1
        error("The time step count in the optimization container is not defined")
    end
    base_timestamp = get_base_date(time_mapping)
    op_time_steps_count = get_total_operation_period_count(time_mapping)
    cap_time_steps_count = get_total_investment_period_count(time_mapping)
    for type in STORE_CONTAINERS
        if type == :parameters
            continue
        end
        field_containers = getfield(container, type)
        results_container = getfield(store, type)
        for (key, field_container) in field_containers
            !should_write_resulting_value(get_entry_type(key)) && continue
            entry_type = get_entry_type(key)
            if is_operation_entry(entry_type)
                count = op_time_steps_count
            elseif is_investment_entry(entry_type)
                count = cap_time_steps_count
            else
                error(
                    "Can't size storage for $(encode_key_as_string(key)): its entry type " *
                    "$entry_type is classified as neither an operations nor an " *
                    "investment entry.",
                )
            end
            @debug "Adding $(encode_key_as_string(key)) to InvestmentModelStore" _group =
                LOG_GROUP_MODEL_STORE
            column_names = get_column_names_from_axis_array(key, field_container)
            data = OrderedDict{Dates.DateTime, DenseAxisArray{Float64, 2}}()
            data[base_timestamp] =
                fill!(DenseAxisArray{Float64}(undef, column_names..., 1:count), NaN)
            results_container[key] = data
        end
    end
    return
end

function write_output!(
    store::InvestmentModelStore,
    name::Symbol,
    key::OptimizationContainerKey,
    index::Dates.DateTime,
    update_timestamp::Dates.DateTime,
    array::DenseAxisArray{<:Any, 2},
)
    columns = axes(array)[1]
    if eltype(columns) !== String
        # TODO: This happens because buses are stored by indexes instead of name.
        columns = string.(columns)
    end
    container = getfield(store, get_store_container_type(key))
    container[key][index] = DenseAxisArray(array.data, columns, 1:size(array)[2])
    return
end

# 1-D outputs are stored as a single row (named "1") with one column per entry.
function write_output!(
    store::InvestmentModelStore,
    name::Symbol,
    key::OptimizationContainerKey,
    index::Dates.DateTime,
    update_timestamp::Dates.DateTime,
    array::DenseAxisArray{<:Any, 1},
)
    columns = axes(array)[1]
    if eltype(columns) !== String
        # TODO: This happens because buses are stored by indexes instead of name.
        columns = string.(columns)
    end
    container = getfield(store, get_store_container_type(key))
    container[key][index] =
        DenseAxisArray(reshape(array.data, 1, length(columns)), ["1"], columns)
    return
end

# Entry types without a classification below (constraints/duals, aux variables, and
# variables or expressions outside the operations/investment hierarchies) error with a
# message naming the type.
is_operation_entry(::Type{T}) where {T <: OptimizationKeyType} =
    _unclassified_entry_error(T)
is_investment_entry(::Type{T}) where {T <: OptimizationKeyType} =
    _unclassified_entry_error(T)

is_operation_entry(::Type{<:OperationsExpressionType}) = true
is_operation_entry(::Type{<:InvestmentExpressionType}) = false

is_investment_entry(::Type{<:OperationsExpressionType}) = false
is_investment_entry(::Type{<:InvestmentExpressionType}) = true

is_operation_entry(::Type{<:OperationsVariableType}) = true
is_operation_entry(::Type{<:InvestmentVariableType}) = false

is_investment_entry(::Type{<:OperationsVariableType}) = false
is_investment_entry(::Type{<:InvestmentVariableType}) = true

function _unclassified_entry_error(::Type{T}) where {T <: VariableType}
    error(
        "Variable type $T is neither an OperationsVariableType nor an " *
        "InvestmentVariableType. Variables stored in an InvestmentModelStore must subtype " *
        "one of the two.",
    )
end

function _unclassified_entry_error(::Type{T}) where {T <: ExpressionType}
    error(
        "Expression type $T is neither an OperationsExpressionType nor an " *
        "InvestmentExpressionType. Expressions stored in an InvestmentModelStore must " *
        "subtype one of the two.",
    )
end

function _unclassified_entry_error(::Type{T}) where {T <: OptimizationKeyType}
    error(
        "$T outputs can't be stored in an InvestmentModelStore: it only stores variables " *
        "and expressions classified as operations or investment entries.",
    )
end
