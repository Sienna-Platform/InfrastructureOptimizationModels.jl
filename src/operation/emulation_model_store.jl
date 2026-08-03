"""
Stores outputs data for one EmulationModel.
Parameterized by `T <: AbstractDataset` to support different storage backends
(e.g., `InMemoryDataset`, `HDF5Dataset`).
"""
mutable struct EmulationModelStore{T <: AbstractDataset} <: AbstractModelStore
    data_container::DatasetContainer{T}
    optimizer_stats::OrderedDict{Int, OptimizerStats}
end

get_data_field(store::EmulationModelStore, ::Val{S}) where {S} =
    getfield(store.data_container, S)
@inline Base.@constprop :aggressive get_data_field(
    store::EmulationModelStore,
    type::Symbol,
) =
    get_data_field(store, Val(type))

function EmulationModelStore()
    return EmulationModelStore(
        DatasetContainer{InMemoryDataset}(),
        OrderedDict{Int, OptimizerStats}(),
    )
end

"""
    Base.empty!(store::EmulationModelStore)

Empty the [`EmulationModelStore`](@ref)
"""
function Base.empty!(store::EmulationModelStore)
    for name in fieldnames(DatasetContainer)
        empty!(get_data_field(store, name))
    end
    empty!(store.optimizer_stats)
    return
end

function Base.isempty(store::EmulationModelStore)
    for name in fieldnames(DatasetContainer)
        !isempty(get_data_field(store, name)) && return false
    end
    return isempty(store.optimizer_stats)
end

function initialize_storage!(
    store::EmulationModelStore{InMemoryDataset},
    container::OptimizationContainer,
    params::ModelStoreParams,
)
    num_of_executions = get_num_executions(params)
    for type in STORE_CONTAINERS
        field_containers = getfield(container, type)
        outputs_container = get_data_field(store, type)
        for (key, field_container) in field_containers
            @debug "Adding $(encode_key_as_string(key)) to EmulationModelStore" _group =
                LOG_GROUP_MODEL_STORE
            column_names = get_column_names(container, type, field_container, key)
            outputs_container[key] = InMemoryDataset(
                fill!(
                    DenseAxisArray{Float64}(undef, column_names..., 1:num_of_executions),
                    NaN,
                ),
            )
        end
    end
    return
end

function write_output!(
    store::EmulationModelStore,
    name::Symbol,
    key::OptimizationContainerKey,
    index::EmulationModelIndexType,
    update_timestamp::Dates.DateTime,
    array::DenseAxisArray{Float64, 2},
)
    if size(array, 2) == 1
        write_output!(store, name, key, index, update_timestamp, array[:, 1])
    else
        container = get_data_field(store, get_store_container_type(key))
        set_value!(
            container[key],
            array,
            index,
        )
        set_last_recorded_row!(container[key], index)
        set_update_timestamp!(container[key], update_timestamp)
    end
    return
end

function write_output!(
    store::EmulationModelStore,
    ::Symbol,
    key::OptimizationContainerKey,
    index::EmulationModelIndexType,
    update_timestamp::Dates.DateTime,
    array::DenseAxisArray{Float64, 1},
)
    container = get_data_field(store, get_store_container_type(key))
    set_value!(
        container[key],
        array,
        index,
    )
    set_last_recorded_row!(container[key], index)
    set_update_timestamp!(container[key], update_timestamp)
    return
end

# Sparse containers (e.g. reserve variables keyed `(service, device, t)`) are stored as
# 2D dense with the non-time tuple flattened into encoded `"a__b"` columns, matching the
# storage `initialize_storage!` allocated via `get_column_names_from_axis_array`. Columns
# are ordered by their encoded string so the flattened matrix aligns with the
# pre-allocated dataset's rows (`set_value!` copies positionally).
function write_output!(
    store::EmulationModelStore,
    name::Symbol,
    key::OptimizationContainerKey,
    index::EmulationModelIndexType,
    update_timestamp::Dates.DateTime,
    array::SparseAxisArray{T, N, K},
) where {T, N, K <: NTuple{N, Any}}
    tuple_columns = unique!([k[1:(N - 1)] for k in keys(array.data)])
    sort!(tuple_columns; by = encode_tuple_to_column)
    matrix = _to_matrix(array, tuple_columns)
    columns = encode_tuple_to_column.(tuple_columns)
    dense = DenseAxisArray(permutedims(matrix), columns, 1:size(matrix, 1))
    write_output!(store, name, key, index, update_timestamp, dense)
    return
end

function read_outputs(
    store::EmulationModelStore{InMemoryDataset},
    key::OptimizationContainerKey;
    index::Union{Int, Nothing} = nothing,
    len::Union{Int, Nothing} = nothing,
)
    container = get_data_field(store, get_store_container_type(key))
    data = container[key].values
    # Return a copy because callers may mutate it.
    if isnothing(index)
        @assert_op len === nothing
        return data[:, :]
    elseif isnothing(len)
        return data[:, index:end]
    else
        return data[:, index:(index + len - 1)]
    end
end

function get_column_names(
    store::EmulationModelStore{InMemoryDataset},
    key::OptimizationContainerKey,
)
    container = get_data_field(store, get_store_container_type(key))
    return get_column_names_from_axis_array(key, container[key].values)
end

function get_dataset_size(store::EmulationModelStore, key::OptimizationContainerKey)
    container = get_data_field(store, get_store_container_type(key))
    return size(container[key].values)
end

function get_last_updated_timestamp(
    store::EmulationModelStore,
    key::OptimizationContainerKey,
)
    container = get_data_field(store, get_store_container_type(key))
    return get_update_timestamp(container[key])
end
function write_optimizer_stats!(
    store::EmulationModelStore,
    stats::OptimizerStats,
    index::EmulationModelIndexType,
)
    @assert !(index in keys(store.optimizer_stats))
    store.optimizer_stats[index] = stats
    return
end

function read_optimizer_stats(store::EmulationModelStore)
    return DataFrames.DataFrame([
        IS.to_namedtuple(x) for x in values(store.optimizer_stats)
    ])
end

function get_last_recorded_row(x::EmulationModelStore, key::OptimizationContainerKey)
    return get_last_recorded_row(x.data_container, key)
end
