_nan(::Missing) = NaN
_nan(x::Real) = Float64(x)

const _STATS_COLUMNS = (
    (:objective_value, "f64", "Money", "USD"),
    (:termination_status, "i64", "", ""),
    (:primal_status, "i64", "", ""),
    (:dual_status, "i64", "", ""),
    (:solve_time, "f64", "Duration", "s"),
    (:solver_solve_time, "f64", "Duration", "s"),
    (:objective_bound, "f64", "Money", "USD"),
    (:relative_gap, "f64", "Fraction", "1"),
)

_stats_values(::Val{:f64}, stats::Vector, field::Symbol) =
    Float64[_nan(getfield(s, field)) for s in stats]
_stats_values(::Val{:i64}, stats::Vector, field::Symbol) =
    Int64[getfield(s, field) for s in stats]

function _stats_column(field::Symbol, dtype::String, kind::String, units::String)
    if isempty(kind)
        return IS.OutputColumn(string(field), dtype)
    end
    return IS.OutputColumn(string(field), dtype, kind, "NATURAL_UNITS", "", units)
end

"Return the `optimizer_stats` table description and its columns for `stats`."
function optimizer_stats_table(stats::AbstractDict, model::String)
    rows = collect(values(stats))
    names = Symbol[:execution]
    vectors = Any[collect(Int64, 1:length(rows))]
    for (field, dtype, _, _) in _STATS_COLUMNS
        push!(names, field)
        push!(vectors, _stats_values(Val(Symbol(dtype)), rows, field))
    end
    table = IS.OutputTable(
        "optimizer_stats",
        "/$model/tables/optimizer_stats",
        length(rows),
        vcat(
            [IS.OutputColumn("execution", "i64")],
            [_stats_column(spec...) for spec in _STATS_COLUMNS],
        ),
    )
    return (table, NamedTuple{Tuple(names)}(Tuple(vectors)))
end

"Return the arrays of `key` in `store`, one per execution, in execution order."
function store_arrays(store::DecisionModelStore, key::OptimizationContainerKey)
    container = get_data_field(store, get_store_container_type(key))
    return collect(values(container[key]))
end

"Return the array of `key` in `store`. The array holds every execution."
function store_arrays(store::EmulationModelStore, key::OptimizationContainerKey)
    container = get_data_field(store, get_store_container_type(key))
    return [container[key].values]
end

"Return the values of one execution in Julia order of the dataset axes, without `execution`."
function execution_slice(array::DenseAxisArray)
    return permutedims(array.data, reverse(1:ndims(array)))
end

function _dataset_values(d::IS.OutputDataset, values::AbstractArray)
    if isempty(d.component_type)
        return vec(values)
    end
    return values
end

function _store_keys(store::AbstractModelStore)
    keys_found = OptimizationContainerKey[]
    for type in STORE_CONTAINERS
        for key in Base.keys(get_data_field(store, type))
            if should_write_resulting_value(key)
                push!(keys_found, key)
            end
        end
    end
    return sort!(keys_found; by = encode_key_as_string)
end

function _trailing_axes(
    ::DecisionModelStore,
    key::OptimizationContainerKey,
    array::DenseAxisArray,
    params::ModelStoreParams,
)
    horizon = get_horizon_count(params)
    steps = length(axes(array)[end])
    if steps == horizon
        return ["horizon"]
    end
    if isone(steps)
        return String[]
    end
    throw(
        ArgumentError(
            "Key $(encode_key_as_string(key)) has $steps time steps, expected $horizon or 1.",
        ),
    )
end

_trailing_axes(::EmulationModelStore, ::OptimizationContainerKey, ::DenseAxisArray, ::Any) =
    String[]

_data_axes(::DecisionModelStore, components::Vector{String}, trailing::Vector{String}) =
    vcat(["execution"], components, trailing)
_data_axes(::EmulationModelStore, components::Vector{String}, ::Vector{String}) =
    vcat(["time"], components)

function _decision_values(d::IS.OutputDataset, array::DenseAxisArray)
    slice = execution_slice(array)
    if "horizon" in d.axes
        return _dataset_values(d, slice)
    end
    return dropdims(slice; dims = 1)
end

function _axis_lengths(reg::AxisRegistry, axes::Vector{String})
    return Int[IS.axis_length(reg.axes[a]) for a in axes]
end

function _key_dataset!(
    reg::AxisRegistry,
    store::AbstractModelStore,
    sys,
    key::OptimizationContainerKey,
    params::ModelStoreParams,
)
    array = first(store_arrays(store, key))
    labels = column_labels(key, array)
    trailing = _trailing_axes(store, key, array, params)
    if is_componentless(key, labels)
        axes = _data_axes(store, String[], trailing)
        return dataset_descriptor_componentless(
            key;
            model = reg.model,
            axes = axes,
            axis_lengths = _axis_lengths(reg, axes),
        )
    end
    components = dataset_axes!(reg, sys, key, labels, encode_key_as_string(key))
    axes = _data_axes(store, components, trailing)
    return dataset_descriptor(
        key;
        model = reg.model,
        axes = axes,
        axis_lengths = _axis_lengths(reg, axes),
    )
end

"""
Return the outputs metadata, the axis data, and the `(key, dataset)` pairs of `store`.
`table` is the `optimizer_stats` table description.
"""
function outputs_metadata(
    model::String,
    store::AbstractModelStore,
    params::ModelStoreParams,
    sys,
    settings::Settings,
    producer::IS.ProducerInfo,
    initial_time::Dates.DateTime,
    table::IS.OutputTable,
)
    reg = AxisRegistry(model_axes(store, params, initial_time, model)..., model)
    datasets = Tuple{OptimizationContainerKey, IS.OutputDataset}[]
    for key in _store_keys(store)
        push!(datasets, (key, _key_dataset!(reg, store, sys, key, params)))
    end
    uuid = get_system_uuid(params)
    entry = IS.ModelEntry(
        model_type_name(store),
        string(uuid),
        get_base_power(params),
        settings_dict(settings),
        reg.axes,
        IS.OutputDataset[d for (_, d) in datasets],
        [table],
    )
    metadata = IS.OutputsMetadata(
        IS._SCHEMA_VERSION,
        "single",
        producer,
        Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z",
        "RUNNING",
        Dict(string(uuid) => make_system_dirname(uuid)),
        Dict(model => entry),
        nothing,
    )
    IS.validate_metadata(metadata)
    return (metadata, reg.data, datasets)
end

_time_last(data::AbstractMatrix) = data

function _time_last(data::AbstractArray{T, N}) where {T, N}
    return permutedims(data, (reverse(1:(N - 1))..., N))
end

function write_executions!(
    w::IS.OutputsWriter,
    model::String,
    store::DecisionModelStore,
    datasets,
    ::ModelStoreParams,
)
    for (key, d) in datasets
        for (i, array) in enumerate(store_arrays(store, key))
            IS.append!(
                w,
                model,
                IS.dataset_key(d),
                _decision_values(d, array),
                i,
            )
        end
    end
    flush(w)
    return
end

function write_executions!(
    w::IS.OutputsWriter,
    model::String,
    store::EmulationModelStore,
    datasets,
    ::ModelStoreParams,
)
    for (key, d) in datasets
        array = only(store_arrays(store, key))
        IS.write_values!(
            w,
            model,
            IS.dataset_key(d),
            _dataset_values(d, _time_last(array.data)),
        )
    end
    flush(w)
    return
end

"""
Write the outputs artifact of `store` to `dir`. Remove a previous artifact in `dir` first.
Return `dir`.
"""
function write_store_outputs!(
    dir::AbstractString,
    model::String,
    store::AbstractModelStore,
    params::ModelStoreParams,
    sys,
    settings::Settings,
    producer::IS.ProducerInfo,
    initial_time::Dates.DateTime,
)
    rm(joinpath(dir, "outputs.json"); force = true)
    rm(joinpath(dir, "outputs.h5"); force = true)
    (table, columns) = optimizer_stats_table(store.optimizer_stats, model)
    (metadata, axis_data, datasets) = outputs_metadata(
        model, store, params, sys, settings, producer, initial_time, table,
    )
    w = IS.open_outputs_writer(dir, metadata; axis_data = axis_data)
    try
        write_executions!(w, model, store, datasets, params)
        IS.write_table!(w, model, "optimizer_stats", columns)
        IS.finalize!(w; status = IS.Simulation.RunStatus.SUCCESSFULLY_FINALIZED)
    catch
        IS._abort!(w)
        rethrow()
    end
    return dir
end

"Write the outputs artifact of `model` to its output directory. Return the directory."
function write_model_outputs!(model::AbstractOptimizationModel, producer::IS.ProducerInfo)
    return write_store_outputs!(
        get_output_dir(model),
        string(get_name(model)),
        get_store(model),
        get_store_params(model),
        get_system(model),
        get_settings(model),
        producer,
        get_initial_time(model),
    )
end
