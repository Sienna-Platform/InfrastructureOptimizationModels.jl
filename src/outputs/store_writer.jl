_nan(::Missing) = NaN
_nan(x::Real) = Float64(x)

_float_column(stats, field::Symbol) = Float64[_nan(getfield(s, field)) for s in stats]
_int_column(stats, field::Symbol) = Int64[getfield(s, field) for s in stats]

_seconds_column(name::String) =
    IS.OutputColumn(name, "f64", "Duration", "NATURAL_UNITS", "", "s")

# The objective has no unit: IOM does not know the currency of the model.
"Return the `optimizer_stats` table description and its columns for `stats`."
function optimizer_stats_table(stats::AbstractDict, model::String)
    rows = collect(values(stats))
    table = IS.OutputTable(
        "optimizer_stats",
        "/$model/tables/optimizer_stats",
        length(rows),
        [
            IS.OutputColumn("execution", "i64"),
            IS.OutputColumn("objective_value", "f64"),
            IS.OutputColumn("termination_status", "i64"),
            IS.OutputColumn("primal_status", "i64"),
            IS.OutputColumn("dual_status", "i64"),
            _seconds_column("solve_time"),
            _seconds_column("solver_solve_time"),
            IS.OutputColumn("objective_bound", "f64"),
            IS.OutputColumn("relative_gap", "f64", "Fraction", "NATURAL_UNITS", "", "1"),
        ],
    )
    columns = (
        execution = collect(Int64, 1:length(rows)),
        objective_value = _float_column(rows, :objective_value),
        termination_status = _int_column(rows, :termination_status),
        primal_status = _int_column(rows, :primal_status),
        dual_status = _int_column(rows, :dual_status),
        solve_time = _float_column(rows, :solve_time),
        solver_solve_time = _float_column(rows, :solver_solve_time),
        objective_bound = _float_column(rows, :objective_bound),
        relative_gap = _float_column(rows, :relative_gap),
    )
    return (table, columns)
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
    skipped = String[]
    for type in STORE_CONTAINERS
        for key in Base.keys(get_data_field(store, type))
            if writes_output_dataset(key)
                push!(keys_found, key)
            else
                push!(skipped, encode_key_as_string(key))
            end
        end
    end
    if !isempty(skipped)
        @info "Outputs artifact skips keys without a schema quantity kind: $(join(sort!(skipped), ", "))"
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
        IS.OUTPUTS_SCHEMA_VERSION,
        "single",
        producer,
        Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z",
        "RUNNING",
        Dict(string(uuid) => make_system_dirname(uuid)),
        Dict(model => entry),
        IS.NoSimulation(),
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
        IS.abort!(w)
        rethrow()
    end
    return dir
end

"""
Write the outputs artifact of `model` to its output directory. Return the directory.
The caller must write the system bundle at `joinpath(dir, make_system_dirname(uuid))` first.
`system_to_file` does not apply: the artifact always declares the bundle.
"""
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
