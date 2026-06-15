mutable struct OutputsByTime{T, N}
    key::OptimizationContainerKey
    data::SortedDict{Dates.DateTime, T}
    # Concrete field; everything upstream already works in Millisecond.
    resolution::Dates.Millisecond
    column_names::NTuple{N, Vector{String}}

    function OutputsByTime(
        key::OptimizationContainerKey,
        data::SortedDict{Dates.DateTime, T},
        resolution::Dates.Period,
        column_names::NTuple{N, Vector{String}},
    ) where {T, N}
        _check_column_consistency(data, column_names)
        return new{T, N}(key, data, Dates.Millisecond(resolution), column_names)
    end
end

function _check_column_consistency(
    data::SortedDict{Dates.DateTime, DenseAxisArray{Float64, 2}},
    cols::Tuple{Vector{String}},
)
    for val in values(data)
        if axes(val)[1] != cols[1]
            error("Mismatch in DenseAxisArray column names: $(axes(val)[1]) $cols")
        end
    end
end

function _check_column_consistency(
    data::SortedDict{Dates.DateTime, Matrix{Float64}},
    cols::Tuple{Vector{String}},
)
    for val in values(data)
        if size(val)[2] != length(cols[1])
            error(
                "Mismatch in length of Matrix columns: $(size(val)[2]) $(length(cols[1]))",
            )
        end
    end
end

function _check_column_consistency(
    data::SortedDict{Dates.DateTime, DenseAxisArray{Float64, 2}},
    cols::NTuple{N, Vector{String}},
) where {N}
    for val in values(data)
        for (i, col) in enumerate(cols)
            if axes(val)[i] != col
                error(
                    "Mismatch in DenseAxisArray axis $i column names: $(axes(val)[i]) $col",
                )
            end
        end
    end
end

function _check_column_consistency(
    data::SortedDict{Dates.DateTime, DataFrame},
    cols::NTuple{N, Vector{String}},
) where {N}
    for df in values(data)
        if DataFrames.ncol(df) != length(cols[1])
            error(
                "Mismatch in length of DataFrame columns: $(DataFrames.ncol(df)) $(length(cols[1]))",
            )
        end
    end
end

# This struct behaves like a dict, delegating to its 'data' field.
Base.length(res::OutputsByTime) = length(res.data)
Base.iterate(res::OutputsByTime) = iterate(res.data)
Base.iterate(res::OutputsByTime, state) = iterate(res.data, state)
Base.getindex(res::OutputsByTime, i) = getindex(res.data, i)
Base.setindex!(res::OutputsByTime, v, i) = setindex!(res.data, v, i)
Base.firstindex(res::OutputsByTime) = firstindex(res.data)
Base.lastindex(res::OutputsByTime) = lastindex(res.data)

get_column_names(x::OutputsByTime) = x.column_names
get_num_rows(::OutputsByTime{DenseAxisArray{Float64, 2}}, data) = size(data, 2)
get_num_rows(::OutputsByTime{DenseAxisArray{Float64, 3}}, data) = size(data, 3)
get_num_rows(::OutputsByTime{Matrix{Float64}}, data) = size(data, 1)
get_num_rows(::OutputsByTime{DataFrame}, data) = DataFrames.nrow(data)

function _add_timestamps!(
    df::DataFrames.DataFrame,
    outputs::OutputsByTime,
    timestamp::Dates.DateTime,
    data,
)
    time_col = _get_timestamps(outputs, timestamp, get_num_rows(outputs, data))
    if !isnothing(time_col)
        DataFrames.insertcols!(df, 1, :DateTime => time_col)
    end
    return
end

function _get_timestamps(outputs::OutputsByTime, timestamp::Dates.DateTime, len::Int)
    if outputs.resolution == Dates.Millisecond(0)
        return
    end
    return range(timestamp; length = len, step = outputs.resolution)
end

function make_dataframe(
    outputs::OutputsByTime{DenseAxisArray{Float64, 2}},
    timestamp::Dates.DateTime;
    table_format::TableFormat = TableFormat.LONG,
)
    array = outputs.data[timestamp]
    timestamps = _get_timestamps(outputs, timestamp, get_num_rows(outputs, array))
    return to_outputs_dataframe(array, timestamps, Val(table_format))
end

function make_dataframe(
    outputs::OutputsByTime{DenseAxisArray{Float64, 3}},
    timestamp::Dates.DateTime;
    table_format::TableFormat = TableFormat.LONG,
)
    array = outputs.data[timestamp]
    num_timestamps = get_num_rows(outputs, array)
    timestamps = _get_timestamps(outputs, timestamp, num_timestamps)
    return to_outputs_dataframe(array, timestamps, Val(table_format))
end

function make_dataframe(
    outputs::OutputsByTime{Matrix{Float64}},
    timestamp::Dates.DateTime;
    table_format::TableFormat = TableFormat.LONG,
)
    array = outputs.data[timestamp]
    df_wide = DataFrames.DataFrame(array, outputs.column_names[1])
    _add_timestamps!(df_wide, outputs, timestamp, array)
    return if table_format == TableFormat.LONG
        measure_vars = [x for x in names(df_wide) if x != "DateTime"]
        DataFrames.stack(
            df_wide,
            measure_vars;
            variable_name = :name,
            value_name = :value,
        )
    elseif table_format == TableFormat.WIDE
        df_wide
    else
        error("Unsupported table format: $table_format")
    end
end

function make_dataframes(
    outputs::OutputsByTime;
    table_format::TableFormat = TableFormat.LONG,
)
    return SortedDict(
        k => make_dataframe(outputs, k; table_format = table_format) for
        k in keys(outputs.data)
    )
end

struct OutputsByKeyAndTime
    "Contains all keys stored in the model."
    output_keys::Vector{OptimizationContainerKey}
    "Contains the outputs that have been read from the store and cached."
    cached_outputs::Dict{OptimizationContainerKey, OutputsByTime}
end

OutputsByKeyAndTime(output_keys) = OutputsByKeyAndTime(
    collect(output_keys),
    Dict{OptimizationContainerKey, OutputsByTime}(),
)

Base.empty!(res::OutputsByKeyAndTime) = empty!(res.cached_outputs)
