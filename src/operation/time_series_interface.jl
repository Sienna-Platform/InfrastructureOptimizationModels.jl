function get_time_series_values!(
    time_series_type::Type{T},
    model::DecisionModel,
    component,
    name::String,
    initial_time::Dates.DateTime,
    horizon::Int;
    interval::Dates.Millisecond = UNSET_INTERVAL,
) where {T <: IS.Forecast}
    is_interval = _to_is_interval(interval)
    settings = get_settings(model)
    resolution = get_resolution(settings)
    if !use_time_series_cache(settings)
        return IS.get_time_series_values(
            T,
            component,
            name;
            start_time = initial_time,
            len = horizon,
            interval = is_interval,
            resolution = resolution,
        )
    end

    cache = get_time_series_cache(model)
    key = IS.TimeSeriesCacheKey(IS.get_id(component), T, name, resolution, is_interval)
    if haskey(cache, key)
        ts_cache = cache[key]
    else
        ts_cache = IS.make_time_series_cache(
            time_series_type,
            component,
            name,
            initial_time,
            horizon;
            interval = is_interval,
            resolution = resolution,
        )
        cache[key] = ts_cache
    end

    ts = IS.get_time_series_array!(ts_cache, initial_time)
    return TimeSeries.values(ts)
end

function get_time_series_values!(
    ::Type{T},
    model::EmulationModel,
    component::U,
    name::String,
    initial_time::Dates.DateTime,
    len::Int = 1;
    resolution::Dates.Millisecond = UNSET_RESOLUTION,
) where {T <: IS.StaticTimeSeries, U <: IS.InfrastructureSystemsComponent}
    settings = get_settings(model)
    key_resolution =
        resolution == UNSET_RESOLUTION ? get_resolution(settings) : resolution
    is_resolution = _to_is_resolution(key_resolution)
    if !use_time_series_cache(settings)
        return IS.get_time_series_values(
            T,
            component,
            name;
            start_time = initial_time,
            len = len,
            resolution = is_resolution,
        )
    end

    cache = get_time_series_cache(model)
    key = IS.TimeSeriesCacheKey(IS.get_id(component), T, name, key_resolution, nothing)
    if haskey(cache, key)
        ts_cache = cache[key]
    else
        ts_cache = IS.make_time_series_cache(
            T,
            component,
            name,
            initial_time,
            len;
            resolution = is_resolution,
        )
        cache[key] = ts_cache
    end

    ts = IS.get_time_series_array!(ts_cache, initial_time)
    return TimeSeries.values(ts)
end

"""
The model's reader over every `T` series named `name` at the model resolution, built on
first use. A forecast reader serves a `DecisionModel` window at each step; a static reader
serves an `EmulationModel` value. `empty_time_series_cache!` drops the readers.
"""
function get_time_series_reader!(
    model::AbstractOptimizationModel,
    ::Type{T},
    name::String,
) where {T <: IS.TimeSeriesData}
    resolution = get_resolution(model)
    readers = get_time_series_readers(get_internal(model))
    # The readers are stored untyped; `T` fixes which reader comes back, so assert it.
    return get!(readers, (T, name, resolution)) do
        _build_time_series_reader(
            get_system(model),
            T,
            name,
            resolution,
        )
    end::_time_series_reader_type(T)
end

_time_series_reader_type(::Type{T}) where {T <: IS.Forecast} = IS.ForecastReader{T}
_time_series_reader_type(::Type{<:IS.StaticTimeSeries}) = IS.StaticTimeSeriesReader

_build_time_series_reader(sys, ::Type{T}, name, resolution) where {T <: IS.Forecast} =
    build_forecast_reader(sys, T; resolution = resolution, name = name)

_build_time_series_reader(sys, ::Type{<:IS.StaticTimeSeries}, name, resolution) =
    build_static_time_series_reader(sys; resolution = resolution, name = name)
