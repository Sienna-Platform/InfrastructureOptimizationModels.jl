using DataStructures: SortedDict

struct TSInterfaceTemplate <: IOM.AbstractProblemTemplate end
IOM.finalize_template!(::TSInterfaceTemplate, args...) = nothing
IOM.get_network_model(::TSInterfaceTemplate) = nothing

struct TSInterfaceProblem <: IOM.AbstractOptimizationProblem end
IOM.validate_time_series!(::IOM.DecisionModel{TSInterfaceProblem}) = nothing
IOM.get_time_series_counts_by_type(::MockSystem) = [Dict("type" => "Deterministic")]

@testset "DecisionModel forecast read passes the model resolution" begin
    # The same forecast name exists at two resolutions.
    data = IS.SystemData()
    component = IS.TestComponent("c", 5)
    IS.add_component!(data, component)
    t0 = Dates.DateTime("2020-01-01")
    make(resolution, values) = IS.Deterministic(;
        name = "ts",
        resolution = resolution,
        data = SortedDict(
            t0 => values,
            t0 + Dates.Hour(1) => values,
        ),
    )
    IS.add_time_series!(data, component, make(Dates.Hour(1), [1.0, 2.0, 3.0, 4.0]))
    IS.add_time_series!(
        data,
        component,
        make(Dates.Minute(30), [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]),
    )

    model = IOM.DecisionModel{TSInterfaceProblem}(
        TSInterfaceTemplate(),
        MockSystem(100.0, false);
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    @test !IOM.use_time_series_cache(IOM.get_settings(model))
    values = IOM.get_time_series_values!(IS.Deterministic, model, component, "ts", t0, 4)
    @test values == [1.0, 2.0, 3.0, 4.0]
end

# The readers live in the system's data; the mock keeps it in its time series dict.
IOM.build_forecast_reader(sys::MockSystem, ::Type{T}; kwargs...) where {T} =
    IS.build_forecast_reader(sys.time_series[:data], T; kwargs...)
IOM.build_static_time_series_reader(sys::MockSystem; kwargs...) =
    IS.build_static_time_series_reader(sys.time_series[:data]; kwargs...)

@testset "A model builds each time series reader once and drops it with the cache" begin
    data = IS.SystemData()
    component = IS.TestComponent("c", 5)
    IS.add_component!(data, component)
    t0 = Dates.DateTime("2020-01-01")
    IS.add_time_series!(
        data,
        component,
        IS.SingleTimeSeries(;
            name = "ts",
            data = IOM.TS.TimeArray(t0 .+ Dates.Hour.(0:5), collect(1.0:6.0)),
        ),
    )
    IS.transform_single_time_series!(
        data,
        IS.DeterministicSingleTimeSeries,
        Dates.Hour(4),
        Dates.Hour(1),
    )
    sys = MockSystem(100.0, false)
    sys.time_series[:data] = data
    model = IOM.DecisionModel{TSInterfaceProblem}(
        TSInterfaceTemplate(),
        sys;
        resolution = Dates.Hour(1),
    )

    reader = @inferred IOM.get_time_series_reader!(
        model,
        IS.DeterministicSingleTimeSeries,
        "ts",
    )
    @test IOM.get_time_series_reader!(model, IS.DeterministicSingleTimeSeries, "ts") ===
          reader
    IS.read_forecast_window!(reader, t0 + Dates.Hour(1))
    @test IS.get_forecast_window(reader, 1) == [2.0, 3.0, 4.0, 5.0]

    static = @inferred IOM.get_time_series_reader!(model, IS.SingleTimeSeries, "ts")
    IS.read_static_time_series_values!(static, t0 + Dates.Hour(2))
    @test IS.get_static_time_series_value(static, 1) == 3.0

    IOM.empty_time_series_cache!(model)
    @test IOM.get_time_series_reader!(model, IS.DeterministicSingleTimeSeries, "ts") !==
          reader
end
