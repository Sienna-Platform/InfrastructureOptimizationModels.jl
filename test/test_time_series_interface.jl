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
