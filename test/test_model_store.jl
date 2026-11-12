struct MockDefaultStoreProblem <: IOM.AbstractOptimizationProblem end
struct MockCustomStoreProblem <: IOM.AbstractOptimizationProblem end

IOM.prepare_time_series!(
    ::Type{<:Union{MockDefaultStoreProblem, MockCustomStoreProblem}},
    ::MockSystem,
    ::IOM.Settings,
) =
    IS.SingleTimeSeries
IOM.validate_time_series!(
    ::IOM.DecisionModel{<:Union{MockDefaultStoreProblem, MockCustomStoreProblem}},
) = nothing

function IOM.initialize_storage!(model::IOM.DecisionModel{MockCustomStoreProblem})
    model.ext["custom_storage"] = true
    return
end

function _build_store_test_model(::Type{M}) where {M}
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    IOM.set_initial_time!(settings, Dates.DateTime(2030, 1, 1))
    model = IOM.DecisionModel{M}(build_mock_invest_template(), sys, settings)
    container = IOM.get_optimization_container(model)
    IOM.add_variable_container!(
        container,
        TestVariableType,
        MockThermalGen,
        ["a", "b"],
        1:1,
    )
    IOM.set_store_params!(
        IOM.get_internal(model),
        IOM.ModelStoreParams(
            1,
            1,
            Dates.Millisecond(Dates.Hour(1)),
            Dates.Millisecond(Dates.Hour(1)),
            1.0,
            IS.make_uuid(),
        ),
    )
    return model
end

@testset "initialize_storage!(model) dispatches on the problem type" begin
    key = IOM.VariableKey(TestVariableType, MockThermalGen)

    # Default: allocates through the store's own method.
    model = _build_store_test_model(MockDefaultStoreProblem)
    IOM.initialize_storage!(model)
    data = IOM.get_store(model).variables[key]
    @test collect(keys(data)) == [Dates.DateTime(2030, 1, 1)]
    @test size(data[Dates.DateTime(2030, 1, 1)]) == (2, 1)
    @test all(isnan, data[Dates.DateTime(2030, 1, 1)])

    # A problem type's method replaces the default.
    custom = _build_store_test_model(MockCustomStoreProblem)
    IOM.initialize_storage!(custom)
    @test custom.ext["custom_storage"]
    @test isempty(IOM.get_store(custom))
end

@testset "Test Model Store" begin
    @testset "DecisionModelStore round-trips a 1-D time-only result" begin
        store = IOM.DecisionModelStore()
        key = IOM.VariableKey(TestVariableType, MockComponentType)
        index = Dates.DateTime(2024, 1, 1)
        store.variables[key] = valtype(store.variables)()

        # A result indexed only by time (no component axis) is a 1-D UnitRange
        # DenseAxisArray; write_output! must handle it without a MethodError.
        array = DenseAxisArray([1.0, 2.0, 3.0], 1:3)
        IOM.write_output!(store, :test, key, index, index, array)

        result = IOM.read_outputs(store, key; index = index)
        @test result == array
        @test axes(result) == axes(array)
    end

    @testset "optimizer stats are read and written the same way for every store" begin
        stats = IOM.OptimizerStats()
        for (store, index) in (
            (IOM.DecisionModelStore(), Dates.DateTime(2024, 1, 1)),
            (IOM.EmulationModelStore(), 1),
        )
            IOM.write_optimizer_stats!(store, stats, index)
            df = IOM.read_optimizer_stats(store)
            @test size(df, 1) == 1
            # Only timestamp-indexed stores get a :DateTime column.
            @test (:DateTime in propertynames(df)) == !(store isa IOM.EmulationModelStore)
            # Rewriting an index warns and replaces the entry.
            @test_logs (:warn, r"Overwriting optimizer stats") IOM.write_optimizer_stats!(
                store,
                stats,
                index,
            )
            @test size(IOM.read_optimizer_stats(store), 1) == 1
        end
    end
end
