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
            (IOM.InvestmentModelStore(), Dates.Date(2030, 1, 1)),
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
