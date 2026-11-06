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

    @testset "get_value infers a concrete type on both stores" begin
        decision_store = IOM.DecisionModelStore()
        key = IOM.VariableKey(TestVariableType, MockComponentType)
        decision_store.variables[key] = valtype(decision_store.variables)()
        @test (@inferred IOM.get_value(
            decision_store,
            TestVariableType,
            MockComponentType,
        )) ===
              decision_store.variables[key]

        emulation_store = IOM.EmulationModelStore()
        data = DenseAxisArray(zeros(2, 3), ["d1", "d2"], 1:3)
        IOM.set_dataset!(emulation_store.data_container, key, IOM.InMemoryDataset(data))
        @test (@inferred IOM.InMemoryDataset IOM.get_value(
            emulation_store,
            TestVariableType,
            MockComponentType,
        )).values ==
              data
    end
end
