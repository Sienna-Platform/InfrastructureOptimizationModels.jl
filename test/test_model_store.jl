@testset "Test Model Store" begin
    @testset "DecisionModelStore round-trips a 1-D time-only result" begin
        store = PSI.DecisionModelStore()
        key = PSI.VariableKey(TestVariableType, MockComponentType)
        index = Dates.DateTime(2024, 1, 1)
        store.variables[key] = valtype(store.variables)()

        # A result indexed only by time (no component axis) is a 1-D UnitRange
        # DenseAxisArray; write_output! must handle it without a MethodError.
        array = DenseAxisArray([1.0, 2.0, 3.0], 1:3)
        PSI.write_output!(store, :test, key, index, index, array)

        result = PSI.read_outputs(store, key; index = index)
        @test result == array
        @test axes(result) == axes(array)
    end
end
