"""
Unit tests for EmulationModelStore: the generated list/get accessors must route
through `get_data_field` (Task 2.1) and empty!/isempty must operate on the
DatasetContainer fields (Task 2.10).
"""

@testset "EmulationModelStore accessors and empty!/isempty" begin
    store = IOM.EmulationModelStore()
    @test isempty(store)

    key = IOM.VariableKey(TestVariableType, MockComponentType)
    data = DenseAxisArray(zeros(2, 3), ["d1", "d2"], 1:3)
    IOM.set_dataset!(store.data_container, key, IOM.InMemoryDataset(data))

    @test !isempty(store)
    # Generated accessors previously called getfield(store, :variables), which
    # errors for EmulationModelStore (containers live inside data_container).
    @test IOM.list_keys(store, IOM.VariableType) == [key]
    @test collect(IOM.list_fields(store, IOM.VariableType)) == [key]
    @test IOM.get_value(store, TestVariableType, MockComponentType).values == data

    empty!(store)
    @test isempty(store)
    @test isempty(IOM.list_keys(store, IOM.VariableType))
end
