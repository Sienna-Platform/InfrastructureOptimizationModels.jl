"""
Unit tests for EmulationModelStore: the generated list/get accessors must route
through `get_data_field` (Task 2.1) and empty!/isempty must operate on the
DatasetContainer fields (Task 2.10).
"""

@testset "EmulationModelStore accessors and empty!/isempty" begin
    store = PSI.EmulationModelStore()
    @test isempty(store)

    key = PSI.VariableKey(TestVariableType, MockComponentType)
    data = DenseAxisArray(zeros(2, 3), ["d1", "d2"], 1:3)
    PSI.set_dataset!(store.data_container, key, PSI.InMemoryDataset(data))

    @test !isempty(store)
    # Generated accessors previously called getfield(store, :variables), which
    # errors for EmulationModelStore (containers live inside data_container).
    @test PSI.list_keys(store, PSI.VariableType) == [key]
    @test collect(PSI.list_fields(store, PSI.VariableType)) == [key]
    @test PSI.get_value(store, TestVariableType, MockComponentType).values == data

    empty!(store)
    @test isempty(store)
    @test isempty(PSI.list_keys(store, PSI.VariableType))
end
