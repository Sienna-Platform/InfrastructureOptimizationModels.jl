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

@testset "EmulationModelStore sparse 3D container write/read round trip" begin
    # Reserve-style containers are sparse and keyed `(service, device, time)`; the store
    # flattens the leading dims to encoded `"service__device"` columns. This mirrors the
    # DecisionModelStore path but the EmulationModelStore write must line the flattened
    # rows up with the pre-allocated dataset (`set_value!` copies positionally).
    store = IOM.EmulationModelStore()
    key = IOM.VariableKey(TestVariableType, MockComponentType)
    values = Dict(
        ("s1", "d1", 1) => 1.0,
        ("s1", "d2", 1) => 2.0,
        ("s2", "d1", 1) => 3.0,
    )
    sparse = JuMP.Containers.SparseAxisArray(values)
    # Storage is pre-allocated exactly as initialize_storage! would, from the encoded
    # column names.
    cols = IOM.get_column_names_from_axis_array(key, sparse)[1]
    storage = DenseAxisArray(fill(NaN, length(cols), 1), cols, 1:1)
    IOM.set_dataset!(store.data_container, key, IOM.InMemoryDataset(storage))

    IOM.write_output!(store, :variables, key, 1, Dates.DateTime(2024, 1, 1), sparse)
    out = IOM.read_outputs(store, key)

    @test out["s1__d1", 1] == 1.0
    @test out["s1__d2", 1] == 2.0
    @test out["s2__d1", 1] == 3.0
end
