"""
Unit tests for JuMP utility functions in jump_utils.jl.
Tests array conversions, DataFrame creation, and JuMP model utilities.
Written entirely by Claude Code, so fairly surface-level, but serves its purpose of 
confirming that the code runs without errors.
"""
# Mock types for testing
struct MockVariableType <: ISOPT.VariableType end

@testset "JuMP Utilities" begin
    @testset "add_jump_parameter" begin
        model = JuMP.Model()
        param = IOM.add_jump_parameter(model, 42.0)
        @test JuMP.is_fixed(param)
        @test JuMP.fix_value(param) == 42.0

        # Test with different values
        param2 = IOM.add_jump_parameter(model, -3.14)
        @test JuMP.fix_value(param2) == -3.14

        param3 = IOM.add_jump_parameter(model, 0.0)
        @test JuMP.fix_value(param3) == 0.0
    end

    @testset "jump_value" begin
        model = JuMP.Model()

        # Test fixed variable
        @variable(model, x)
        JuMP.fix(x, 5.0; force = true)
        @test IOM.jump_value(x) == 5.0

        # Test literal passthrough
        @test IOM.jump_value(3.14) == 3.14
        @test IOM.jump_value(42) == 42

        # Test unfixed variable without solution returns NaN
        @variable(model, y)
        @test isnan(IOM.jump_value(y))
    end

    @testset "jump_fixed_value" begin
        model = JuMP.Model()

        # Test number passthrough
        @test IOM.jump_fixed_value(5.0) == 5.0
        @test IOM.jump_fixed_value(42) == 42

        # Test fixed variable
        @variable(model, x)
        JuMP.fix(x, 3.0; force = true)
        @test IOM.jump_fixed_value(x) == 3.0

        # Test affine expression with fixed variables
        @variable(model, y)
        JuMP.fix(y, 2.0; force = true)
        expr = 2.0 * x + 3.0 * y + 1.0
        @test IOM.jump_fixed_value(expr) == 2.0 * 3.0 + 3.0 * 2.0 + 1.0  # = 13.0
    end

    @testset "fix_parameter_value" begin
        model = JuMP.Model()
        @variable(model, x)
        JuMP.fix(x, 1.0; force = true)

        IOM.fix_parameter_value(x, 99.0)
        @test JuMP.fix_value(x) == 99.0
    end

    @testset "to_matrix - Vector" begin
        vec = [1.0, 2.0, 3.0]
        mat = IOM.to_matrix(vec)
        @test size(mat) == (3, 1)
        @test mat[:, 1] == vec
    end

    @testset "to_matrix - Matrix passthrough" begin
        mat_in = [1.0 2.0; 3.0 4.0]
        mat_out = IOM.to_matrix(mat_in)
        @test mat_out === mat_in
    end

    @testset "to_matrix - DenseAxisArray 1D" begin
        data = DenseAxisArray([1.0, 2.0, 3.0], ["a", "b", "c"])
        mat = IOM.to_matrix(data)
        @test size(mat) == (3, 1)
        @test mat[:, 1] == [1.0, 2.0, 3.0]
    end

    @testset "to_matrix - DenseAxisArray 2D" begin
        data = DenseAxisArray(
            [1.0 2.0 3.0; 4.0 5.0 6.0],
            ["a", "b"],
            1:3,
        )
        mat = IOM.to_matrix(data)
        # permutedims transposes the data
        @test size(mat) == (3, 2)
        @test mat[1, :] == [1.0, 4.0]
        @test mat[2, :] == [2.0, 5.0]
    end

    @testset "to_matrix - SparseAxisArray" begin
        contents = Dict(
            ("gen1", 1) => 1.0,
            ("gen1", 2) => 2.0,
            ("gen2", 1) => 3.0,
            ("gen2", 2) => 4.0,
        )
        sparse = SparseAxisArray(contents)
        mat = IOM.to_matrix(sparse)
        @test size(mat) == (2, 2)
    end

    @testset "to_dataframe - SparseAxisArray mixed-type column labels" begin
        # Regression: matrix columns were ordered by raw tuple while labels were ordered
        # by encoded string. For mixed-type tuples these diverge — `("a",2) < ("a",10)`
        # as tuples but `"a__10" < "a__2"` lexically — silently mislabeling exports.
        key = IOM.ConstraintKey(TestConstraintType, IS.TestComponent)
        contents = Dict(
            ("a", 2, 1) => 10.0,
            ("a", 10, 1) => 20.0,
        )
        sparse = SparseAxisArray(contents)
        df = IOM.to_dataframe(sparse, key)
        # Each value must sit under the label encoding its own key.
        @test df[1, "a__2"] == 10.0
        @test df[1, "a__10"] == 20.0
    end

    @testset "to_dataframe - SparseAxisArray non-contiguous time keys" begin
        # Regression: `_to_matrix` indexed rows by the raw time value, so time keys that
        # are not `1:n` (here {2, 4}) overran the matrix and threw.
        key = IOM.ConstraintKey(TestConstraintType, IS.TestComponent)
        contents = Dict(
            ("g", 2) => 1.0,
            ("g", 4) => 2.0,
        )
        sparse = SparseAxisArray(contents)
        df = IOM.to_dataframe(sparse, key)
        @test size(df) == (2, 1)
        @test df[!, "g"] == [1.0, 2.0]
    end

    @testset "get_column_names_from_axis_array - 1D String axis" begin
        data = DenseAxisArray([1.0, 2.0], ["gen1", "gen2"])
        cols = IOM.get_column_names_from_axis_array(data)
        @test cols == (["gen1", "gen2"],)
    end

    @testset "get_column_names_from_axis_array - 1D Int axis" begin
        data = DenseAxisArray([1.0, 2.0, 3.0], 1:3)
        cols = IOM.get_column_names_from_axis_array(data)
        @test cols == (["1", "2", "3"],)
    end

    @testset "get_column_names_from_axis_array - 2D" begin
        data = DenseAxisArray(
            [1.0 2.0; 3.0 4.0],
            ["gen1", "gen2"],
            1:2,
        )
        cols = IOM.get_column_names_from_axis_array(data)
        @test cols == (["gen1", "gen2"],)
    end

    @testset "encode_tuple_to_column" begin
        @test IOM.encode_tuple_to_column(("a", "b")) == "a__b"
        @test IOM.encode_tuple_to_column(("gen1", "area1")) == "gen1__area1"
        @test IOM.encode_tuple_to_column(("name", 42)) == "name__42"
    end

    @testset "container_spec" begin
        # Test Float64 container (should be NaN-filled)
        cont = IOM.container_spec(Float64, ["a", "b"], 1:3)
        @test size(cont) == (2, 3)
        @test all(isnan.(cont.data))

        # Test other type container
        cont2 = IOM.container_spec(Int, ["x", "y"], 1:2)
        @test size(cont2) == (2, 2)
    end

    @testset "sparse_container_spec - Number" begin
        sparse = IOM.sparse_container_spec(Float64, ["a", "b"], 1:2)
        @test length(sparse.data) == 4
        @test all(v == 0.0 for v in values(sparse.data))
    end

    @testset "remove_undef!" begin
        # Note: For Float64 arrays, isassigned() always returns true (even for garbage values).
        # The remove_undef! function is designed for arrays of reference types (e.g., JuMP.AffExpr)
        # where elements can actually be unassigned.
        #
        # This test verifies the function doesn't fail on Float64 arrays and preserves set values.
        data = DenseAxisArray{Float64}(undef, ["a", "b"], 1:2)
        data["a", 1] = 1.0
        data["a", 2] = 2.0
        data["b", 1] = 3.0
        data["b", 2] = 4.0

        IOM.remove_undef!(data)
        # Values should be preserved
        @test data["a", 1] == 1.0
        @test data["a", 2] == 2.0
        @test data["b", 1] == 3.0
        @test data["b", 2] == 4.0
    end

    @testset "remove_undef! - SparseAxisArray passthrough" begin
        contents = Dict(("a", 1) => 1.0)
        sparse = SparseAxisArray(contents)
        output = IOM.remove_undef!(sparse)
        @test output === sparse
    end

    @testset "supports_milp" begin
        # Create a model with a solver that supports MILP
        model = JuMP.Model()
        # Without optimizer, this should still work (returns based on backend)
        output = IOM.supports_milp(model)
        @test isa(output, Bool)
    end

    @testset "to_dataframe - DenseAxisArray 2D" begin
        data = DenseAxisArray(
            [1.0 2.0 3.0; 4.0 5.0 6.0],
            ["gen1", "gen2"],
            1:3,
        )
        df = IOM.to_dataframe(data)
        @test isa(df, DataFrame)
        @test size(df) == (3, 2)
        @test names(df) == ["gen1", "gen2"]
    end

    @testset "to_outputs_dataframe - 2D LONG format with timestamps" begin
        data = DenseAxisArray(
            [1.0 2.0; 3.0 4.0],
            ["gen1", "gen2"],
            1:2,
        )
        timestamps = [DateTime(2024, 1, 1, 0), DateTime(2024, 1, 1, 1)]
        df = IOM.to_outputs_dataframe(data, timestamps, Val(IOM.TableFormat.LONG))
        @test isa(df, DataFrame)
        @test :DateTime in propertynames(df)
        @test :name in propertynames(df)
        @test :value in propertynames(df)
        @test nrow(df) == 4
    end

    @testset "to_outputs_dataframe - 2D LONG format without timestamps" begin
        data = DenseAxisArray(
            [1.0 2.0; 3.0 4.0],
            ["gen1", "gen2"],
            1:2,
        )
        df = IOM.to_outputs_dataframe(data, nothing, Val(IOM.TableFormat.LONG))
        @test isa(df, DataFrame)
        @test :time_index in propertynames(df)
        @test :name in propertynames(df)
        @test :value in propertynames(df)
    end

    @testset "to_outputs_dataframe - 2D WIDE format with timestamps" begin
        data = DenseAxisArray(
            [1.0 2.0; 3.0 4.0],
            ["gen1", "gen2"],
            1:2,
        )
        timestamps = [DateTime(2024, 1, 1, 0), DateTime(2024, 1, 1, 1)]
        df = IOM.to_outputs_dataframe(data, timestamps, Val(IOM.TableFormat.WIDE))
        @test isa(df, DataFrame)
        @test :DateTime in propertynames(df)
        @test Symbol("gen1") in propertynames(df)
        @test Symbol("gen2") in propertynames(df)
        @test nrow(df) == 2
    end

    @testset "to_outputs_dataframe - 3D LONG format" begin
        data = DenseAxisArray(
            zeros(2, 2, 3),
            ["gen1", "gen2"],
            ["area1", "area2"],
            1:3,
        )
        data["gen1", "area1", 1] = 1.0
        data["gen2", "area2", 2] = 2.0

        timestamps = [DateTime(2024, 1, 1, i) for i in 0:2]
        df = IOM.to_outputs_dataframe(data, timestamps, Val(IOM.TableFormat.LONG))
        @test isa(df, DataFrame)
        @test :DateTime in propertynames(df)
        @test :name in propertynames(df)
        @test :name2 in propertynames(df)
        @test :value in propertynames(df)
        @test nrow(df) == 12  # 2 * 2 * 3
    end

    @testset "to_outputs_dataframe - 3D LONG format (String, Int, Int)" begin
        # Integer second axis — the DecisionModelStore (Vector{String}, UnitRange,
        # UnitRange) case that previously hit a MethodError with timestamps.
        data = DenseAxisArray(
            zeros(2, 2, 3),
            ["gen1", "gen2"],
            1:2,
            1:3,
        )
        data["gen1", 1, 1] = 1.0
        data["gen2", 2, 2] = 2.0

        timestamps = [DateTime(2024, 1, 1, i) for i in 0:2]
        df = IOM.to_outputs_dataframe(data, timestamps, Val(IOM.TableFormat.LONG))
        @test isa(df, DataFrame)
        @test :DateTime in propertynames(df)
        @test :name in propertynames(df)
        @test :name2 in propertynames(df)
        @test eltype(df.name2) == Int
        @test :value in propertynames(df)
        @test nrow(df) == 12  # 2 * 2 * 3
        @test sum(df.value) == 3.0
    end

    @testset "to_outputs_dataframe - 1D LONG (Task 2.2)" begin
        data = DenseAxisArray([10.0, 20.0, 30.0], ["g1", "g2", "g3"])
        # Without timestamps: one row per component, time_index column all 1.
        df = IOM.to_outputs_dataframe(data, nothing, Val(IOM.TableFormat.LONG))
        @test nrow(df) == 3
        @test :time_index in propertynames(df)
        @test df.value == [10.0, 20.0, 30.0]
        # With a single timestamp: DateTime column filled, multi-name no longer errors.
        ts = [DateTime(2024, 1, 1)]
        df2 = IOM.to_outputs_dataframe(data, ts, Val(IOM.TableFormat.LONG))
        @test nrow(df2) == 3
        @test all(df2.DateTime .== DateTime(2024, 1, 1))
        @test df2.name == ["g1", "g2", "g3"]
    end

    @testset "to_outputs_dataframe - 2D LONG offset time axis (Task 2.2)" begin
        # Time axis 0:1 (not 1-based) must index timestamps positionally, not by value.
        data = DenseAxisArray([1.0 2.0; 3.0 4.0], ["g1", "g2"], 0:1)
        ts = [DateTime(2024, 1, 1, 0), DateTime(2024, 1, 1, 1)]
        df = IOM.to_outputs_dataframe(data, ts, Val(IOM.TableFormat.LONG))
        @test nrow(df) == 4
        @test sort(unique(df.DateTime)) == ts
    end

    @testset "find_timestamp_index (Task 2.18)" begin
        dates = [DateTime(2024, 1, 1, h) for h in 0:3]
        @test IOM.find_timestamp_index(dates, DateTime(2024, 1, 1, 0)) == 1
        @test IOM.find_timestamp_index(dates, DateTime(2024, 1, 1, 2)) == 3
        @test IOM.find_timestamp_index(dates, DateTime(2024, 1, 1, 3)) == 4
        # Off-grid timestamp (30 min past) must error, not silently floor.
        @test_throws ErrorException IOM.find_timestamp_index(
            dates,
            DateTime(2024, 1, 1, 1, 30),
        )
        # Length-1 vector with a non-matching date must error, not BoundsError.
        @test_throws ErrorException IOM.find_timestamp_index(
            [DateTime(2024, 1, 1)],
            DateTime(2024, 1, 2),
        )
    end
end
