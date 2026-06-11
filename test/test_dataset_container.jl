"""
Unit tests for DatasetContainer in dataset_container.jl.
Tests container operations for storing optimization outputs.
Written entirely by Claude Code, so fairly surface-level, but serves its purpose of
confirming that the code runs without errors.
"""
# Test types are defined in test_utils/test_types.jl

@testset "DatasetContainer" begin
    @testset "Constructor" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()
        @test isa(container, IOM.DatasetContainer{IOM.InMemoryDataset})
        @test isempty(container.duals)
        @test isempty(container.aux_variables)
        @test isempty(container.variables)
        @test isempty(container.parameters)
        @test isempty(container.expressions)
    end

    @testset "set_dataset! and get_dataset" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        # Create a simple InMemoryDataset
        values = DenseAxisArray(
            fill(0.0, 2, 3),
            ["gen1", "gen2"],
            1:3,
        )
        dataset = IOM.InMemoryDataset(values)

        # Test with VariableKey
        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)
        @test IOM.has_dataset(container, var_key)
        @test IOM.get_dataset(container, var_key) === dataset

        # Test with ConstraintKey
        con_key = IOM.ConstraintKey(TestConstraintType, IS.TestComponent)
        IOM.set_dataset!(container, con_key, dataset)
        @test IOM.has_dataset(container, con_key)

        # Test with AuxVarKey
        aux_key = IOM.AuxVarKey(TestAuxVariableType, IS.TestComponent)
        IOM.set_dataset!(container, aux_key, dataset)
        @test IOM.has_dataset(container, aux_key)

        # Test with ParameterKey
        param_key = IOM.ParameterKey(TestParameterType, IS.TestComponent)
        IOM.set_dataset!(container, param_key, dataset)
        @test IOM.has_dataset(container, param_key)

        # Test with ExpressionKey
        expr_key = IOM.ExpressionKey(TestExpressionType, IS.TestComponent)
        IOM.set_dataset!(container, expr_key, dataset)
        @test IOM.has_dataset(container, expr_key)
    end

    @testset "has_dataset - missing key" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()
        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        @test !IOM.has_dataset(container, var_key)
    end

    @testset "get_dataset with type dispatch" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        values = DenseAxisArray(fill(1.0, 2, 3), ["a", "b"], 1:3)
        dataset = IOM.InMemoryDataset(values)

        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)

        # Get using type dispatch (instance of type + Type parameter)
        retrieved = IOM.get_dataset(container, TestVariableType, IS.TestComponent)
        @test retrieved === dataset
    end

    @testset "get_dataset_keys" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        values = DenseAxisArray(fill(0.0, 2, 3), ["a", "b"], 1:3)
        dataset = IOM.InMemoryDataset(values)

        var_key1 = IOM.VariableKey(TestVariableType, IS.TestComponent)
        var_key2 = IOM.VariableKey(TestVariableType2, IS.TestComponent)
        con_key = IOM.ConstraintKey(TestConstraintType, IS.TestComponent)

        IOM.set_dataset!(container, var_key1, dataset)
        IOM.set_dataset!(container, var_key2, dataset)
        IOM.set_dataset!(container, con_key, dataset)

        all_keys = collect(IOM.get_dataset_keys(container))
        @test length(all_keys) == 3
        @test var_key1 in all_keys
        @test var_key2 in all_keys
        @test con_key in all_keys
    end

    @testset "get_dataset_values" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        values = DenseAxisArray(
            [1.0 2.0 3.0; 4.0 5.0 6.0],
            ["gen1", "gen2"],
            1:3,
        )
        dataset = IOM.InMemoryDataset(values)

        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)

        retrieved_values = IOM.get_dataset_values(container, var_key)
        @test retrieved_values === values
    end

    @testset "get_*_values for InMemoryDataset" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        values = DenseAxisArray(fill(0.0, 2, 3), ["a", "b"], 1:3)
        dataset = IOM.InMemoryDataset(values)

        # Add datasets to each category
        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        con_key = IOM.ConstraintKey(TestConstraintType, IS.TestComponent)
        aux_key = IOM.AuxVarKey(TestAuxVariableType, IS.TestComponent)
        param_key = IOM.ParameterKey(TestParameterType, IS.TestComponent)
        expr_key = IOM.ExpressionKey(TestExpressionType, IS.TestComponent)

        IOM.set_dataset!(container, var_key, dataset)
        IOM.set_dataset!(container, con_key, dataset)
        IOM.set_dataset!(container, aux_key, dataset)
        IOM.set_dataset!(container, param_key, dataset)
        IOM.set_dataset!(container, expr_key, dataset)

        # Test accessor functions
        @test haskey(IOM.get_variables_values(container), var_key)
        @test haskey(IOM.get_duals_values(container), con_key)
        @test haskey(IOM.get_aux_variables_values(container), aux_key)
        @test haskey(IOM.get_parameters_values(container), param_key)
        @test haskey(IOM.get_expression_values(container), expr_key)
    end

    @testset "get_last_recorded_row" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        initial_time = DateTime(2024, 1, 1)
        resolution = Dates.Hour(1)
        values = DenseAxisArray(fill(0.0, 2, 24), ["gen1", "gen2"], 1:24)
        timestamps = collect(range(initial_time; step = resolution, length = 24))
        dataset = IOM.InMemoryDataset(values, timestamps, Dates.Millisecond(resolution), 1)

        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)

        # Initially 0
        @test IOM.get_last_recorded_row(container, var_key) == 0

        # Update the dataset
        IOM.set_last_recorded_row!(dataset, 5)
        @test IOM.get_last_recorded_row(container, var_key) == 5
    end

    @testset "get_update_timestamp and get_last_updated_timestamp" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        initial_time = DateTime(2024, 1, 1)
        resolution = Dates.Hour(1)
        values = DenseAxisArray(fill(0.0, 2, 24), ["gen1", "gen2"], 1:24)
        timestamps = collect(range(initial_time; step = resolution, length = 24))
        dataset = IOM.InMemoryDataset(values, timestamps, Dates.Millisecond(resolution), 1)

        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)

        # Test update_timestamp
        @test IOM.get_update_timestamp(container, var_key) == IOM.UNSET_INI_TIME

        update_time = DateTime(2024, 1, 1, 5)
        IOM.set_update_timestamp!(dataset, update_time)
        @test IOM.get_update_timestamp(container, var_key) == update_time

        # Test last_updated_timestamp (based on last_recorded_row)
        # When last_recorded_row is 0, returns UNSET_INI_TIME
        @test IOM.get_last_updated_timestamp(container, var_key) == IOM.UNSET_INI_TIME

        IOM.set_last_recorded_row!(dataset, 3)
        @test IOM.get_last_updated_timestamp(container, var_key) == timestamps[3]
    end

    @testset "set_dataset_values!" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()

        initial_time = DateTime(2024, 1, 1)
        resolution = Dates.Hour(1)
        values = DenseAxisArray(fill(0.0, 2, 5), ["gen1", "gen2"], 1:5)
        timestamps = collect(range(initial_time; step = resolution, length = 5))
        dataset = IOM.InMemoryDataset(values, timestamps, Dates.Millisecond(resolution), 1)

        var_key = IOM.VariableKey(TestVariableType, IS.TestComponent)
        IOM.set_dataset!(container, var_key, dataset)

        # Set values at index 2
        new_vals = DenseAxisArray([10.0, 20.0], ["gen1", "gen2"])
        IOM.set_dataset_values!(container, var_key, 2, new_vals)

        # Verify the values were set
        @test dataset.values["gen1", 2] == 10.0
        @test dataset.values["gen2", 2] == 20.0
    end

    @testset "empty! resets all dataset dicts" begin
        container = IOM.DatasetContainer{IOM.InMemoryDataset}()
        var_key = IOM.VariableKey(MockVariable, MockComponentType)
        data = DenseAxisArray(zeros(2, 3), ["gen1", "gen2"], 1:3)
        IOM.set_dataset!(container, var_key, IOM.InMemoryDataset(data))
        @test IOM.has_dataset(container, var_key)
        empty!(container)
        @test !IOM.has_dataset(container, var_key)
    end
end
