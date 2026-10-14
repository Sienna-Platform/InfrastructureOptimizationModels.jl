"""
Unit tests for InvestmentModelStore (investments/investment_model_store.jl):
the empty constructor, operation/investment entry classification, optimizer-stats
round-trip, and `initialize_storage!` row-count sizing from the TimeMapping.
Uses mocks — no PowerSystems dependency.
"""

using Test
using InfrastructureOptimizationModels
using Dates
using InfrastructureSystems
import JuMP.Containers: DenseAxisArray
const IS = InfrastructureSystems

# Concrete operation/investment entry types for classification + storage sizing.
struct MockStoreInvestVar <: IOM.InvestmentVariableType end
struct MockStoreOpsVar <: IOM.OperationsVariableType end
struct MockStoreInvestExpr <: IOM.InvestmentExpressionType end
struct MockStoreOpsExpr <: IOM.OperationsExpressionType end

@testset "InvestmentModelStore" begin
    @testset "empty constructor" begin
        store = IOM.InvestmentModelStore()
        @test isempty(store.variables)
        @test isempty(store.duals)
        @test isempty(store.aux_variables)
        @test isempty(store.expressions)
        @test isempty(store.optimizer_stats)
    end

    @testset "operation vs investment entry classification" begin
        @test IOM.is_operation_entry(MockStoreOpsVar)
        @test !IOM.is_investment_entry(MockStoreOpsVar)
        @test IOM.is_investment_entry(MockStoreInvestVar)
        @test !IOM.is_operation_entry(MockStoreInvestVar)

        @test IOM.is_operation_entry(MockStoreOpsExpr)
        @test !IOM.is_investment_entry(MockStoreOpsExpr)
        @test IOM.is_investment_entry(MockStoreInvestExpr)
        @test !IOM.is_operation_entry(MockStoreInvestExpr)

        # The abstract VariableType base is ambiguous and must error.
        @test_throws Exception IOM.is_operation_entry(IOM.VariableType)
        @test_throws Exception IOM.is_investment_entry(IOM.VariableType)
    end

    @testset "optimizer stats round-trip" begin
        store = IOM.InvestmentModelStore()
        stats = IOM.OptimizerStats()
        IOM.write_optimizer_stats!(store, stats, Date(2030, 1, 1))
        df = IOM.read_optimizer_stats(store)
        @test size(df, 1) == 1
        @test :DateTime in propertynames(df)
        # Writing the same index warns and overwrites in place (count stays at one row).
        @test_logs (:warn, r"Overwriting optimizer stats") IOM.write_optimizer_stats!(
            store,
            stats,
            Date(2030, 1, 1),
        )
        @test size(IOM.read_optimizer_stats(store), 1) == 1
    end

    @testset "initialize_storage! sizes rows by entry type" begin
        sys = MockSystem(100.0)
        settings = IOM.Settings(
            sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)

        # 1 investment interval, 1 operational slice of 2 timestamps, no feasibility
        #   => operation period count = 2, investment period count = 1
        tmap = IOM.TimeMapping(
            [(Date(2030, 1, 1), Date(2030, 12, 31))],
            [[DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1)]],
            Vector{Vector{DateTime}}(),
        )
        IOM.set_time_mapping!(container, tmap)

        cols = ["c1", "c2"]
        ops_key = IOM.VariableKey(MockStoreOpsVar, MockComponentType)
        inv_key = IOM.VariableKey(MockStoreInvestVar, MockComponentType)
        IOM.get_variables(container)[ops_key] = DenseAxisArray(zeros(2, 2), cols, 1:2)
        IOM.get_variables(container)[inv_key] = DenseAxisArray(zeros(2, 1), cols, 1:1)

        store = IOM.InvestmentModelStore()
        params = IOM.ModelStoreParams(
            1,
            1,
            Dates.Millisecond(Dates.Hour(1)),
            Dates.Millisecond(Dates.Hour(1)),
            100.0,
            Base.UUID(UInt128(0)),
        )
        IOM.initialize_storage!(store, container, params)

        @test haskey(store.variables, ops_key)
        @test haskey(store.variables, inv_key)

        base_ts = DateTime(2030, 1, 1)  # get_base_date -> Date, stored as DateTime key
        # operations variable gets operation-period rows (2); investment variable gets
        # investment-period rows (1). Columns match the injected axis in both cases.
        @test size(store.variables[ops_key][base_ts]) == (2, 2)
        @test size(store.variables[inv_key][base_ts]) == (2, 1)

        # get_column_names on the store resolves through the canonical helper.
        @test IOM.get_column_names(store, ops_key) == (cols,)
    end

    @testset "write_output! / read_outputs round-trip" begin
        store = IOM.InvestmentModelStore()
        ts = DateTime(2030, 1, 1)
        ops_key = IOM.VariableKey(MockStoreOpsVar, MockComponentType)
        inv_key = IOM.VariableKey(MockStoreInvestVar, MockComponentType)
        for key in (ops_key, inv_key)
            store.variables[key] = IOM.OrderedDict{DateTime, DenseAxisArray{Float64, 2}}()
        end

        # 2-D outputs keep their shape; integer columns are stored as strings.
        IOM.write_output!(
            store,
            :model,
            ops_key,
            ts,
            ts,
            DenseAxisArray([1.0 2.0; 3.0 4.0], [10, 20], 1:2),
        )
        ops = IOM.read_outputs(store, ops_key)
        @test axes(ops) == (["10", "20"], 1:2)
        @test ops.data == [1.0 2.0; 3.0 4.0]

        # 1-D outputs become a single row named "1" with one column per entry.
        IOM.write_output!(
            store,
            :model,
            inv_key,
            ts,
            ts,
            DenseAxisArray([5.0, 6.0], ["a", "b"]),
        )
        inv = IOM.read_outputs(store, inv_key; index = ts)
        @test axes(inv) == (["1"], ["a", "b"])
        @test inv.data == [5.0 6.0]

        # read_outputs returns a copy.
        inv.data[1, 1] = -1.0
        @test IOM.read_outputs(store, inv_key).data == [5.0 6.0]
    end

    @testset "initialize_storage! errors when time steps are undefined" begin
        sys = MockSystem(100.0)
        settings = IOM.Settings(
            sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)
        IOM.set_time_mapping!(container, IOM.TimeMapping())  # empty mapping
        store = IOM.InvestmentModelStore()
        params = IOM.ModelStoreParams(
            1,
            1,
            Dates.Millisecond(Dates.Hour(1)),
            Dates.Millisecond(Dates.Hour(1)),
            100.0,
            Base.UUID(UInt128(0)),
        )
        @test_throws ErrorException IOM.initialize_storage!(store, container, params)
    end
end
