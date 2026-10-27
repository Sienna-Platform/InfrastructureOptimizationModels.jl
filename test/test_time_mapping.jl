"""
Unit tests for the time mapping abstraction (core/time_mapping.jl): the generic functions on
`AbstractTimeMapping`, `validate_time_mapping`, and `NestedTimeMapping`.
"""

using Test
using InfrastructureOptimizationModels
using Dates

@testset "Time mapping" begin
    @testset "Generic functions on a multi-block mapping" begin
        tm = build_mock_invest_time_mapping()

        @test IOM.get_time_steps(tm) == 1:8
        @test IOM.get_periods(tm) == 1:2
        @test IOM.get_blocks(tm, 1) == [1, 2]
        @test IOM.get_blocks(tm, 2) == [3, 4]
        @test IOM.get_period_time_steps(tm, 1) == [1, 2, 3, 4]
        @test IOM.get_period_time_steps(tm, 2) == [5, 6, 7, 8]
        @test IOM.get_period_index(tm, 1) == 1
        @test IOM.get_period_index(tm, 4) == 1
        @test IOM.get_period_index(tm, 5) == 2
        @test_throws ErrorException IOM.get_period_index(tm, 9)
        @test IOM.get_last_time_step(tm, 1) == 4
        @test IOM.get_last_time_step(tm, 2) == 8
        @test IOM.get_block_weights(tm) == ones(4)
        @test IOM.validate_time_mapping(tm) === nothing
    end

    @testset "Empty mapping" begin
        tm = build_empty_mock_invest_time_mapping()
        @test IOM.get_time_steps(tm) == 1:0
        @test IOM.get_periods(tm) == 1:0
        @test isempty(IOM.get_block_weights(tm))
        @test IOM.validate_time_mapping(tm) === nothing
    end

    @testset "validate_time_mapping rejects malformed mappings" begin
        ts = [DateTime(2030, 1, 1, h) for h in 0:3]
        mock(ranges, index; periods = 2) =
            MockInvestTimeMapping(Date(2030, 1, 1), ts, ranges, index, periods, 4)

        # gap between blocks
        @test_throws r"must be non-empty and start at time step 3" IOM.validate_time_mapping(
            mock([1:2, 4:4], [1, 2]),
        )
        # overlapping blocks
        @test_throws r"start at time step 3" IOM.validate_time_mapping(
            mock([1:2, 2:4], [1, 2]),
        )
        # blocks do not reach the last time stamp
        @test_throws r"cover 3 time steps" IOM.validate_time_mapping(
            mock([1:2, 3:3], [1, 2]),
        )
        # block in a period that does not exist
        @test_throws r"belongs to period 3" IOM.validate_time_mapping(
            mock([1:2, 3:4], [1, 3]),
        )
        # one period index per block
        @test_throws r"2 blocks but 1 block period indexes" IOM.validate_time_mapping(
            mock([1:2, 3:4], [1]),
        )
    end

    @testset "NestedTimeMapping" begin
        start = DateTime(2030, 1, 1)
        tm = IOM.NestedTimeMapping(start, Hour(1), 1:48, Day(1))

        @test tm isa IOM.AbstractTimeMapping
        @test IOM.get_period_count(tm) == 2
        @test IOM.get_block_ranges(tm) == [1:24, 25:48]
        @test IOM.get_block_period_index(tm) == [1, 2]
        @test IOM.get_blocks(tm, 2) == [2]
        @test IOM.get_period_time_steps(tm, 1) == collect(1:24)
        @test IOM.get_period_index(tm, 25) == 2
        @test IOM.get_last_time_step(tm, 1) == 24
        @test first(IOM.get_timestamps(tm)) == start
        @test last(IOM.get_timestamps(tm)) == start + Hour(47)
        @test IOM.get_time_steps(tm) == 1:48
        @test IOM.validate_time_mapping(tm) === nothing

        weekly = IOM.NestedTimeMapping(start, Hour(4), 1:84, Week(1))
        @test IOM.get_block_ranges(weekly) == [1:42, 43:84]

        @test_throws r"allow_partial" IOM.NestedTimeMapping(start, Hour(1), 1:30, Day(1))
        partial =
            IOM.NestedTimeMapping(start, Hour(1), 1:30, Day(1); allow_partial = true)
        @test IOM.get_block_ranges(partial) == [1:24, 25:30]
        @test IOM.get_last_time_step(partial, 2) == 30
        @test IOM.validate_time_mapping(partial) === nothing

        @test_throws r"not a positive multiple" IOM.NestedTimeMapping(
            start,
            Hour(5),
            1:24,
            Day(1),
        )
    end

    @testset "Container accepts any AbstractTimeMapping" begin
        sys = MockSystem(100.0)
        settings = IOM.Settings(
            sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)
        @test IOM.get_time_mapping(container) === nothing

        nested = IOM.NestedTimeMapping(DateTime(2030, 1, 1), Hour(1), 1:24, Day(1))
        IOM.set_time_mapping!(container, nested)
        @test IOM.get_time_mapping(container) === nested

        invest = build_mock_invest_time_mapping()
        IOM.set_time_mapping!(container, invest)
        @test IOM.get_time_mapping(container) === invest
    end
end
