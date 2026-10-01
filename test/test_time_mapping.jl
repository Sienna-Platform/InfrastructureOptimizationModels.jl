"""
Unit tests for the TimeMapping time abstraction (core/time_mapping.jl).
Pure-logic tests — no mocks required.
"""

using Test
using InfrastructureOptimizationModels
using Dates

@testset "TimeMapping" begin
    @testset "Empty / `nothing` constructor" begin
        tm = IOM.TimeMapping(nothing)
        @test isempty(IOM.get_time_stamps(tm))
        @test isempty(IOM.get_investment_time_stamps(tm))
        @test IOM.get_total_period_count(tm) == 0
        @test IOM.get_total_investment_period_count(tm) == 0
        @test IOM.get_time_steps(tm) == 1:0
        @test IOM.get_investment_time_steps(tm) == 1:0
        @test IOM.is_feasibility_empty(tm)
        @test isempty(IOM.get_operational_indexes(tm))
        @test isempty(IOM.get_feasibility_indexes(tm))
        @test isempty(IOM.get_all_indexes(tm))
    end

    @testset "Two investment intervals, no feasibility" begin
        invest = [
            (Date(2030, 1, 1), Date(2030, 12, 31)),
            (Date(2040, 1, 1), Date(2040, 12, 31)),
        ]
        ops = [
            [DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1)],
            [DateTime(2040, 1, 1, 0), DateTime(2040, 1, 1, 1)],
        ]
        feas = Vector{Vector{DateTime}}()
        tm = IOM.TimeMapping(invest, ops, feas)

        @test IOM.get_base_date(tm) == Date(2030, 1, 1)
        @test IOM.get_investment_time_stamps(tm) == invest
        @test IOM.get_total_investment_period_count(tm) == 2
        @test IOM.get_investment_time_steps(tm) == 1:2

        # 2 slices x 2 timestamps = 4 operational timestamps
        @test IOM.get_total_period_count(tm) == 4
        @test length(IOM.get_time_stamps(tm)) == 4
        @test IOM.get_time_steps(tm) == 1:4
        @test IOM.get_total_operation_period_count(tm) == 4
        @test IOM.get_operational_time_steps(tm) == 1:4

        @test IOM.get_operational_indexes(tm) == [1, 2]
        @test IOM.is_feasibility_empty(tm)
        @test isempty(IOM.get_feasibility_indexes(tm))
        @test IOM.get_all_indexes(tm) == [1, 2]

        # consecutive (1-based) period indices per slice
        @test IOM.get_consecutive_slices(tm) == [[1, 2], [3, 4]]

        # each slice maps back to its investment interval
        @test IOM.get_inverse_invest_mapping(tm) == [1, 2]

        invest_map = IOM.get_investment_map_to_operational_slices(tm)
        @test invest_map[1] == [1]
        @test invest_map[2] == [2]
    end

    @testset "With feasibility periods" begin
        invest = [(Date(2030, 1, 1), Date(2030, 12, 31))]
        ops = [[DateTime(2030, 1, 1, 0), DateTime(2030, 1, 1, 1)]]
        feas = [[DateTime(2030, 6, 1, 0)]]
        tm = IOM.TimeMapping(invest, ops, feas)

        @test !IOM.is_feasibility_empty(tm)
        @test IOM.get_operational_indexes(tm) == [1]
        @test IOM.get_feasibility_indexes(tm) == [2]       # slice 2 is the feasibility slice
        @test IOM.get_all_indexes(tm) == [1, 2]
        @test IOM.get_total_operation_period_count(tm) == 2   # 1 op slice x 2 ts
        @test IOM.get_total_feasibility_period_count(tm) == 3 # + 1 feasibility ts
        @test IOM.get_feasibility_time_steps(tm) == 3:3
    end

    @testset "Slice outside all investment intervals errors" begin
        invest = [(Date(2030, 1, 1), Date(2030, 12, 31))]
        ops = [[DateTime(2050, 1, 1, 0)]]  # not contained in any interval
        feas = Vector{Vector{DateTime}}()
        @test_throws Exception IOM.TimeMapping(invest, ops, feas)
    end
end
