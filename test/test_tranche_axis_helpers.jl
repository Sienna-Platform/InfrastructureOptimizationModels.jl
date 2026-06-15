"""
Unit tests for the PSY-free tranche-axis helpers used to size and fill the 3-D
`(component, tranche, time)` PWL parameter arrays for time-varying ORDC / market-bid
costs. Helpers live in `src/objective_function/value_curve_cost.jl` (Section 6).
"""

@testset "Tranche-axis helpers (time-varying PWL)" begin
    @testset "make_tranche_axis" begin
        @test IOM.make_tranche_axis(3) == ["tranche_1", "tranche_2", "tranche_3"]
        @test IOM.make_tranche_axis(1) == ["tranche_1"]
        @test isempty(IOM.make_tranche_axis(0))
    end

    @testset "lookup_additional_axes" begin
        # (component, tranche, time) -> only the middle (tranche) axis is "additional"
        arr3 = DenseAxisArray(zeros(1, 2, 4), ["g1"], ["tranche_1", "tranche_2"], 1:4)
        @test IOM.lookup_additional_axes(arr3) == (["tranche_1", "tranche_2"],)

        # (component, time) -> no additional axes
        arr2 = DenseAxisArray(zeros(1, 4), ["g1"], 1:4)
        @test IOM.lookup_additional_axes(arr2) == ()
    end

    @testset "get_max_tranches" begin
        # length(PiecewiseStepData) == number of segments == length(x_coords) - 1
        a = IS.PiecewiseStepData([0.0, 1.0, 2.0], [10.0, 20.0])        # 2 tranches
        b = IS.PiecewiseStepData([0.0, 1.0, 2.0, 3.0], [5.0, 6.0, 7.0]) # 3 tranches

        # Vector method: global maximum over time
        @test IOM.get_max_tranches([a, b]) == 3
        @test IOM.get_max_tranches([a]) == 2

        # TimeArray method: unwraps to its values and reuses the Vector method
        ts = IOM.TS.TimeArray([DateTime(2024, 1, 1), DateTime(2024, 1, 1, 1)], [a, b])
        @test IOM.get_max_tranches(ts) == 3

        # Dict method: global maximum across all entries
        @test IOM.get_max_tranches(Dict("x" => [a], "y" => [a, b])) == 3
    end

    @testset "unwrap_for_param" begin
        # 2 segments / 3 breakpoints
        psd = IS.PiecewiseStepData([0.0, 1.0, 2.0], [10.0, 20.0])

        # default fallback: any non-PWL parameter passes its element through unchanged
        @test IOM.unwrap_for_param(TestParameterType(), 3.0, (1:5,)) == 3.0

        @testset "slope parameter (pads y-coords with 0.0)" begin
            slope_param = IOM.DecrementalPiecewiseLinearSlopeParameter()
            # exact length: no padding
            @test IOM.unwrap_for_param(slope_param, psd, (IOM.make_tranche_axis(2),)) ==
                  [10.0, 20.0]
            # short curve: pad slope = 0.0 up to the tranche-axis length
            @test IOM.unwrap_for_param(slope_param, psd, (IOM.make_tranche_axis(4),)) ==
                  [10.0, 20.0, 0.0, 0.0]
            # too many coords for the axis: error
            @test_throws ErrorException IOM.unwrap_for_param(
                slope_param, psd, (IOM.make_tranche_axis(1),))
        end

        @testset "breakpoint parameter (repeats last breakpoint)" begin
            bp_param = IOM.DecrementalPiecewiseLinearBreakpointParameter()
            # exact length (tranches + 1): no padding
            @test IOM.unwrap_for_param(bp_param, psd, (IOM.make_tranche_axis(3),)) ==
                  [0.0, 1.0, 2.0]
            # short curve: pad by repeating the last breakpoint (so dx = 0)
            @test IOM.unwrap_for_param(bp_param, psd, (IOM.make_tranche_axis(5),)) ==
                  [0.0, 1.0, 2.0, 2.0, 2.0]
            # too many coords for the axis: error
            @test_throws ErrorException IOM.unwrap_for_param(
                bp_param, psd, (IOM.make_tranche_axis(2),))
        end
    end
end
