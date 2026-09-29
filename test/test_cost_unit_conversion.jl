"""
Unit tests for the cost-coefficient base conversions in src/utils/component_utils.jl.
Cost curves are in natural units; the helpers normalize them to the system base.
"""

@testset "Cost coefficient conversion to system base" begin
    system_base = 100.0

    @testset "proportional term" begin
        # \$/MW -> \$/(system pu MW): one system pu is 100 MW
        @test IOM.get_proportional_cost_per_system_unit(3.0, system_base) == 300.0
    end

    @testset "quadratic term scales with the square of the base" begin
        @test IOM.get_quadratic_cost_per_system_unit(2.0, system_base) == 2.0e4
    end

    @testset "point curve rescales x only" begin
        curve = IS.PiecewiseLinearData([(x = 0.0, y = 0.0), (x = 20.0, y = 100.0)])
        normalized = IOM.get_piecewise_pointcurve_per_system_unit(curve, system_base)
        @test IS.get_x_coords(normalized) == [0.0, 0.2]
        @test IS.get_y_coords(normalized) == [0.0, 100.0]
    end

    @testset "step curve rescales x down and y up" begin
        curve = IS.PiecewiseStepData([0.0, 20.0], [5.0])
        normalized = IOM.get_piecewise_curve_per_system_unit(curve, system_base)
        @test IS.get_x_coords(normalized) == [0.0, 0.2]
        @test IS.get_y_coords(normalized) == [500.0]
    end
end
