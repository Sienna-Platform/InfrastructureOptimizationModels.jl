"""
Unit tests for the cost-coefficient base conversions in src/utils/component_utils.jl.
`IS.convert_cost_coefficient` takes a bare x-axis ratio; the helpers derive it from the
curve's unit system and the two base powers.
"""

@testset "Cost coefficient conversion to system base" begin
    system_base = 100.0
    device_base = 20.0

    @testset "proportional term" begin
        @test IOM.get_proportional_cost_per_system_unit(
            3.0,
            IS.SU,
            system_base,
            device_base,
        ) == 3.0
        # \$/MW -> \$/(system pu MW): one system pu is 100 MW
        @test IOM.get_proportional_cost_per_system_unit(
            3.0,
            IS.NU,
            system_base,
            device_base,
        ) == 300.0
        # \$/(device pu MW) -> \$/(system pu MW): one system pu is five device pu
        @test IOM.get_proportional_cost_per_system_unit(
            3.0,
            IS.CU,
            system_base,
            device_base,
        ) == 15.0
    end

    @testset "quadratic term scales with the square of the ratio" begin
        @test IOM.get_quadratic_cost_per_system_unit(
            2.0,
            IS.SU,
            system_base,
            device_base,
        ) == 2.0
        @test IOM.get_quadratic_cost_per_system_unit(
            2.0,
            IS.NU,
            system_base,
            device_base,
        ) == 2.0e4
        @test IOM.get_quadratic_cost_per_system_unit(
            2.0,
            IS.CU,
            system_base,
            device_base,
        ) == 50.0
    end

    @testset "point curve rescales x only" begin
        curve = IS.PiecewiseLinearData([(x = 0.0, y = 0.0), (x = 20.0, y = 100.0)])
        natural = IOM.get_piecewise_pointcurve_per_system_unit(
            curve,
            IS.NU,
            system_base,
            device_base,
        )
        @test IS.get_x_coords(natural) == [0.0, 0.2]
        @test IS.get_y_coords(natural) == [0.0, 100.0]
        device = IOM.get_piecewise_pointcurve_per_system_unit(
            curve,
            IS.CU,
            system_base,
            device_base,
        )
        @test IS.get_x_coords(device) == [0.0, 4.0]
        @test IOM.get_piecewise_pointcurve_per_system_unit(
            curve,
            IS.SU,
            system_base,
            device_base,
        ) === curve
    end

    @testset "step curve rescales x down and y up" begin
        curve = IS.PiecewiseStepData([0.0, 20.0], [5.0])
        natural =
            IOM.get_piecewise_curve_per_system_unit(curve, IS.NU, system_base, device_base)
        @test IS.get_x_coords(natural) == [0.0, 0.2]
        @test IS.get_y_coords(natural) == [500.0]
        device =
            IOM.get_piecewise_curve_per_system_unit(curve, IS.CU, system_base, device_base)
        @test IS.get_x_coords(device) == [0.0, 4.0]
        @test IS.get_y_coords(device) == [25.0]
        @test IOM.get_piecewise_curve_per_system_unit(
            curve,
            IS.SU,
            system_base,
            device_base,
        ) === curve
    end
end
